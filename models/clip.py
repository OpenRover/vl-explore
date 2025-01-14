# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import warnings, torch, open_clip, numpy as np, gc
from multiprocessing.pool import ThreadPool as Pool

import torch.nn.functional as F

from open_clip.model import CLIP
from open_clip.tokenizer import SimpleTokenizer, HFTokenizer
from open_clip.transform import PreprocessCfg
from torchvision.transforms import Compose, Resize, InterpolationMode
from PIL.Image import fromarray

from .__dir__ import DIR
from lib.slicer import Slicer
from util.geometry import Region
from util.env import to_device, Logger

log = Logger(__file__)


class Preprocess(PreprocessCfg):
    def __init__(self, cfg: CLIP):
        assert hasattr(cfg.visual, "preprocess_cfg"), "Visual model is not initialized."
        super().__init__(**cfg.visual.preprocess_cfg)
        itp = InterpolationMode(self.interpolation)
        self.resize = Resize(self.size, interpolation=itp)
        self.STD = torch.tensor(self.std).view(3, 1, 1)
        self.MEAN = torch.tensor(self.mean).view(3, 1, 1)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = self.resize(t)
        t = (t - self.MEAN) / self.STD
        return t

    def to(self, device: str | torch.device):
        self.resize = self.resize.to(device)
        self.MEAN = self.MEAN.to(device)
        self.STD = self.STD.to(device)
        return self


# Define model parameters
name = "ViT-B-32"
pretrained = "laion400m_e32"

# Available after initialization
clip_model: CLIP = None
clip_prep: Compose
preprocess: Preprocess = None
tokenizer: SimpleTokenizer | HFTokenizer = None


# Load the full model
def init(visual: bool = False, text: bool = False):
    global clip_model, clip_prep, preprocess, tokenizer
    log.info("Loading CLIP model", name, "...")
    with warnings.catch_warnings():
        # OpenCLIP calls "torch.load()" without weights_only flag
        warnings.filterwarnings("ignore", category=FutureWarning)
        clip_model, __prep__ = open_clip.create_model_from_pretrained(
            name, pretrained=pretrained, cache_dir=DIR / "weights"
        )
    log.info("CLIP Model", name, "loaded.")
    if visual:
        preprocess = to_device(Preprocess(clip_model))
        clip_prep = __prep__
    else:
        del clip_model.visual
    if text:
        tokenizer = open_clip.get_tokenizer(name)
    else:
        del clip_model.transformer
    to_device(clip_model)


def deinit():
    global clip_model
    clip_model = None
    gc.collect()


def unload_text_model():
    if clip_model is not None:
        del clip_model.transformer
    gc.collect()


def unload_vision_model():
    if clip_model is not None:
        del clip_model.visual
    gc.collect()


@torch.no_grad()
def encode_text(*text: list[str], norm: bool = True) -> torch.Tensor:
    """
    Encode a batch of text into a single tensor
    ---
    Input: N strings of text
    Returns: A tensor of shape (N, 512)
    """
    if clip_model is None or not hasattr(clip_model, "transformer"):
        init(text=True)
    tokens = to_device(tokenizer(text))
    features = clip_model.encode_text(tokens)
    if norm:
        return F.normalize(features, dim=-1)
    else:
        return features


@torch.no_grad()
def _prepare(slicer: Slicer, frame: np.ndarray, threads: int = None) -> torch.Tensor:
    # Original approach
    def pp(tile: np.ndarray) -> torch.Tensor:
        return clip_prep(fromarray(tile))

    if threads is None:
        data = list(map(pp, slicer(frame)))
    else:
        with Pool(threads) as pool:
            data = pool.map(pp, slicer(frame))
    return to_device(torch.stack(data))


@torch.no_grad()
def prepare(
    slicer: Slicer,
    frame: np.ndarray | torch.Tensor,
    threads: int = None,
    check: bool = False,
) -> torch.Tensor:
    # Torch tensor approach
    assert preprocess is not None, "Visual model not initialized."
    if isinstance(frame, np.ndarray):
        t: torch.Tensor = to_device(torch.from_numpy(frame)) / 255.0
    else:
        t = to_device(frame)

    def pp(region: Region) -> torch.Tensor:
        return region(t).permute(2, 0, 1)

    if threads is None or threads <= 1:
        data = list(map(pp, slicer.regions))
    else:
        with Pool(threads) as pool:
            data = pool.map(pp, slicer.regions)

    t = preprocess(torch.stack(data))
    if not check:
        return t
    # Check the tensor
    d1, d2 = _prepare(slicer, frame, threads), t
    assert d1.shape == d2.shape, (d1.shape, d2.shape)
    if not torch.allclose(d1, d2):
        # Dump tensors as images
        from torchvision.transforms import ToPILImage

        def save(t: torch.Tensor, path: str):
            ToPILImage()(t).save(path)

        save(d1[0], "d1.png")
        save(d2[0], "d2.png")
        diff = (d1 - d2) / torch.abs(d1)
        save(0.5 + diff / 2, "diff.png")
        max_diff = torch.max(diff) * 100.0
        avg_diff = torch.mean(d1 - d2) * 100.0
        raise AssertionError(
            f"Preprocess mismatch (max {max_diff:.2}, avg {avg_diff:.2})"
        )


@torch.no_grad()
def encode_image(data: torch.Tensor, norm: bool = True) -> torch.Tensor:
    if clip_model is None or not hasattr(clip_model, "visual"):
        init(visual=True)
    return clip_model.encode_image(data, normalize=norm)
