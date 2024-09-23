# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import warnings, torch, open_clip, numpy as np
from open_clip.model import CLIP
from open_clip.tokenizer import SimpleTokenizer, HFTokenizer
from torchvision.transforms import Compose
from PIL.Image import Image, fromarray

from .__dir__ import DIR
from env import to_device, Logger
from util import norm

log = Logger(__file__)

# Define model parameters
name = "ViT-B-32"
pretrained = "laion400m_e32"

# Available after initialization
text_model: CLIP = None
visual_model: CLIP = None
preprocess: Compose
tokenizer: SimpleTokenizer | HFTokenizer
initialized = False


# Load the full model
def init():
    global preprocess, tokenizer, initialized
    if initialized:
        return
    log.info("Loading CLIP model", name, "...")
    with warnings.catch_warnings():
        # OpenCLIP calls "torch.load()" without weights_only flag
        warnings.filterwarnings("ignore", category=FutureWarning)
        model, preprocess = open_clip.create_model_from_pretrained(
            name, pretrained=pretrained, cache_dir=DIR / "weights"
        )
    tokenizer = open_clip.get_tokenizer(name)
    return model


@torch.no_grad()
def encode_text(*text: list[str]) -> torch.Tensor:
    """
    Encode a batch of text into a single tensor
    ---
    Input: N strings of text
    Returns: A tensor of shape (N, 512)
    """
    global text_model
    if text_model is None:
        text_model = init()
        # Remove visual model
        del text_model.visual
        to_device(text_model)
    # Prepare input tensor of text tokens
    tokens = to_device(tokenizer(text))
    # Encode the text
    return norm(text_model.encode_text(tokens))


@torch.no_grad()
def encode_image(*images: list[np.ndarray | Image]) -> torch.Tensor:
    """
    Encode a batch of images into a single tensor
    ---
    Input: N images in the form of numpy arrays
    Returns: A tensor of shape (N, 512)
    """
    global visual_model
    if visual_model is None:
        visual_model = init()
        # Remove visual model
        del visual_model.transformer
        to_device(visual_model)
    # Preprocess input images
    data: list[torch.Tensor] = []
    for img in images:
        if type(img) is np.ndarray:
            img = fromarray(img)
        elif type(img) is not Image:
            raise ValueError(f"Invalid image type <{type(img)}>.")
        data.append(preprocess(img))
    # Prepare input tensor of visual images
    data = to_device(torch.stack(data))
    # Encode the image
    return norm(visual_model.encode_image(data))
