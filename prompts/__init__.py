# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import torch
from pathlib import Path
from .__util__ import load_from, embeddings

DIR = Path(__file__).parent


class Prompt:
    yaml_path: Path
    # N entries each
    prompts: list[str] = []
    weights: list[float] = []
    embeddings: torch.Tensor  # Shape = (N, 512)
    # Transposed embeddings.
    embeddings_transposed: torch.Tensor  # Shape = (512, N)

    def __init__(self, name: str, *objects: str, **lv0: str):
        # positive, negative = load_from(DIR / f"{name}.yaml")
        # self.positive = PromptList(*positive)
        # self.negative = PromptList(*negative)
        self.prompts, self.weights, self.embeddings = embeddings(
            *load_from(DIR / f"{name}.yaml", *objects, **lv0), dir=DIR / "__cache__"
        )
        # Transpose embeddings to shape (512, N)
        self.embeddings_transposed: torch.Tensor = self.embeddings.T

    def __call__(self, pred: torch.Tensor):
        """
        Match the closest prompt to each prediction,
        returns prompt texts and cosine similarities.
        Input tensor shape: (N, 512)
        Output: [str] * N, float tensor of shape (N,)
        """
        score = pred @ self.embeddings_transposed
        score *= self.weights
        idx: list[int] = torch.argmax(torch.abs(score), dim=1).cpu().numpy().tolist()
        return [(self.prompts[n], float(score[i, n])) for i, n in enumerate(idx)]

    def __iter__(self):
        for i, text in enumerate(self.prompts):
            yield text, self.embeddings[i : i + 1]
