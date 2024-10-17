# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import torch
from numpy import ndarray
from pathlib import Path
from .__util__ import load_from, embeddings

DIR = Path(__file__).parent


class Prompt:
    def to(self, device: str | torch.device):
        self.weights = self.weights.to(device)
        self.embeddings = self.embeddings.to(device)
        self.embeddings_transposed = self.embeddings_transposed.to(device)

    def __init__(self, name: str, *objects: str, **lv0: str):
        prompts = load_from(DIR, f"{name}.yaml", *objects, **lv0)
        dir = DIR / "__cache__"
        self.prompts, self.weights, self.embeddings = embeddings(prompts, dir=dir)
        # Transpose embeddings to shape (512, N)
        self.embeddings_transposed: torch.Tensor = self.embeddings.T

    def __call__(self, pred: torch.Tensor | ndarray):
        """
        Match the closest prompt to each prediction,
        returns prompt texts and cosine similarities.
        Input tensor shape: (N, 512)
        Output: [str] * N, float tensor of shape (N,)
        """
        if isinstance(pred, ndarray):
            pred = torch.from_numpy(pred).to(self.embeddings.device)
        score = pred @ self.embeddings_transposed
        score *= self.weights
        idx: list[int] = torch.argmax(torch.abs(score), dim=1).cpu().numpy().tolist()
        return [(self.prompts[n], float(score[i, n])) for i, n in enumerate(idx)]

    def __iter__(self):
        for i, text in enumerate(self.prompts):
            yield text, self.embeddings[i : i + 1]
