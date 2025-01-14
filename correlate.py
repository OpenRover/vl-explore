import sys, cv2, torch
from argparse import ArgumentParser
from pathlib import Path
from util.env import select_device, to_device

select_device("cpu")

parser = ArgumentParser()
parser.add_argument("images", nargs="+", type=Path, help="Image files to correlate")

import models.clip as clip
from prompts import Prompt

print("Initializing CLIP ...", file=sys.stderr)
clip.init(visual=True, text=True)

print("Loading prompts ...", file=sys.stderr)
NAV = Prompt("navigation")
TRG = Prompt("target")

def report(m: float, l: list[tuple[float, str]]):
    for s, t in l:
        selected = "*" if abs(s) >= m else " "
        sign = "+" if s > 0 else "-"
        yield f"[{selected}] {sign}{abs(s):.3f} | {t}"


for P in map(Path, parser.parse_args().images):
    if not P.exists():
        print(f"File not found: {P}", file=sys.stderr)
        continue
    else:
        print(f"Processing {P} ...", file=sys.stderr)
    with open(P.with_suffix(".txt"), "w") as f:
        # Input image is expected to have BGR8 pixel format
        image = cv2.imread(str(P))
        # Encode image into visual embedding
        t: torch.Tensor = to_device(torch.from_numpy(image)) / 255.0
        m = clip.preprocess(torch.stack([t.permute(2, 0, 1)]))
        v = clip.encode_image(m)
        # Produce correlation scores
        s = (NAV @ v).cpu().numpy().squeeze()
        nav = list(zip(s, NAV.prompts))
        print(f"========== Navigation {len(nav)} ==========", file=f)
        nav.sort(key=lambda x: x[0], reverse=True)
        nav = nav[:5] + nav[-5:]
        m = max([abs(s) for s, _ in nav])
        print(*report(m, nav), sep="\n", file=f)

        s = (TRG @ v).cpu().numpy().squeeze()
        trg = list(zip(s, TRG.prompts))
        print(f"=========== Target {len(trg)} ===========", file=f)
        trg.sort(key=lambda x: x[0], reverse=True)
        trg = trg[:5] + trg[-5:]
        m = max([abs(s) for s, _ in trg])
        print(*report(m, trg), sep="\n", file=f)
