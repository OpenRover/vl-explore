# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import hashlib, yaml, torch, re, itertools
from pathlib import Path
from models.clip import encode_text
from util.env import to_device, Logger

log = Logger(__file__)


def embeddings(
    *prompts: tuple[str, float], dir: Path
) -> tuple[list[str], torch.Tensor]:
    dir.mkdir(parents=True, exist_ok=True)
    # Create unique string representation for all pairs
    prompts = list(prompts)
    assert len(prompts) > 0, "no prompts provided"
    # Sort prompts so change of order does not affect hash
    # Compute hash value for all strings
    prompts, weights = zip(*sorted(prompts, key=lambda p: p[0]))
    hash = hashlib.md5("\n".join(prompts).encode()).hexdigest()[:12]
    # Check if prompt hash exists
    path = dir / f"{hash}.pt"
    embeddings: torch.Tensor = None
    if not path.exists():
        log.info(f"Encoding new prompts (hash={hash})")
        embeddings = encode_text(*prompts)
        torch.save(embeddings, path)
        # Save as text
        with open(dir / f"{hash}.txt", "w") as txt:
            for p, w in zip(prompts, weights):
                txt.write(f"{p} @ {w}\n")
    else:
        log.info(f"Loading prompts from disk (hash={hash})")
        embeddings = torch.load(path, weights_only=True)
    weights = to_device(torch.tensor(weights, dtype=torch.float32))
    embeddings = to_device(embeddings)
    return prompts, weights, embeddings


def extend(templates: list[str], items: list[tuple[str, float]]):
    if len(templates) == 0:
        return items
    templates = preprocess(*templates)
    _templates = []
    for t, s1 in templates:
        if "{}" not in t:
            yield t, s1
        else:
            _templates.append((t, s1))
    count = 0
    for (t, s1), (i, s2) in itertools.product(_templates, items):
        assert "{}" not in i, f"invalid item: {i}"
        yield t.format(i), s1 * s2
        count += 1
    if count == 0:
        for t, s1 in _templates:
            yield t, s1


HIERARCHY = [
    "objects",
    "states",
    "templates",
]


def preprocess(*entries: str):
    for s in entries:
        if type(s) is not str:
            raise ValueError(f"invalid entry type: {type(s)}")
        # Find tailing "@ 123.456" and parse as weight
        s, weight = [*s.split("@"), None][:2]
        if weight:
            try:
                weight = float(weight)
            except:
                raise ValueError(f"invalid weight: {weight}")
        else:
            weight = 1.0
        # Match "(aaa|bbb)" and expand to ["aaa", "bbb"]
        segments = [e.split("|") for e in re.split(r"\(([^)]+)\)", s)]
        for result in itertools.product(*segments):
            yield " ".join("".join(result).split()), weight


def load_from(yaml_file: str, *objects: str, **lv0: list[str]):
    positive, negative = extend([], []), extend([], [])

    def process(level: dict[str, list[str]] | list[str]):
        nonlocal positive, negative
        if type(level) is dict:
            positive_templates = []
            negative_templates = []
            for key, template in level.items():
                if type(template) is not list:
                    template = [str(template)]
                match (key.lower()):
                    case "positive":
                        positive_templates += template
                    case "negative":
                        negative_templates += template
                    case "neutral":
                        positive_templates += template
                        negative_templates += template
                    case _:
                        raise ValueError(f"invalid yaml level entry: {key}")
            positive = extend(positive_templates, positive)
            negative = extend(negative_templates, negative)
        elif type(level) is list:
            positive = extend(level, positive)
            negative = extend(level, negative)
        else:
            raise ValueError("invalid yaml level entry")

    if len(objects) > 0:
        if "neutral" in lv0:
            lv0["neutral"] += list(objects)
        else:
            lv0["neutral"] = list(objects)

    if len(lv0) > 0:
        process(lv0)

    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    for key in HIERARCHY:
        if key in data:
            process(data[key])

    for p, s in positive:
        yield p, s
    for n, s in negative:
        yield n, -s
