# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import hashlib, yaml, torch, re, itertools
from pathlib import Path
import models.clip as clip
from util.logger import Logger
from util.env import select_device

log = Logger(__file__)


def embeddings(prompts: list[tuple[str, float]], dir: Path):
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
        select_device()
        log.info(f"Encoding new prompts (hash={hash}, count={len(prompts)})")
        embeddings = clip.encode_text(*prompts)
        torch.save(embeddings, path)
        # Save as text
        with open(dir / f"{hash}.txt", "w") as txt:
            for p, w in zip(prompts, weights):
                txt.write(f"{p} @ {w}\n")
    else:
        log.info(f"Loading prompts from disk (hash={hash})")
        embeddings = torch.load(path, weights_only=True)
    weights = torch.tensor(weights, dtype=embeddings.dtype)
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


load_cache: dict[
    str, dict[str, tuple[list[tuple[str, float]], list[tuple[str, float]]]]
] = {}


def process(
    dir: Path,
    lv_key: str,
    level: dict[str, list[str]] | list[str],
    positive: list[tuple[str, float]] = [],
    negative: list[tuple[str, float]] = [],
):
    if type(level) is dict:
        positive_templates = []
        negative_templates = []
        includes = []
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
                case "include":
                    for inc in template:
                        if type(inc) is not str:
                            raise ValueError(f"invalid yaml include entry: {inc}")
                        load_from(dir, inc)
                        assert (
                            lv_key in load_cache[inc]
                        ), f'key "{lv_key}" missing in {inc}'
                        includes.append(load_cache[inc][lv_key])
                case _:
                    raise ValueError(f"invalid yaml level entry: {key}")
        positive = list(extend(positive_templates, positive))
        negative = list(extend(negative_templates, negative))
        for p, n in includes:
            positive += p
            negative += n
    elif type(level) is list:
        positive = list(extend(level, positive))
        negative = list(extend(level, negative))
    else:
        raise ValueError("invalid yaml level entry")
    return positive, negative


def load_from(dir: Path, yaml_file: str, *objects: str, **lv0: list[str]):
    # Check for existing cache
    if yaml_file in load_cache:
        if "__res__" in load_cache[yaml_file]:
            return load_cache[yaml_file]["__res__"]
        else:
            raise ValueError(f"Loop reference to {yaml_file}")
    # Create new cache entry
    cache = load_cache[yaml_file] = {}

    if len(objects) > 0:
        if "neutral" in lv0:
            lv0["neutral"] += list(objects)
        else:
            lv0["neutral"] = list(objects)

    if len(lv0) > 0:
        positive, negative = process(dir, "__runtime__", lv0, [], [])
    else:
        positive, negative = [], []

    with open(Path(dir) / yaml_file, "r") as file:
        data = yaml.safe_load(file)
    for key in HIERARCHY:
        if key in data:
            positive, negative = process(dir, key, data[key], positive, negative)
            cache[key] = positive, negative

    cache["__res__"] = list(positive) + list([(p, -s) for p, s in negative])
    return cache["__res__"]
