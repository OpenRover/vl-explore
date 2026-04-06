# vl-explore

Model-centric visual-language exploration and navigation pipeline implementation for the VL-explore paper ([arXiv:2502.08791](https://arxiv.org/abs/2502.08791)).

> [!IMPORTANT]
> This repository is **not intended to run stand-alone** for full robot navigation.
> It is used as part of the `OpenRover/ROS2-WorkSpace` integration on branch `zero-shot-nav`, where it is included under the `src/perception` stack.

## What this repository contains

This package provides the model-centric perception and navigation logic centered around:

- **CLIP-based visual embedding** (`models/clip.py`)
- **Prompt-based semantic scoring** (`prompts/*.yaml`, `prompts/__init__.py`)
- **Per-tile navigation correlation and familiarity memory** (`ros2/threads/correlator.py`)
- **Motion mixing and decision logic** (`lib/motion_mixer.py`, `ros2/threads/navigation.py`)
- **Look-around recovery behavior when trapped** (`ros2/threads/navigation.py`, `ros2/utils/look_around.py`)

## Pipeline overview

The ROS2 execution path is split into cooperative components:

1. **Perception** (`ros2/threads/perception.py`)
   - Subscribes to `image`
   - Slices frames into tiles based on strategy (default `6T1P`)
   - Encodes tiles with CLIP image encoder
   - Publishes embeddings over local socket transport

2. **Correlator** (`ros2/threads/correlator.py`)
   - Loads navigation and target prompts
   - Correlates prompt embeddings with visual embeddings
   - Maintains rolling familiarity database for known-space awareness
   - Publishes correlation frames

3. **Navigation** (`ros2/threads/navigation.py`)
   - Consumes correlation frames
   - Produces velocity commands (`geometry_msgs/Twist` on `motion`)
   - Tracks trap conditions via `halt` and odometry travel
   - Runs look-around + reorientation recovery

4. **Recorder** (`ros2/threads/recorder.py`, optional)
   - Dumps perception/correlation/navigation streams and frames for analysis

## Repository status and integration model

This repository is designed as a reusable module in a larger ROS2 system. Full operation requires external workspace-level assets (robot bringup, topic sources, launch orchestration, and platform integration).

Use it through:

- **Organization**: `OpenRover`
- **Workspace repo**: `ROS2-WorkSpace`
- **Branch**: `zero-shot-nav`
- **Role**: integrated `perception`/navigation stack (submodule-managed in workspace setup)

If you only clone this repo, you will have the algorithm code but not the complete runtime environment needed for end-to-end robot deployment.

## Prerequisites

- Python 3.10+
- ROS 2 (ament Python package environment)
- PyTorch-compatible runtime (CPU/GPU)
- OpenCV runtime dependencies

Python dependencies are listed in `requirements.txt`:

- `torch`, `torchvision`
- `open-clip-torch`
- `ultralytics`
- `opencv-python`
- `numpy`, `Pillow`, `tqdm`, `termcolor`, `black`

## Local development setup (module-level)

From this repository root:

```bash
make init
```

This creates `.env`, installs Python requirements, and marks the venv with `COLCON_IGNORE` for workspace compatibility.

Regenerate cached prompt embeddings after prompt changes:

```bash
make prompts
```

Remove local environment:

```bash
make deinit
```

## ROS2 entry points

`setup.py` exposes these console scripts:

- `perception` → `ros2.threads.perception:main`
- `correlator` → `ros2.threads.correlator:main`
- `navigation` → `ros2.threads.navigation:main`
- `recorder` → `ros2.threads.recorder:main`
- `node` → `ros2.node:main`

In integrated workspace usage, these are typically launched by workspace-level orchestration rather than manually.

## Strategy and prompts

Default strategy is parameterized in `ros2/utils/ros.py` via ROS param `strategy`:

- `6T1P` (default): 6 image tiles, 1 prompt group
- Additional strategy classes exist in `lib/strategies.py`

Prompt families are defined under `prompts/`:

- `navigation.yaml`
- `target.yaml`
- directional prompt sets and templates

## Offline scripts

This repository also includes standalone analysis utilities:

- `main.py`: video-file navigation overlay prototype (`data/<dataset>.mp4` input)
- `correlate.py`: image-to-prompt correlation dump

These are useful for algorithm inspection, but they are not a substitute for full ROS2 workspace integration.

## Directory map

- `models/` — CLIP/YOLO model wrappers
- `prompts/` — prompt templates and cached embeddings
- `lib/` — slicing, rendering, motion-mixing strategies
- `ros2/threads/` — ROS2 pipeline components
- `ros2/utils/` — ROS helpers, look-around rendering/analysis tools
- `util/` — shared utilities (transport, queue, logging, math, geometry)

## License

MIT (see `LICENSE`).
