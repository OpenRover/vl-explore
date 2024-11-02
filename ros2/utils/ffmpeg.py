# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from pathlib import Path

def FFMPEG(ff_concat: Path, output: Path):
    command = [
        # fmt: off
        "ffmpeg",
        "-f",         "concat",   # Input mux: concat
        "-i",         ff_concat,  # Input file: list of images
        "-vsync",     "vfr",      # Variable frame rate
        "-vcodec",    "libx264",  # Use x264 codec for encoding
        "-preset",    "medium",   # Medium encoding preset
        "-profile:v", "main",     # Main profile
        "-pix_fmt",   "yuv420p",  # QuickTime compatibility
        "-crf",       "23",       # Quality level (lower is higher quality)
        output,  # Output file
    ]
    return list(map(str, command))
