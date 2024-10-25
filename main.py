#!/usr/bin/env python3
# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
from tqdm import tqdm

import models.clip as clip, cv2
from prompts import Prompt
from util.video import VideoWriter, VideoCapture
from util.iter import skip
from util.logger import Logger
from util.env import args, HOME
import lib.navigation as Nav

log = Logger(__file__)


# Load video from data/nav.mp4
dataset: str = args.dataset
video = VideoCapture(f"data/{dataset}.mp4")
# Get video properties
w, h = video.size

prompts = Prompt("navigation"),
# prompts = Prompt("nav-left"), Prompt("nav-center"), Prompt("nav-right")
nav = Nav.Nav6T1P(prompts, (w, h), (1280, 720))
# nav = Nav.Nav1T3P(prompts, (w, h), (1280, 720))

outfile = HOME / "data" / f"{dataset}_{type(nav).__name__}.mp4"
output, cc = VideoWriter(outfile, video.fps / float(args.frame_skip + 1), (w, h))
log.info(f"Output video: {outfile} ({cc})")

# Progress bar
progress = tqdm(video, desc=f"Processing {dataset}.mp4", unit="frames")

for frame in skip(progress, args.frame_skip):
    pred, confidence, frame = nav(frame)
    print(pred, confidence)
    nav.render(frame, pred, confidence)
    output.write(frame)
    # Display frame
    if args.display:
        cv2.imshow("frame", frame)
        # Exit on any key press
        if cv2.waitKey(1) >= 0:
            break

if args.display:
    cv2.destroyAllWindows()
    cv2.waitKey(1)
log.info("Saving video ...")
output.release()
video.release()
