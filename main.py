# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import models.clip as clip, cv2, torch, numpy as np
from prompts import Prompt
from util import Region, TextBox
from env import Logger, args

log = Logger(__file__)

nav = Prompt("navigation")
# Release CLIP text model from memory
clip.text_model = None

# Load video from data/nav.mp4
dataset: str = args.dataset
video = cv2.VideoCapture(f"data/{dataset}.mp4")
# Get video properties
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
raw_scale = min(w / 1280, h / 720)
if raw_scale > 1:
    w = int(w / raw_scale)
    h = int(h / raw_scale)
# Box size
s = min(h, w // 3)
t = (h - s) // 2
log.info(f"Video size: {w} x {h}, Box size: {s} x {s}")
# Cropping regions
L = Region(0, t, s, s)
C = Region((w - s) // 2, t, s, s)
R = Region(w, t, -s, s)

RED = np.array([0.0, 0.0, 0.8], dtype=np.float64)
GREEN = np.array([0.0, 0.8, 0.0], dtype=np.float64)
GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float64)
ARROW_LENGTH = int(s * 0.2)
TB_SIZE = int(s * 0.5)
PAD = int(s * 0.05)
T = max(1, s // 200)

# Output video pipe
output = cv2.VideoWriter(
    f"data/{dataset}_output.mp4",
    cv2.VideoWriter_fourcc(*"hvc1"),
    video.get(cv2.CAP_PROP_FPS),
    (w, h),
)


def draw_corners(
    frame,
    r: Region,
    length: int = 10,
    color=(0, 0, 0),
    thickness=1,
    line_type=cv2.LINE_AA,
):
    for (x, y), dx, dy in r.corners():
        for p2 in [(x + dx * length, y), (x, y + dy * length)]:
            cv2.line(
                frame,
                (x, y),
                p2,
                color=color,
                thickness=thickness,
                lineType=line_type,
            )


# Iterate over video frames
while video.isOpened():
    # Read frame
    ret, frame = video.read()
    if not ret:
        break
    if raw_scale > 1:
        frame = cv2.resize(frame, (w, h))
    # Crop regions
    l, c, r = L(frame), C(frame), R(frame)
    v = clip.encode_image(l, c, r)
    # Correlation with navigation prompts
    l, c, r = nav(v)
    # Display text and score on frame
    for region, (text, score) in zip([L, C, R], [l, c, r]):
        sat = min(abs(score) / 0.5, 1)
        color = GREEN if score >= 0 else RED
        color = color * sat + GRAY * (1 - sat)
        color = list(map(int, color * 255))
        t_box: TextBox
        if region is C:
            draw_corners(frame, region, length=s // 8, color=color, thickness=T)
            p1 = region.tc[0], PAD
            p2 = x, y = p1[0], p1[1] + ARROW_LENGTH
            t_box = TextBox(
                Region(x - TB_SIZE // 2, y + PAD, TB_SIZE, TB_SIZE),
                vertical_align="top",
            )
        elif region is L:
            p1 = region.ml
            p2 = x, y = p1[0] + ARROW_LENGTH, p1[1]
            t_box = TextBox(Region(x + PAD, y - TB_SIZE // 2, TB_SIZE, TB_SIZE))
        elif region is R:
            p1 = region.mr
            p2 = x, y = p1[0] - ARROW_LENGTH, p1[1]
            t_box = TextBox(Region(x - PAD, y - TB_SIZE // 2, -TB_SIZE, TB_SIZE))
        cv2.arrowedLine(frame, p2, p1, color, T * 2, tipLength=0.6)
        t_box(
            frame,
            f"{text} ({score:.2f})",
            color=color,
            line_height=1.0,
            thickness=T,
            scale=s / 600,
        )
    # Display frame
    cv2.imshow("frame", frame)
    output.write(frame)
    # Exit on any key press
    if cv2.waitKey(1) >= 0:
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
log.info("saving video ...")
output.release()
video.release()
