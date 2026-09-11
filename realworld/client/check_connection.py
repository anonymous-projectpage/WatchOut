import os
import time
from pathlib import Path
import numpy as np, cv2
from openpi_client import websocket_client_policy
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import ColorMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PROMPT = "pick up the wood cube and place it on the rubik's cube"
JOINTS = ("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")

cams = {
    "top":   OpenCVCameraConfig(index_or_path=int(os.environ.get("TOP_CAM", "2")), width=640, height=480, fps=30, color_mode=ColorMode.RGB),
    "wrist": OpenCVCameraConfig(index_or_path=int(os.environ.get("WRIST_CAM", "0")), width=640, height=480, fps=30, color_mode=ColorMode.RGB),
}
robot = SO101Follower(SO101FollowerConfig(
    port="/dev/ttyACM0", id="my_follower",
    calibration_dir=Path(os.environ.get("CALIB_DIR", os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower"))),
    cameras=cams, use_degrees=True, max_relative_target=30,
))
robot.connect()
print("robot connected")

def rz(img):
    a = np.asarray(img)
    if a.shape[:2] != (256,256):
        a = cv2.resize(a, (256,256), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(a, dtype=np.uint8)

try:
    obs = robot.get_observation()
    print("obs keys:", sorted(obs.keys()))
    state = np.asarray([obs[f"{j}.pos"] for j in JOINTS], np.float32)

    c = websocket_client_policy.WebsocketClientPolicy("localhost", 8200)
    t0 = time.perf_counter()
    res = c.infer({
        "observation/image":       rz(obs["top"]),
        "observation/wrist_image": rz(obs["wrist"]),
        "observation/state":       state,
        "prompt": PROMPT,
    })
    dt = time.perf_counter() - t0
    a = np.asarray(res["actions"], np.float64)

    print(f"\ninference {dt*1000:.0f} ms   shape {a.shape}   nan {np.isnan(a).any()}")
    print("joint      :", "  ".join(f"{j[:9]:>9}" for j in JOINTS))
    print("current    :", "  ".join(f"{v:9.2f}" for v in state))
    print("pred[0]    :", "  ".join(f"{v:9.2f}" for v in a[0,:6]))
    print("pred[10]   :", "  ".join(f"{v:9.2f}" for v in a[10,:6]))
    print("pred[-1]   :", "  ".join(f"{v:9.2f}" for v in a[-1,:6]))
    print("|pred0-cur|:", "  ".join(f"{v:9.2f}" for v in np.abs(a[0,:6]-state)))
    print("pad col    :", np.round(a[:3,6], 4))
finally:
    robot.disconnect()
    print("disconnected")
