import os
import time, sys
from pathlib import Path
import numpy as np, cv2
from openpi_client import websocket_client_policy
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import ColorMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PROMPT = "pick up the wood cube and place it on the rubik's cube"
JOINTS = ("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
MAX_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
FPS = 30

cams = {
    "top":   OpenCVCameraConfig(index_or_path=int(os.environ.get("TOP_CAM", "2")), width=640, height=480, fps=30, color_mode=ColorMode.RGB),
    "wrist": OpenCVCameraConfig(index_or_path=int(os.environ.get("WRIST_CAM", "0")), width=640, height=480, fps=30, color_mode=ColorMode.RGB),
}
robot = SO101Follower(SO101FollowerConfig(
    port="/dev/ttyACM0", id="my_follower",
    calibration_dir=Path(os.environ.get("CALIB_DIR", os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower"))),
    cameras=cams, use_degrees=True, max_relative_target=30.0,
))

def rz(img):
    a = np.asarray(img)
    if a.shape[:2] != (256,256):
        a = cv2.resize(a, (256,256), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(a, dtype=np.uint8)

robot.connect()
c = websocket_client_policy.WebsocketClientPolicy("localhost", 8200)
print("connected. Ctrl+C to stop.")
try:
    for k in range(MAX_CHUNKS):
        obs = robot.get_observation()
        st = np.asarray([obs[f"{j}.pos"] for j in JOINTS], np.float32)
        t0 = time.perf_counter()
        res = c.infer({"observation/image": rz(obs["top"]),
                       "observation/wrist_image": rz(obs["wrist"]),
                       "observation/state": st, "prompt": PROMPT})
        acts = np.asarray(res["actions"], np.float64)[:, :6]
        print(f"chunk {k+1}/{MAX_CHUNKS}  infer {(time.perf_counter()-t0)*1000:.0f}ms  "
              f"cur {np.round(st,1)}  a0 {np.round(acts[0],1)}")
        dl = time.perf_counter()
        for a in acts:
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, a)})
            dl += 1.0/FPS
            s = dl - time.perf_counter()
            if s > 0: time.sleep(s)
except KeyboardInterrupt:
    print("\nstopping")
finally:
    try:
        o = robot.get_observation()
        robot.send_action({f"{j}.pos": float(o[f"{j}.pos"]) for j in JOINTS})
    except Exception: pass
    robot.disconnect()
    print("disconnected")
