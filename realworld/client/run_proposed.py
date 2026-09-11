import time, sys
from pathlib import Path
import numpy as np, cv2
from openpi_client import websocket_client_policy
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras import ColorMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PROMPT = "pick up the doll and place it in the box"
JOINTS = ("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
M      = int(sys.argv[1]) if len(sys.argv) > 1 else 16
DELTA  = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9   # 1e9 logs the gap without ever replanning
STEPS  = int(sys.argv[3]) if len(sys.argv) > 3 else 100
EPS, DIM = 1e-12, 6

cams = {"top":   OpenCVCameraConfig(index_or_path=int(os.environ.get("TOP_CAM", "2")), width=640, height=480, fps=30, color_mode=ColorMode.RGB),
        "wrist": OpenCVCameraConfig(index_or_path=int(os.environ.get("WRIST_CAM", "0")), width=640, height=480, fps=30, color_mode=ColorMode.RGB)}
robot = SO101Follower(SO101FollowerConfig(
    port="/dev/ttyACM0", id="my_follower",
    calibration_dir=Path(os.environ.get("CALIB_DIR", os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower"))),
    cameras=cams, use_degrees=True, max_relative_target=30.0))

def rz(img):
    a = np.asarray(img)
    if a.shape[:2] != (256,256):
        a = cv2.resize(a, (256,256), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(a, dtype=np.uint8)

import os
VID = os.environ.get("VID_DIR", "./rollout")
os.makedirs(VID, exist_ok=True)
robot.connect()
c = websocket_client_policy.WebsocketClientPolicy("localhost", 8200)
ck = ckraw = None; off = 0; nrep = 0; gaps = []
try:
    for step in range(STEPS):
        for _try in range(5):
            try:
                obs = robot.get_observation(); break
            except TimeoutError:
                time.sleep(0.05)
        else:
            print("camera timeout x5, skip"); continue
        st  = np.asarray([obs[f"{j}.pos"] for j in JOINTS], np.float32)
        t0 = time.perf_counter()
        r = c.infer({"observation/image": rz(obs["top"]),
                     "observation/wrist_image": rz(obs["wrist"]),
                     "observation/state": st, "prompt": PROMPT, "_n_cand": M})
        dt = (time.perf_counter()-t0)*1000
        raw = np.asarray(r["raw"], np.float64)[:, :, :DIM]   # first six dims only
        exe = np.asarray(r["exe"], np.float64)
        q   = np.asarray(r["q"],   np.float64).reshape(-1)

        C  = raw[:, 0, :]
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + EPS)
        self_cs = float((Cn @ Cn.T)[~np.eye(M, dtype=bool)].mean())
        if ckraw is not None and off < len(ckraw):
            cu = ckraw[off] / (np.linalg.norm(ckraw[off]) + EPS)
            gap = self_cs - float((Cn @ cu).mean())
        else:
            gap = 1e9
        if gap < 1e8: gaps.append(gap)

        rep = gap > DELTA or ck is None or off >= len(ck)
        if rep:
            b = int(np.argmax(q)); ck, ckraw, off = exe[b], raw[b], 0; nrep += 1
        a = ck[off][:6]; off += 1
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, a)})
        cv2.imwrite(f"{VID}/top_{step:04d}.jpg",
                    cv2.cvtColor(np.asarray(obs["top"]), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{VID}/wrist_{step:04d}.jpg",
                    cv2.cvtColor(np.asarray(obs["wrist"]), cv2.COLOR_RGB2BGR))
        with open(f"{VID}/log.csv", "a") as _f:
            _f.write(f"{step},{gap if gap<1e8 else ''},{self_cs:.6f},{int(rep)}\n")
        print(f"s{step:3d} {'REPLAN!' if rep else 'continue':8s} {off:3d}", flush=True)
except KeyboardInterrupt:
    print("\nstopping")
finally:
    if gaps:
        g = np.array(gaps)
        print(f"\ngap n={len(g)} p50={np.percentile(g,50):.4f} p90={np.percentile(g,90):.4f} "
              f"p95={np.percentile(g,95):.4f} p99={np.percentile(g,99):.4f} max={g.max():.4f}")
    print(f"replan {nrep}")
    try:
        o = robot.get_observation()
        robot.send_action({f"{j}.pos": float(o[f"{j}.pos"]) for j in JOINTS})
    except Exception: pass
    robot.disconnect()
