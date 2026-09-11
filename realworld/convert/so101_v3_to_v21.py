import os
#!/usr/bin/env python
"""SO-101 LeRobot v3.0 (3 repos) -> merged v2.1 dataset for openpi."""
import glob, io, json, os
import numpy as np
import pandas as pd
import av
from PIL import Image

SRC = os.environ.get("SO101_SRC", "./so101_data")
OUT = os.environ.get("SO101_OUT", os.path.expanduser("~/.cache/huggingface/lerobot/YOUR_HF_USER/so101_mt"))
FPS = 30
SIZE = 256

# ---- EDIT HERE after checking camcheck/*.png ----
CAM = {
    "push_cube_v2":          {"image": "front", "wrist_image": "top"},
    "stack_cube_v4_lerobot": {"image": "top",   "wrist_image": "front"},
    "pick_and_place_v2":     {"image": "front", "wrist_image": "top"},
}
PROMPT = {
    "push_cube_v2":          "push the cube onto the black zone",
    "stack_cube_v4_lerobot": "pick up the wood cube and place it on the rubik's cube",
    "pick_and_place_v2":     "pick up the doll and place it in the box",
}
REPOS = list(CAM.keys())
# -------------------------------------------------


def png_bytes(arr, idx):
    im = Image.fromarray(arr).resize((SIZE, SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": f"frame_{idx:06d}.png"}


def load_frames(path):
    out = []
    with av.open(path) as c:
        c.streams.video[0].thread_type = "AUTO"
        for fr in c.decode(video=0):
            out.append(fr.to_ndarray(format="rgb24"))
    return out


def episode_table(repo):
    files = sorted(glob.glob(f"{SRC}/{repo}/meta/episodes/**/*.parquet", recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")


def data_table(repo):
    files = sorted(glob.glob(f"{SRC}/{repo}/data/**/*.parquet", recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("index")


def main():
    os.makedirs(f"{OUT}/data/chunk-000", exist_ok=True)
    os.makedirs(f"{OUT}/meta", exist_ok=True)

    ep_lines, task_lines = [], []
    global_ep = 0
    global_idx = 0
    total_frames = 0

    for task_idx, repo in enumerate(REPOS):
        task_lines.append({"task_index": task_idx, "task": PROMPT[repo]})
        eps = episode_table(repo)
        data = data_table(repo)
        print(f"[{repo}] {len(eps)} episodes, {len(data)} frames", flush=True)

        cache = {}
        for _, ep in eps.iterrows():
            a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            sl = data.iloc[a:b]
            n = len(sl)

            cols = {}
            for tgt, cam in CAM[repo].items():
                fi = int(ep[f"videos/observation.images.{cam}/file_index"])
                ci = int(ep[f"videos/observation.images.{cam}/chunk_index"])
                key = (cam, ci, fi)
                if key not in cache:
                    cache.clear()
                    p = (f"{SRC}/{repo}/videos/observation.images.{cam}/"
                         f"chunk-{ci:03d}/file-{fi:03d}.mp4")
                    cache[key] = load_frames(p)
                frames = cache[key]
                t0 = float(ep[f"videos/observation.images.{cam}/from_timestamp"])
                off = int(round(t0 * FPS))
                seg = frames[off:off + n]
                if len(seg) != n:
                    raise RuntimeError(
                        f"{repo} ep{int(ep['episode_index'])} {cam}: "
                        f"got {len(seg)} want {n} (off={off}, file={len(frames)})")
                cols[tgt] = [png_bytes(f, i) for i, f in enumerate(seg)]

            df = pd.DataFrame({
                "image": cols["image"],
                "wrist_image": cols["wrist_image"],
                "state": [np.asarray(v, dtype=np.float32) for v in sl["observation.state"]],
                "actions": [np.asarray(v, dtype=np.float32) for v in sl["action"]],
                "timestamp": np.arange(n, dtype=np.float32) / FPS,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, global_ep, dtype=np.int64),
                "index": np.arange(global_idx, global_idx + n, dtype=np.int64),
                "task_index": np.full(n, task_idx, dtype=np.int64),
            })
            df.to_parquet(f"{OUT}/data/chunk-000/episode_{global_ep:06d}.parquet",
                          index=False)
            ep_lines.append({"episode_index": global_ep,
                             "tasks": [PROMPT[repo]], "length": n})
            global_idx += n
            total_frames += n
            global_ep += 1
            print(f"  ep {global_ep-1:03d}  n={n}", flush=True)
        cache.clear()

    info = {
        "codebase_version": "v2.1",
        "robot_type": "so_follower",
        "total_episodes": global_ep,
        "total_frames": total_frames,
        "total_tasks": len(REPOS),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{global_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "image": {"dtype": "image", "shape": [SIZE, SIZE, 3],
                      "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": [SIZE, SIZE, 3],
                            "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": [6], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [6], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(f"{OUT}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)
    with open(f"{OUT}/meta/episodes.jsonl", "w") as f:
        for l in ep_lines:
            f.write(json.dumps(l) + "\n")
    with open(f"{OUT}/meta/tasks.jsonl", "w") as f:
        for l in task_lines:
            f.write(json.dumps(l) + "\n")

    print(f"\nDONE  {global_ep} episodes  {total_frames} frames  -> {OUT}")


if __name__ == "__main__":
    main()
