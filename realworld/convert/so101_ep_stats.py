import os
import glob, io, json
import numpy as np, pandas as pd
from PIL import Image

BASE = os.environ.get("SO101_OUT", os.path.expanduser("~/.cache/huggingface/lerobot/YOUR_HF_USER/so101_mt"))
MAXIMG = 100

def vec_stats(arr):
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
            "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
            "count": [len(arr)]}

def scal_stats(arr):
    return {"min": [float(arr.min())], "max": [float(arr.max())],
            "mean": [float(arr.mean())], "std": [float(arr.std())],
            "count": [len(arr)]}

def img_stats(col):
    idx = np.linspace(0, len(col)-1, min(MAXIMG, len(col))).astype(int)
    acc = []
    for i in idx:
        a = np.asarray(Image.open(io.BytesIO(col.iloc[int(i)]["bytes"])),
                       dtype=np.float32) / 255.0
        acc.append([a[:, :, c] for c in range(3)])
    mins = [min(float(f[c].min()) for f in acc) for c in range(3)]
    maxs = [max(float(f[c].max()) for f in acc) for c in range(3)]
    means = [float(np.mean([f[c].mean() for f in acc])) for c in range(3)]
    stds = [float(np.sqrt(np.mean([f[c].var() + f[c].mean()**2 for f in acc])
                          - np.mean([f[c].mean() for f in acc])**2)) for c in range(3)]
    w = lambda v: [[[x]] for x in v]
    return {"min": w(mins), "max": w(maxs), "mean": w(means),
            "std": w(stds), "count": [len(idx)]}

out = []
for f in sorted(glob.glob(f"{BASE}/data/chunk-000/*.parquet")):
    d = pd.read_parquet(f)
    ep = int(d["episode_index"].iloc[0])
    s = {
        "image": img_stats(d["image"]),
        "wrist_image": img_stats(d["wrist_image"]),
        "state": vec_stats(np.stack(d["state"].values).astype(np.float64)),
        "actions": vec_stats(np.stack(d["actions"].values).astype(np.float64)),
    }
    for c in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        s[c] = scal_stats(d[c].values.astype(np.float64))
    out.append({"episode_index": ep, "stats": s})
    print(f"ep {ep:03d}", flush=True)

with open(f"{BASE}/meta/episodes_stats.jsonl", "w") as fh:
    for l in out:
        fh.write(json.dumps(l) + "\n")
print("DONE", len(out))
