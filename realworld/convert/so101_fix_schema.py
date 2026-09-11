import os
import glob, json
import pyarrow.parquet as pq

BASE = os.environ.get("SO101_OUT", os.path.expanduser("~/.cache/huggingface/lerobot/YOUR_HF_USER/so101_mt"))

feat = {
    "image": {"_type": "Image"},
    "wrist_image": {"_type": "Image"},
    "state": {"feature": {"dtype": "float32", "_type": "Value"},
              "length": 6, "_type": "Sequence"},
    "actions": {"feature": {"dtype": "float32", "_type": "Value"},
                "length": 6, "_type": "Sequence"},
    "timestamp": {"dtype": "float32", "_type": "Value"},
    "frame_index": {"dtype": "int64", "_type": "Value"},
    "episode_index": {"dtype": "int64", "_type": "Value"},
    "index": {"dtype": "int64", "_type": "Value"},
    "task_index": {"dtype": "int64", "_type": "Value"},
}
meta = {b"huggingface": json.dumps({"info": {"features": feat}}).encode()}

for f in sorted(glob.glob(f"{BASE}/data/chunk-000/*.parquet")):
    t = pq.read_table(f)
    t = t.replace_schema_metadata(meta)
    pq.write_table(t, f)
    print(f.split("/")[-1], flush=True)
print("DONE")
