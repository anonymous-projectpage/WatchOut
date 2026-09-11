# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import argparse
import logging
import traceback
import os
import os.path as osp
import json
import torch
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
import sys

def main():
    parser = argparse.ArgumentParser(description="Launch XVLA inference FastAPI server")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained XVLA model directory")
    parser.add_argument('--processor_path', type=str, default=None)
    parser.add_argument('--LoRA_path', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./logs",
                        help="Directory to save runtime info (info.json)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to load model on (cuda / cpu / auto)")
    parser.add_argument("--port", default=8010, type=int,
                        help="Port number for FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", type=str,
                        help="Host address for FastAPI server")
    parser.add_argument("--disable_slurm", action="store_true", default=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Starting XVLA Inference Server...")
    print(f"🔹 Model Path  : {args.model_path}")
    print(f"🔹 Output Dir  : {args.output_dir}")
    print(f"🔹 Device Arg  : {args.device}")
    print(f"🔹 Port        : {args.port}")

    # --------------------------------------------------------------------------
    # Select device automatically
    # --------------------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🧠 Using device: {device}")

    # --------------------------------------------------------------------------
    # Load processor (if available)
    # --------------------------------------------------------------------------
    processor = None
    try:
        print("\n🧩 Loading XVLAProcessor...")
        processor_path = args.processor_path if args.processor_path else args.model_path
        processor =  XVLAProcessor.from_pretrained(processor_path)
        print("✅ XVLAProcessor loaded successfully.")
    except Exception as e:
        print(f"⚠️ No processor found or failed to load: {e}")

    # --------------------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------------------
    print("\n📦 Loading XVLA model from pretrained checkpoint...")
    try:
        model = XVLA.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device).to(torch.float32)
        
        if args.LoRA_path is not None:
            print(f"🔸 Applying LoRA weights from {args.LoRA_path} ...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                args.LoRA_path,
                torch_dtype=torch.float32,
            ).to(device)
            
            print("✅ LoRA weights applied successfully.")
            
            
        print("✅ Model successfully loaded and moved to device.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # --------------------------------------------------------------------------
    # SLURM environment detection
    # --------------------------------------------------------------------------
    node_list = os.environ.get("SLURM_NODELIST")
    job_id = os.environ.get("SLURM_JOB_ID", "none")

    if node_list and not args.disable_slurm:
        print("\n🖥️  SLURM Environment Detected:")
        print(f"   Node list : {node_list}")
        print(f"   Job ID    : {job_id}")

        # Extract host
        try:
            host = ".".join(node_list.split("-")[1:]) if "-" in node_list else node_list
        except Exception:
            host = args.host
    else:
        print("\n⚠️  No SLURM environment detected, defaulting to 0.0.0.0")
        host = args.host

    # --------------------------------------------------------------------------
    # Write info.json for bookkeeping (safe version)
    # --------------------------------------------------------------------------
    info_path = osp.join(args.output_dir, "info.json")
    infos = {
        "host": host,
        "port": args.port,
        "job_id": job_id,
        "node_list": node_list or "none",
    }

    # --- Check existence before writing ---
    if osp.exists(info_path):
        print(f"❌ Error: {info_path} already exists. "
            f"This usually means another server is still running or the previous job did not clean up properly.")
        print("👉 Please remove it manually or use a different --output_dir.")
        sys.exit(1)

    # --- Write safely ---
    try:
        with open(info_path, "w") as f:
            json.dump(infos, f, indent=4)
        print(f"📝 Server info written to {info_path}")
    except Exception as e:
        print(f"⚠️ Failed to write {info_path}: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Launch FastAPI server
    # --------------------------------------------------------------------------
    print(f"\n🌐 Launching FastAPI service at http://{host}:{args.port} ...")
    # ---- adaptive chunking: critic + /act_cands ----
    _critic = None
    _cp = os.environ.get("IQL_CRITIC_PATH", "")
    if _cp and osp.exists(_cp):
        import sys as _sys
        _sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
        from xvla_iql_std import XVLAIQL
        import numpy as _np
        _sd = torch.load(_cp, map_location="cpu")
        _ctxd = _sd["chunk_critic.q1.context_proj.weight"].shape[1]
        _std  = _sd["chunk_critic.q1.state_proj.weight"].shape[1]
        _actd = _sd["chunk_critic.q1.action_proj.weight"].shape[1]
        _H    = _sd["chunk_critic.q1.position_embedding"].shape[1] - 3
        print(f"[cands] critic ctx={_ctxd} state={_std} act={_actd} H={_H}", flush=True)
        _critic = XVLAIQL(context_dim=_ctxd, state_dim=_std,
                          action_dim=_actd, action_horizon=_H)
        _critic.load_state_dict(_sd)
        _critic = _critic.to(device).to(torch.float32).eval()
        for _q in _critic.parameters():
            _q.requires_grad_(False)

        model._build_app(processor)
        _app = model.app
        from fastapi.responses import JSONResponse as _JR
        import json_numpy as _jn
        import numpy as _np2
        from PIL import Image as _Img
        import cv2 as _cv2

        @_app.post("/act_cands")
        def act_cands(payload: dict):
            try:
                model.eval()
                imgs = []
                for k in ("image0", "image1", "image2"):
                    if k not in payload: continue
                    v = _jn.loads(payload[k])
                    if isinstance(v, _np2.ndarray):
                        if v.ndim == 1:
                            v = _cv2.imdecode(v, _cv2.IMREAD_COLOR)
                        imgs.append(_Img.fromarray(v))
                if not imgs:
                    return _JR({"error": "no images"}, status_code=400)
                inp = processor(imgs, payload["language_instruction"])
                proprio = torch.as_tensor(_np2.asarray(_jn.loads(payload["proprio"])))
                did = torch.tensor([int(payload["domain_id"])], dtype=torch.long)
                dt = next(model.parameters()).dtype
                def _tm(t):
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dt) if t.is_floating_point() else t.to(device=device)
                inp = {k: _tm(v) for k, v in inp.items()}
                inp.update({"proprio": _tm(proprio.unsqueeze(0)), "domain_id": did.to(device)})

                M = int(payload.get("n_cand", 16))
                steps = int(payload.get("steps", 10))
                with torch.no_grad():
                    enc = model.forward_vlm(inp["input_ids"], inp["image_input"], inp["image_mask"])
                    ctx = enc["vlm_features"]
                    D = model.action_space.dim_action
                    encM = {k: (v.repeat(M, *([1] * (v.dim() - 1)))
                                if torch.is_tensor(v) else v)
                            for k, v in enc.items()}
                    didM = inp["domain_id"].repeat(M)
                    prM = inp["proprio"].repeat(M, 1)
                    x1 = torch.randn(M, model.num_actions, D,
                                     device=device, dtype=inp["proprio"].dtype)
                    act = torch.zeros_like(x1)
                    for i in range(max(1, steps), 0, -1):
                        t = torch.full((M,), i / steps, device=device, dtype=inp["proprio"].dtype)
                        x_t = x1 * t.view(-1,1,1) + act * (1 - t).view(-1,1,1)
                        pm, xm = model.action_space.preprocess(prM, x_t)
                        act = model.transformer(domain_id=didM,
                                                action_with_noise=xm, proprio=pm, t=t, **encM)
                    raws = list(act.float().cpu().numpy())
                    exes = list(model.action_space.postprocess(act.clone()).float().cpu().numpy())
                    raw_t = act.float()
                    ctxM = ctx.float().repeat(M, 1, 1)
                    stM  = inp["proprio"].float().repeat(M, 1)
                    q1, q2 = _critic.chunk_critic(ctxM, None, stM, raw_t)
                    q = torch.minimum(q1, q2).squeeze(-1).cpu().numpy()
                return _JR({"raw": _np2.stack(raws).tolist(),
                            "exe": _np2.stack(exes).tolist(),
                            "q": q.tolist()})
            except Exception:
                logging.error(traceback.format_exc())
                return _JR({"error": "act_cands failed"}, status_code=400)
        print("[cands] /act_cands registered", flush=True)

    try:
        if hasattr(model, "run"):
            model.run(processor=processor, host=host, port=args.port)
        else:
            print("❌ The loaded model does not implement `.run()` (FastAPI entrypoint).")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped manually.")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")


if __name__ == "__main__":
    main()
