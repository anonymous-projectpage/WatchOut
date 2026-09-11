import os, sys, json, time, atexit
import numpy as np
import cv2
import h5py

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.append(os.path.join(os.path.dirname(parent_directory), "pi05"))
from pi_model import *

OUT_DIR    = os.environ.get("ONLINE_OUT_DIR", "/tmp/online_buffer")
NOISE_STD  = float(os.environ.get("ONLINE_NOISE_STD", "0.05"))
NORM_STATS = os.environ.get("ONLINE_NORM_STATS", "")
CKPT_OVERRIDE  = os.environ.get("ONLINE_CKPT_ID", "")
MODEL_OVERRIDE = os.environ.get("ONLINE_MODEL_NAME", "")
JPEG_Q = 95
os.makedirs(OUT_DIR, exist_ok=True)


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


class Recorder:
    def __init__(self):
        self.reset()
        self.task_env = None
        self.instruction = ""
        self.ep_idx = 0
        atexit.register(self.flush)

    def reset(self):
        self.rgb, self.state, self.action = [], [], []
        self.ep_success = 0

    def add(self, rgb_arr, state, action):
        enc = [cv2.imencode(".jpg", c[:, :, ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tobytes()
               for c in rgb_arr]
        self.rgb.append(enc)
        self.state.append(np.asarray(state, np.float32))
        self.action.append(np.asarray(action, np.float32))

    def flush(self):
        if not self.action:
            return
        success = self.ep_success
        if self.task_env is not None:
            success = max(success, int(bool(getattr(self.task_env, "eval_success", False))))
        T = len(self.action)
        path = os.path.join(OUT_DIR,
            "ep_%d_%d_%d.hdf5" % (os.getpid(), int(time.time()*1000), self.ep_idx))
        with h5py.File(path, "w") as f:
            f.attrs["success"] = success
            f.attrs["length"] = T
            f.attrs["instruction"] = str(self.instruction).encode("utf-8")
            f.attrs["task"] = str(os.environ.get("ONLINE_TASK", "unknown")).encode("utf-8")
            f.attrs["noise_std"] = NOISE_STD
            f.attrs["ckpt_id"] = str(CKPT_OVERRIDE or "yml_default").encode("utf-8")
            f.create_dataset("state", data=np.stack(self.state))
            f.create_dataset("action", data=np.stack(self.action))
            dt = h5py.vlen_dtype(np.dtype("uint8"))
            for ci, cname in enumerate(["head", "right", "left"]):
                d = f.create_dataset("rgb_" + cname, (T,), dtype=dt)
                for t in range(T):
                    d[t] = np.frombuffer(self.rgb[t][ci], dtype=np.uint8)
        print("[collect] saved %s T=%d success=%d" % (os.path.basename(path), T, success),
              flush=True)
        self.ep_idx += 1
        self.reset()


_REC = Recorder()
_ASTD = None


def _action_std():
    global _ASTD
    if _ASTD is not None:
        return _ASTD
    if NORM_STATS and os.path.exists(NORM_STATS):
        with open(NORM_STATS) as f:
            ns = json.load(f)
        for k in ("norm_stats", "data"):
            ns = ns.get(k, ns)
        a = ns["actions"] if "actions" in ns else ns["action"]
        _ASTD = np.asarray(a["std"], np.float32)
    else:
        _ASTD = np.ones(14, np.float32)
        print("[collect] WARN: norm_stats missing -> std=1", flush=True)
    return _ASTD


def get_model(usr_args):
    model_name = MODEL_OVERRIDE or usr_args["model_name"]
    ckpt_id    = os.environ.get("CKPT_OVERRIDE") or CKPT_OVERRIDE or usr_args["checkpoint_id"]
    print("[collect] loading %s/%s/%s" % (usr_args["train_config_name"], model_name, ckpt_id),
          flush=True)
    global _PI0_STEP
    _PI0_STEP = usr_args["pi0_step"]
    return PI0(usr_args["train_config_name"], model_name, ckpt_id,
               usr_args["pi0_step"])


_WEAK = None
_PI0_STEP = 50
def _get_weak():
    """Weak policy for BID, selected via BID_WEAK_CONFIG / BID_WEAK_MODEL / BID_WEAK_CKPT."""
    global _WEAK
    if _WEAK is None:
        cfg = os.environ.get("BID_WEAK_CONFIG", "pi05_mt50_off2on")
        mdl = os.environ.get("BID_WEAK_MODEL", "adaptive_base")
        ck  = os.environ.get("BID_WEAK_CKPT", "30000")
        print(f"[bid] loading weak {cfg}/{mdl}/{ck}", flush=True)
        os.environ["_LOADING_BID_WEAK"] = "1"
        try:
            _WEAK = PI0(cfg, mdl, ck, _PI0_STEP)
        finally:
            os.environ["_LOADING_BID_WEAK"] = "0"
    return _WEAK


def _find_actors(task_env):
    """Collect actors in TASK_ENV that support set_pose."""
    found = {}
    for name in dir(task_env):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(task_env, name)
        except Exception:
            continue
        if hasattr(obj, "get_pose") or hasattr(obj, "actor") or hasattr(obj, "entity"):
            found[name] = obj
    return found


def _apply_perturb(task_env, step):
    """Spread the displacement over PERTURB_DUR steps when it is greater than 1."""
    _cx = os.environ.get("CUP_CENTER_X", "")
    if _cx and step == 0:
        try:
            import sapien as _sp2
            for _n, _o in _find_actors(task_env).items():
                if "cup" in _n:
                    _e = getattr(_o, "actor", None) or _o
                    _p = _e.get_pose()
                    _np2 = np.asarray(_p.p).copy(); _np2[0] = float(_cx)
                    _cy = os.environ.get("CUP_CENTER_Y", "")
                    if _cy: _np2[1] = float(_cy)
                    _e.set_pose(_sp2.Pose(_np2, _p.q))
                    print(f"[setup] cup centered x={_cx}", flush=True)
        except Exception as _e2:
            print(f"[setup] center failed: {_e2}", flush=True)
    if os.environ.get("PERTURB_RANDOM", "0") == "1" and step == 0:
        _sd = int(os.environ.get("PERTURB_SEED", "0"))
        _rg = np.random.RandomState(_sd)
        _s0, _s1 = [int(x) for x in os.environ.get("PERTURB_STEP_RANGE", "10,150").split(",")]
        _lo, _hi = [float(x) for x in os.environ.get("PERTURB_DIST_RANGE", "0.05,0.12").split(",")]
        _st = int(_rg.randint(_s0, _s1 + 1))
        _a0, _a1 = [float(x) for x in os.environ.get("PERTURB_ANGLE_RANGE", "0,360").split(",")]
        _ang = float(_rg.uniform(np.radians(_a0), np.radians(_a1)))
        _dist = float(_rg.uniform(_lo, _hi))
        os.environ["PERTURB_STEP"] = str(_st)
        os.environ["PERTURB_DX"] = str(_dist * np.cos(_ang))
        os.environ["PERTURB_DY"] = str(_dist * np.sin(_ang))
        print(f"[perturb-rand] seed={_sd} step={_st} angle={np.degrees(_ang):.0f}deg "
              f"dist={_dist*100:.1f}cm dx={_dist*np.cos(_ang):+.3f} dy={_dist*np.sin(_ang):+.3f}", flush=True)
    _dur = int(os.environ.get("PERTURB_DUR", "1"))
    if _dur > 1:
        _p0 = int(os.environ.get("PERTURB_STEP", "-1"))
        if _p0 < 0 or not (_p0 <= step < _p0 + _dur):
            return
        _saved = os.environ.get("PERTURB_DX", "0.08")
        _savedy = os.environ.get("PERTURB_DY", "0.0")
        os.environ["PERTURB_DX"] = str(float(_saved) / _dur)
        os.environ["PERTURB_DY"] = str(float(_savedy) / _dur)
        os.environ["PERTURB_STEP"] = str(step)
        os.environ["PERTURB_DUR"] = "1"
        try:
            _apply_perturb(task_env, step)
        finally:
            os.environ["PERTURB_DX"] = _saved
            os.environ["PERTURB_DY"] = _savedy
            os.environ["PERTURB_STEP"] = str(_p0)
            os.environ["PERTURB_DUR"] = str(_dur)
        return

    """Displace PERTURB_ACTOR by (dx, dy) at PERTURB_STEP."""
    pstep = int(os.environ.get("PERTURB_STEP", "-1"))
    if pstep < 0 or step != pstep:
        return
    actors = _find_actors(task_env)
    target = os.environ.get("PERTURB_ACTOR", "")
    if not target:
        print(f"[perturb] available actors: {list(actors.keys())}", flush=True)
        try:
            import sapien as _sp
            _PD = _sp.physx.PhysxRigidDynamicComponent
        except Exception as _e:
            _PD = None
            print(f"[perturb] sapien physx import failed: {_e}", flush=True)
        for _k, _v in actors.items():
            _inner = getattr(_v, "entity", None) or getattr(_v, "actor", None) or _v
            _comp = None
            if _PD is not None and hasattr(_inner, "find_component_by_type"):
                try:
                    _comp = _inner.find_component_by_type(_PD)
                except Exception:
                    pass
            print(f"[perturb]   {_k}: {type(_v).__name__} inner={type(_inner).__name__} "
                  f"physx={'YES' if _comp is not None else 'no'}", flush=True)
        return
    hit = [k for k in actors if target in k]
    if not hit:
        print(f"[perturb] '{target}' not found in {list(actors.keys())}", flush=True)
        return
    import numpy as _np, sapien as _sp
    a = actors[hit[0]]
    ent = getattr(a, "actor", None) or getattr(a, "entity", None) or a
    dx = float(os.environ.get("PERTURB_DX", "0.08"))
    dy = float(os.environ.get("PERTURB_DY", "0.0"))
    mode = os.environ.get("PERTURB_MODE", "teleport")
    comp = None
    try:
        comp = ent.find_component_by_type(_sp.physx.PhysxRigidDynamicComponent)
    except Exception:
        pass
    if mode == "velocity" and comp is not None:
        if os.environ.get("PERTURB_MIRROR", "") == "1":
            _px = float(_np.asarray(ent.get_pose().p)[0])
            _txs = os.environ.get("_PERT_TX", "")
            if not _txs:
                _tx = -_px
                os.environ["_PERT_TX"] = str(_tx)
                print(f"[perturb] mirror start x={_px:+.3f} -> {_tx:+.3f}", flush=True)
            else:
                _tx = float(_txs)
            if abs(_px - _tx) < 0.02:
                comp.set_linear_velocity([0.0, 0.0, 0.0])
                print(f"[perturb] mirror DONE step={step} x={_px:+.3f}", flush=True)
                return
            dx = float(os.environ.get("PERTURB_VEL", "0.5")) * (1.0 if _tx > _px else -1.0)
            dy = 0.0
        comp.set_linear_velocity([dx, dy, 0.0])
        print(f"[perturb] step={step} {hit[0]} velocity=({dx},{dy})", flush=True)
    else:
        pose = ent.get_pose()
        if os.environ.get("PERTURB_MIRROR", "") == "1":
            _px = float(_np.asarray(pose.p)[0])
            _txs = os.environ.get("_PERT_TX", "")
            if not _txs:
                _tx = -_px
                os.environ["_PERT_TX"] = str(_tx)
                print(f"[perturb] mirror start x={_px:+.3f} -> {_tx:+.3f}", flush=True)
            else:
                _tx = float(_txs)
            if abs(_px - _tx) < 0.01:
                print(f"[perturb] mirror DONE step={step} x={_px:+.3f}", flush=True)
                return
            _n = max(1, int(os.environ.get("MIRROR_STEPS", "40")))
            dx = (_tx - _px) / _n
            dy = 0.0
        if os.environ.get("PERTURB_DIR", "") == "outward":
            _sgn = 1.0 if _np.asarray(pose.p)[0] > 0 else -1.0
            dx = abs(dx) * _sgn
        elif os.environ.get("PERTURB_DIR", "") == "center":
            _sgn = -1.0 if _np.asarray(pose.p)[0] > 0 else 1.0
            dx = abs(dx) * _sgn
        newp = _np.asarray(pose.p) + _np.array([dx, dy, 0.0])
        ent.set_pose(_sp.Pose(newp, pose.q))
        if comp is not None:
            comp.set_linear_velocity([0.0, 0.0, 0.0])
            comp.set_angular_velocity([0.0, 0.0, 0.0])
        print(f"[perturb] step={step} {hit[0]} teleport d=({dx},{dy}) {pose.p}->{newp}", flush=True)


def eval(TASK_ENV, model, observation):
    _REC.task_env = TASK_ENV
    if model.observation_window is None:
        _lim = int(os.environ.get("ONLINE_STEP_LIM", "0"))
        if _lim > 0:
            TASK_ENV.step_lim = _lim
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)
        _REC.instruction = instruction

    import sys as _sys
    _sys.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
    from cs_rule import AdaptiveChunkRule

    DELTA = float(os.environ.get("CS_DELTA", "0.35"))
    KCHK  = int(os.environ.get("CS_KCHECK", "5"))
    NCAND = int(os.environ.get("CS_NCAND", "1"))
    CMPS  = int(os.environ.get("CS_CMPSTEPS", "5"))
    MODE  = os.environ.get("CHUNK_MODE", "fixed")

    _mu = _sd = None
    if os.environ.get("CS_NORM", "1") == "1":
        try:
            import json as _js, glob as _gl
            _cfg = os.environ.get("CS_NORM_CONFIG", "pi05_mt50_off2on")
            _c = _gl.glob(f"policy/pi05/assets/{_cfg}/*/*/norm_stats.json")
            _ns = _js.load(open(_c[0]))["norm_stats"]["actions"]
            _mu = np.asarray(_ns["mean"], np.float64)
            _sd = np.asarray(_ns["std"], np.float64) + 1e-6
            print(f"[cs] norm loaded from {_c[0]}", flush=True)
        except Exception as _e:
            print(f"[cs] norm load failed: {_e}", flush=True)
    rule = AdaptiveChunkRule(delta=DELTA, cmp_steps=CMPS, k_check=KCHK, mu=_mu, sd=_sd)
    _te = _aac_cfg = None
    if MODE in ("te", "aac"):
        import sys as _sy2
        _sy2.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
        import aac_te as _AT
        if MODE == "te":
            _te = _AT.TemporalEnsemble(horizon=50)
            print(f"[te] m={_te.m}", flush=True)
        else:
            _aac_cfg = _AT.aac_config()
            print(f"[aac] cfg={_aac_cfg}", flush=True)
        rule._h_exec = 0
    H = int(os.environ.get("FIXED_H", "0")) or model.pi0_step
    n_replan = n_keep = 0

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    if MODE == "fixed":
        rule.set_chunk(np.asarray(model.get_action()[:H], np.float32))

    for step in range(int(getattr(TASK_ENV, "step_lim", 400))):
        if MODE == "bid":
            import sys as _sy4
            _sy4.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
            from adaptive_infer import sample_and_rank_jit as _srj
            _NB = int(os.environ.get("BID_NCAND", "8"))
            _NW = int(os.environ.get("BID_NWEAK", "8"))
            _LAM = float(os.environ.get("BID_LAMBDA", "1.0"))
            _KB = int(os.environ.get("BID_HORIZON", "5"))
            if rule.chunk is None or rule.offset >= len(rule.chunk):
                _cd, _cdr, _qv, _qb = _srj(model.policy, model.observation_window, _NB, H)
                _wk = _get_weak()
                if getattr(_wk, "instruction", None) is None:
                    _wk.set_language(TASK_ENV.get_instruction())
                _wk.update_observation_window(input_rgb_arr, input_state)
                _wd, _wdr, _, _ = _srj(_wk.policy, _wk.observation_window, _NW, H)
                _C = np.stack([np.asarray(c[:_KB], np.float64).ravel() for c in _cd])
                _W = np.stack([np.asarray(c[:_KB], np.float64).ravel() for c in _wd])
                if os.environ.get("BID_CENTER", "1") == "1":
                    _mu0 = np.concatenate([_C, _W]).mean(0, keepdims=True)
                    _C = _C - _mu0; _W = _W - _mu0
                _Cn = _C / (np.linalg.norm(_C, axis=1, keepdims=True) + 1e-12)
                _Wn = _W / (np.linalg.norm(_W, axis=1, keepdims=True) + 1e-12)

                _fc = -(_Cn @ _Wn.T).mean(axis=1)

                if getattr(rule, "_prev", None) is not None:
                    _pv = np.asarray(rule._prev, np.float64).ravel()
                    _pv = _pv / (np.linalg.norm(_pv) + 1e-12)
                    _bc = _Cn @ _pv
                else:
                    _bc = np.zeros(len(_Cn))
                _sc = _bc + _LAM * _fc
                _pick = int(np.argmax(_sc))
                rule.set_chunk(_cd[_pick]); n_replan += 1
                rule._prev = np.asarray(_cd[_pick][:_KB], np.float64)
                if os.environ.get("BID_VERBOSE", "0") == "1":
                    print(f"[bid] step={step} pick={_pick} bc={_bc[_pick]:.3f} "
                          f"fc={_fc[_pick]:.3f}", flush=True)
            else:
                n_keep += 1
            action = rule.next_action()
            _REC.add(input_rgb_arr, input_state, action)
            _apply_perturb(TASK_ENV, step)
            TASK_ENV.take_action(action)
            observation = TASK_ENV.get_obs()
            input_rgb_arr, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)
            if bool(getattr(TASK_ENV, "eval_success", False)):
                _REC.ep_success = 1
                break
            continue
        if MODE in ("te", "aac"):
            if MODE == "te":
                _ch = np.asarray(model.get_action()[:H], np.float32)
                if step == 0:
                    _te.H = len(_ch)
                    print(f"[te] chunk={_ch.shape} H={_te.H}", flush=True)
                _te.add_chunk(step, _ch)
                action = np.asarray(_te.action(step), np.float32).ravel()
                n_replan += 1
            else:
                if rule.chunk is None or rule.offset >= max(1, getattr(rule, "_h_exec", 0))                        or rule.offset >= len(rule.chunk):
                    import sys as _sy3
                    _sy3.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
                    from adaptive_infer import sample_and_rank_jit as _srj
                    from aac_te import aac_chunk_size as _acs
                    _N = int(os.environ.get("AAC_NCAND", "20"))
                    _cd, _cdr, _qv, _qb = _srj(model.policy, model.observation_window, _N, H)
                    _pick = 0 if os.environ.get("AAC_USE_Q","0") != "1" else _qb
                    _hs, _ = _acs(np.stack([np.asarray(c) for c in _cd]), _aac_cfg)
                    rule.set_chunk(_cd[_pick]); rule._h_exec = _hs; n_replan += 1
                else:
                    n_keep += 1
                action = rule.next_action()
            _REC.add(input_rgb_arr, input_state, action)
            _apply_perturb(TASK_ENV, step)
            TASK_ENV.take_action(action)
            observation = TASK_ENV.get_obs()
            input_rgb_arr, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)
            if bool(getattr(TASK_ENV, "eval_success", False)):
                _REC.ep_success = 1
                break
            continue
        if MODE == "cs_q" and (step % max(1, int(os.environ.get("CS_KCHECK","1"))) != 0)                and rule.chunk is not None and rule.offset < len(rule.chunk):
            action = rule.next_action()
            _REC.add(input_rgb_arr, input_state, action)
            _apply_perturb(TASK_ENV, step)
            TASK_ENV.take_action(action)
            observation = TASK_ENV.get_obs()
            input_rgb_arr, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)
            if bool(getattr(TASK_ENV, "eval_success", False)):
                _REC.ep_success = 1
                break
            continue
        if MODE == "cs_q":
            import sys as _sy
            _sy.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
            from adaptive_infer import sample_and_rank_jit
            if "_CSDUMP" not in globals():
                globals()["_CSDUMP"] = []
            _M = int(os.environ.get("CS_NCAND", "6"))
            cands, cands_raw, qv, qbest = sample_and_rank_jit(model.policy, model.observation_window, _M, H)
            if os.environ.get("CS_GAP", "0") == "1":
                _C = np.stack([np.asarray(c[0], np.float64) for c in cands_raw])
                _Cn = _C / (np.linalg.norm(_C, axis=1, keepdims=True) + 1e-12)
                _M = len(cands)
                _pw = _Cn @ _Cn.T
                _self = float(_pw[~np.eye(_M, dtype=bool)].mean())
                if getattr(rule, "chunk_raw", None) is not None and rule.offset < len(rule.chunk_raw):
                    _cu = np.asarray(rule.chunk_raw[rule.offset], np.float64)
                    _cu = _cu / (np.linalg.norm(_cu) + 1e-12)
                    _cross = float((_Cn @ _cu).mean())
                    _gap = _self - _cross
                    if os.environ.get("CS_RATIO", "0") == "1":
                        _den = max(abs(_self), 1e-6)
                        _gap = _gap / _den
                else:
                    _gap = 1e9
                _use = _gap > DELTA
                rule.log.append((step, _gap, _use))
                _qcur = float("nan")
                if os.environ.get("CS_QCUR", "0") == "1" and getattr(rule, "chunk_raw", None) is not None:
                    try:
                        from adaptive_infer import make_observation, make_jitted_q
                        import jax.numpy as _jnp
                        _obsv = make_observation(model.policy, model.observation_window)
                        _qfn = make_jitted_q(model.policy._model)
                        _crf = np.asarray(rule.chunk_raw, np.float32)
                        if os.environ.get("CS_QCUR_SHIFT", "0") == "1":
                            _off = int(rule.offset)
                            _rem = _crf[_off:]
                            if len(_rem) == 0: _rem = _crf[-1:]
                            _n = len(_crf) - len(_rem)
                            if _n > 0:
                                _rem = np.concatenate([_rem, np.repeat(_rem[-1:], _n, axis=0)], axis=0)
                            _crf = _rem
                        _cr = _crf[None, ...]
                        _qcur = float(np.asarray(_qfn(_obsv, _jnp.asarray(_cr)))[0])
                    except Exception as _e:
                        pass
                if os.environ.get("CS_DUMP", "0") == "1":
                    _qa = np.asarray(qv, np.float32)
                    _t3 = np.argsort(_qa)[::-1][:3]
                    _CSDUMP.append(dict(
                        step=int(step), gap=float(_gap), qcur=float(_qcur),
                        q=_qa, top3=_t3.astype(np.int32),
                        top3_raw=np.asarray(cands_raw, np.float32)[_t3],
                        cur_raw=(np.asarray(rule.chunk_raw, np.float32)
                                 if getattr(rule, "chunk_raw", None) is not None
                                 else np.zeros((1, 1), np.float32)),
                        cur_off=int(rule.offset) if rule.chunk is not None else -1,
                        replan=bool(_use or rule.chunk is None
                                    or rule.offset >= len(rule.chunk))))
                if _use or rule.chunk is None or rule.offset >= len(rule.chunk):
                    rule.set_chunk(cands[qbest]); rule.chunk_raw = cands_raw[qbest]; n_replan += 1
                    if os.environ.get("CS_VERBOSE","0") == "1":
                        print(f"[gap] step={step} gap={_gap:.5f} self={_self:.5f} REPLAN pick={qbest} qcur={_qcur:.4f} q={np.round(np.asarray(qv),4).tolist()}", flush=True)
                else:
                    n_keep += 1
                    if os.environ.get("CS_VERBOSE","0") == "1":
                        print(f"[gap] step={step} gap={_gap:.5f} self={_self:.5f} keep qcur={_qcur:.4f} q={np.round(np.asarray(qv),4).tolist()}", flush=True)
                action = rule.next_action()
                _REC.add(input_rgb_arr, input_state, action)
                _apply_perturb(TASK_ENV, step)
                TASK_ENV.take_action(action)
                observation = TASK_ENV.get_obs()
                input_rgb_arr, input_state = encode_obs(observation)
                model.update_observation_window(input_rgb_arr, input_state)
                if bool(getattr(TASK_ENV, "eval_success", False)):
                    _REC.ep_success = 1
                    break
                continue
            if os.environ.get("CS_PLAIN", "0") == "1":
                if rule.chunk is not None and rule.offset < len(rule.chunk):
                    _cur = np.asarray(rule.chunk[rule.offset], np.float64)
                    _cs = []
                    for _c in cands:
                        _cd = np.asarray(_c[0], np.float64)
                        _n = np.linalg.norm(_cur) * np.linalg.norm(_cd)
                        _cs.append(float(_cur @ _cd / _n) if _n > 1e-9 else 0.0)
                    _mcs = float(np.mean(_cs))
                else:
                    _mcs = 1.0
                _use = _mcs > DELTA
                rule.log.append((step, _mcs, _use))
                if _use or rule.chunk is None or rule.offset >= len(rule.chunk):
                    rule.set_chunk(cands[qbest]); n_replan += 1
                    if os.environ.get("CS_VERBOSE","0") == "1":
                        print(f"[plain] step={step} cs={_mcs:.4f} USE q={np.round(qv,3)} pick={qbest}", flush=True)
                else:
                    n_keep += 1
                action = rule.next_action()
                _REC.add(input_rgb_arr, input_state, action)
                _apply_perturb(TASK_ENV, step)
                TASK_ENV.take_action(action)
                observation = TASK_ENV.get_obs()
                input_rgb_arr, input_state = encode_obs(observation)
                model.update_observation_window(input_rgb_arr, input_state)
                if bool(getattr(TASK_ENV, "eval_success", False)):
                    _REC.ep_success = 1
                    break
                continue
            if os.environ.get("CS_MODE_DELTA", "0") == "1":
                _K = int(os.environ.get("CS_HORIZON", "0"))
                if _K > 0:
                    _A = np.stack([np.asarray(c[min(_K, len(c)-1)], np.float64) for c in cands])
                    _d = _A - _A.mean(0, keepdims=True)
                else:
                    _d = np.stack([np.diff(np.asarray(c[:CMPS], np.float64), axis=0).ravel() for c in cands])
                _dn = _d / (np.linalg.norm(_d, axis=1, keepdims=True) + 1e-9)
                _pw = _dn @ _dn.T
                _mcs = float(_pw[~np.eye(len(cands), dtype=bool)].mean())
                _PERSIST = int(os.environ.get("CS_PERSIST", "3"))
                if not hasattr(rule, "_hi"): rule._hi = 0
                rule._hi = rule._hi + 1 if _mcs > DELTA else 0
                _use = rule._hi >= _PERSIST
                rule.log.append((step, _mcs, _use))
                if _use: rule._hi = 0
                if _use or rule.chunk is None or rule.offset >= len(rule.chunk):
                    rule.set_chunk(cands[qbest]); n_replan += 1
                    if os.environ.get("CS_VERBOSE","0") == "1":
                        print(f"[dis] step={step} d={_mcs:.3f} REPLAN q={np.round(qv,3)} pick={qbest}", flush=True)
                else:
                    n_keep += 1
                action = rule.next_action()
                _REC.add(input_rgb_arr, input_state, action)
                _apply_perturb(TASK_ENV, step)
                TASK_ENV.take_action(action)
                observation = TASK_ENV.get_obs()
                input_rgb_arr, input_state = encode_obs(observation)
                model.update_observation_window(input_rgb_arr, input_state)
                if bool(getattr(TASK_ENV, "eval_success", False)):
                    _REC.ep_success = 1
                    break
                continue
            if rule.chunk is not None and rule.offset < len(rule.chunk):
                _cur = np.asarray(rule.chunk[rule.offset], np.float64)
                _css = []
                for _c in cands:
                    _cd = np.asarray(_c[0], np.float64)
                    _n = np.linalg.norm(_cur) * np.linalg.norm(_cd)
                    _css.append(float(_cur @ _cd / _n) if _n > 1e-9 else 0.0)
                _mcs = float(np.mean(_css))
            else:
                _mcs = 1.0
            _use = (_mcs < DELTA) if os.environ.get("CS_INVERT","0")=="1" else (_mcs > DELTA)
            rule.log.append((step, _mcs, _use))
            if _use:
                rule.set_chunk(cands[qbest]); n_replan += 1
                if os.environ.get("CS_VERBOSE","0") == "1":
                    print(f"[cs_q] step={step} mcs={_mcs:.4f} USE_SAMPLE q={np.round(qv,3)} pick={qbest}", flush=True)
            else:
                if rule.chunk is None or rule.offset >= len(rule.chunk):
                    rule.set_chunk(cands[qbest]); n_replan += 1
                    if os.environ.get("CS_VERBOSE","0") == "1":
                        print(f"[cs_q] step={step} mcs={_mcs:.4f} FORCED(chunk empty)", flush=True)
                else:
                    n_keep += 1
                    if os.environ.get("CS_VERBOSE","0") == "1":
                        print(f"[cs_q] step={step} mcs={_mcs:.4f} KEEP", flush=True)
        elif MODE != "fixed" and rule.should_check(step):
            _nd = int(os.environ.get("CS_DIAG_N", "0"))
            if _nd > 1:
                _cs_list = [np.asarray(model.get_action()[:H], np.float32) for _ in range(_nd)]
                _f = np.stack([c[:CMPS].ravel() for c in _cs_list])
                _fn = _f / (np.linalg.norm(_f, axis=1, keepdims=True) + 1e-9)
                _pw = _fn @ _fn.T
                _off = _pw[~np.eye(_nd, dtype=bool)]
                _d = np.stack([np.diff(c[:CMPS], axis=0).ravel() for c in _cs_list])
                _dn = _d / (np.linalg.norm(_d, axis=1, keepdims=True) + 1e-9)
                _pwd = _dn @ _dn.T
                _offd = _pwd[~np.eye(_nd, dtype=bool)]
                print(f"[diag] step={step} raw[min={_off.min():.3f} mean={_off.mean():.3f}] "
                      f"delta[min={_offd.min():.3f} mean={_offd.mean():.3f}]", flush=True)
                cand = _cs_list[0]
            else:
                cand = np.asarray(model.get_action()[:H], np.float32)
            replan, cs = rule.decide(cand, step)
            if replan:
                rule.set_chunk(cand); n_replan += 1
            else:
                n_keep += 1
        elif rule.chunk is None or rule.offset >= len(rule.chunk):
            rule.set_chunk(np.asarray(model.get_action()[:H], np.float32))
            n_replan += 1

        _apply_perturb(TASK_ENV, step)
        action = rule.next_action()
        _REC.add(input_rgb_arr, input_state, action)
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

        if bool(getattr(TASK_ENV, "eval_success", False)):
            _REC.ep_success = 1
            break

    if os.environ.get("CS_DUMP", "0") == "1" and globals().get("_CSDUMP"):
        import pathlib as _pl
        _od = _pl.Path(os.environ.get("CASE_DUMP_DIR", "./case_study/dump"))
        _od.mkdir(parents=True, exist_ok=True)
        _sd = os.environ.get("PERTURB_SEED", "0")
        _rg = os.environ.get("CS_FIXRNG", "")
        _f = _od / (f"dump_s{_sd}_r{_rg}.npz" if _rg else f"dump_s{_sd}.npz")
        np.savez_compressed(_f, **{f"{k}_{d['step']}": d[k]
                                   for d in _CSDUMP
                                   for k in ("q", "top3", "top3_raw", "cur_raw")},
                            steps=np.array([d["step"] for d in _CSDUMP]),
                            gaps=np.array([d["gap"] for d in _CSDUMP]),
                            qcurs=np.array([d["qcur"] for d in _CSDUMP]),
                            offs=np.array([d["cur_off"] for d in _CSDUMP]),
                            replans=np.array([d["replan"] for d in _CSDUMP]))
        print(f"[dump] saved {_f} ({len(_CSDUMP)} steps)", flush=True)
        _CSDUMP.clear()

    if os.environ.get("CS_PROFILE", "0") == "1":
        cs_vals = np.array([c for _, c, _ in rule.log if c == c])
        if len(cs_vals):
            print(f"[cs] mode={MODE} replan={n_replan} keep={n_keep} n={len(cs_vals)} "
                  f"mean={cs_vals.mean():.4f} min={cs_vals.min():.4f} "
                  f"p5={np.percentile(cs_vals,5):.4f} p10={np.percentile(cs_vals,10):.4f} "
                  f"p25={np.percentile(cs_vals,25):.4f} p50={np.percentile(cs_vals,50):.4f}", flush=True)
            print("[cs] series=" + ",".join(f"{s}:{c:.3f}" for s, c, _ in rule.log if c == c), flush=True)
        else:
            print(f"[cs] mode={MODE} replan={n_replan} keep={n_keep} (no cs)", flush=True)


def reset_model(model):
    _REC.flush()
    model.reset_obsrvationwindows()
