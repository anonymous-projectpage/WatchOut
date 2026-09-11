import sys
from pathlib import Path

robowin_root = Path(os.environ["ROBOTWIN_ROOT"])

if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))

import os
os.chdir(robowin_root)

import argparse
import collections
from collections import Counter, defaultdict
import logging
import os
import importlib
import numpy as np
import torch
import yaml
import json_numpy
import requests
import PIL.Image as Image
from tqdm import tqdm
import traceback
import json
import random
import imageio


import numpy as np
import sapien as _sp
import numpy as _np

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
                if not os.environ.get("_PERT_DONE"):
                    os.environ["_PERT_DONE"] = "1"
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
                if not os.environ.get("_PERT_DONE"):
                    os.environ["_PERT_DONE"] = "1"
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


_PSTEP = [0]


import cv2
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.float32)

ALL_TASKS = [
   "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan", "place_mouse_pad",
    "place_object_basket",
    "place_object_scale", "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle",
    "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two", "stamp_seal", "turn_switch"
]

print("Number of tasks evaluating:", len(ALL_TASKS))

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6D_to_quat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_quat()

def decode_image_from_bytes(camera_rgb_image):
    if isinstance(camera_rgb_image, (bytes, bytearray)): camera_rgb_image = np.frombuffer(camera_rgb_image, dtype=np.uint8)
    rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8)
        if rgb.size == 2764800:
            rgb = rgb.reshape(720, 1280, 3)
        elif rgb.size == 921600:
            rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)


class ClientModel:
    def __init__(self, host, port):
        self.url = f"http://{host}:{port}/act"
        self.vision_record = []

    def set_instruction(self, instruction):
        self.instruction = instruction

    def return_vision_record(self):
        video = np.stack(self.vision_record)
        self.vision_record = []
        return video

    def step(self, obs):
        head_view = obs['observation']['head_camera']['rgb']
        left_view = obs['observation']['left_camera']['rgb']
        right_view = obs['observation']['right_camera']['rgb']
        front_view = obs['observation']['front_camera']['rgb']
        image_obs = np.stack([head_view, left_view, right_view, front_view])[None, ]
        left_ee = np.expand_dims(np.array(obs["endpose"]["left_endpose"]), axis=0)
        right_ee = np.expand_dims(np.array(obs["endpose"]["right_endpose"]), axis=0)
        left_grip = np.expand_dims(np.array(obs["endpose"]["left_gripper"]), axis=0)
        right_grip = np.expand_dims(np.array(obs["endpose"]["right_gripper"]), axis=0)
        left_grip = 1 - left_grip * 2
        right_grip = 1 - right_grip * 2
        abs_eef = np.concatenate([
            left_ee[:, :3],
            quat_to_rotate6D(left_ee[:, 3:]),
            left_grip[:, None],
            right_ee[:, :3],
            quat_to_rotate6D(right_ee[:, 3:]),
            right_grip[:, None]
        ], axis=-1)
        self.vision_record.append(image_obs)
        query = {
                "domain_id": 6,
                "proprio": json_numpy.dumps(abs_eef.squeeze(0)),
                "language_instruction": self.instruction,
                "image0": json_numpy.dumps(head_view),
                "image1": json_numpy.dumps(left_view),
                "image2": json_numpy.dumps(right_view)}

        response = requests.post(self.url, json=query)
        action = np.array(response.json()['action'])
        return action

    def step_cands(self, obs, n_cand=16):
        """Return n candidates and their Q values: (raw [N,H,A], exe [N,H,A], q [N])."""
        head_view = obs['observation']['head_camera']['rgb']
        left_view = obs['observation']['left_camera']['rgb']
        right_view = obs['observation']['right_camera']['rgb']
        left_ee = np.expand_dims(np.array(obs["endpose"]["left_endpose"]), axis=0)
        right_ee = np.expand_dims(np.array(obs["endpose"]["right_endpose"]), axis=0)
        left_grip = np.expand_dims(np.array(obs["endpose"]["left_gripper"]), axis=0)
        right_grip = np.expand_dims(np.array(obs["endpose"]["right_gripper"]), axis=0)
        left_grip = 1 - left_grip * 2
        right_grip = 1 - right_grip * 2
        abs_eef = np.concatenate([
            left_ee[:, :3], quat_to_rotate6D(left_ee[:, 3:]), left_grip[:, None],
            right_ee[:, :3], quat_to_rotate6D(right_ee[:, 3:]), right_grip[:, None],
        ], axis=-1)
        query = {"domain_id": 6,
                 "proprio": json_numpy.dumps(abs_eef.squeeze(0)),
                 "language_instruction": self.instruction,
                 "image0": json_numpy.dumps(head_view),
                 "image1": json_numpy.dumps(left_view),
                 "image2": json_numpy.dumps(right_view),
                 "n_cand": int(n_cand)}
        r = requests.post(self.url.replace("/act", "/act_cands"), json=query)
        d = r.json()
        return (np.array(d["raw"], np.float32),
                np.array(d["exe"], np.float32),
                np.array(d["q"], np.float32))

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_env(task_name, task_config):
    CONFIGS_PATH = "task_config"
    task = class_decorator(task_name)

    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    args["embodiment_name"] = "+".join(embodiment_type) if len(embodiment_type) > 1 else embodiment_type[0]
    args["task_config"] = task_config

    return task, args


def _to_rollout_action(a):
    """Convert model actions (H,20) into executable actions (H,16)."""
    a = np.asarray(a)
    lq = rotate6D_to_quat(a[:, 3:9])
    lg = 1 - 2 * (a[:, 9:10] > 0.7)
    rq = rotate6D_to_quat(a[:, 13:19])
    rg = 1 - 2 * (a[:, 19:20] > 0.7)
    return np.concatenate([a[:, :3], lq, lg, a[:, 10:13], rq, rg], axis=1)


def _rollout_te(env, policy, obs, images):
    """Temporal Ensemble (ACT): resample every step and exponentially average."""
    import sys as _sy
    _sy.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
    from aac_te import TemporalEnsemble
    LIM = int(os.environ.get("CS_STEP_LIM", "300"))
    H = int(os.environ.get("XVLA_HORIZON", "30"))
    te = TemporalEnsemble(horizon=H)
    for step in range(LIM):
        raw, exe, q = policy.step_cands(obs, 2)
        te.add_chunk(step, _to_rollout_action(exe[0]))
        action = te.action(step)
        if action is None:
            action = _to_rollout_action(exe[0])[0]
        _apply_perturb(env, _PSTEP[0]); _PSTEP[0] += 1
        env.take_action(action, action_type='ee')
        obs = env.get_obs()
        obs['endpose']['left_endpose'] = list(np.asarray(action[:7]).reshape(7,))
        obs['endpose']['right_endpose'] = list(np.asarray(action[8:-1]).reshape(7,))
        images.append(obs["observation"]["head_camera"]["rgb"])
        if env.check_success():
            print(f"\nsuccess! te steps={step+1}")
            env.suc += 1
            return 1, images
        if env.actor_pose == False:
            print("\nfail due to false actor_pose!")
            return 0, images
    print(f"\nfail! te steps={LIM}")
    return 0, images


def _rollout_aac(env, policy, obs, images):
    """AAC: pick chunk length h* from candidate entropy and execute that many steps."""
    import sys as _sy
    _sy.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
    from aac_te import aac_chunk_size, aac_config
    NC = int(os.environ.get("AAC_NCAND", "16"))
    LIM = int(os.environ.get("CS_STEP_LIM", "300"))
    cfg = aac_config()
    chunk, off, hstar = None, 0, 0
    n_replan = 0
    for step in range(LIM):
        if chunk is None or off >= hstar:
            raw, exe, q = policy.step_cands(obs, NC)
            hstar, _info = aac_chunk_size(raw, cfg)
            b = int(np.argmax(q))
            chunk = _to_rollout_action(exe[b])
            off = 0
            n_replan += 1
            if os.environ.get("AAC_VERBOSE", "0") == "1":
                print(f"[aac] step={step} h*={hstar} pick={b}", flush=True)
        action = chunk[off]; off += 1
        _apply_perturb(env, _PSTEP[0]); _PSTEP[0] += 1
        env.take_action(action, action_type='ee')
        obs = env.get_obs()
        obs['endpose']['left_endpose'] = list(action[:7].reshape(7,))
        obs['endpose']['right_endpose'] = list(action[8:-1].reshape(7,))
        images.append(obs["observation"]["head_camera"]["rgb"])
        if env.check_success():
            print(f"\nsuccess! aac replan={n_replan}")
            env.suc += 1
            return 1, images
        if env.actor_pose == False:
            print("\nfail due to false actor_pose!")
            return 0, images
    print(f"\nfail! aac replan={n_replan}")
    return 0, images


def _rollout_h1(env, policy, obs, images):
    """H=1: resample candidates every step and execute only the first action of argmax Q."""
    NC = int(os.environ.get("CS_NCAND", "16"))
    LIM = int(os.environ.get("CS_STEP_LIM", "300"))
    n_replan = 0
    for step in range(LIM):
        raw, exe, q = policy.step_cands(obs, NC)
        b = int(np.argmax(q))
        chunk = _to_rollout_action(exe[b])
        action = chunk[0]
        n_replan += 1
        _apply_perturb(env, _PSTEP[0]); _PSTEP[0] += 1
        env.take_action(action, action_type='ee')
        obs = env.get_obs()
        obs['endpose']['left_endpose'] = list(action[:7].reshape(7,))
        obs['endpose']['right_endpose'] = list(action[8:-1].reshape(7,))
        images.append(obs["observation"]["head_camera"]["rgb"])
        if env.check_success():
            print(f"\nsuccess! replan={n_replan} keep=0")
            env.suc += 1
            return 1, images
        if env.actor_pose == False:
            print("\nfail due to false actor_pose!")
            return 0, images
    print(f"\nfail! replan={n_replan} keep=0")
    return 0, images


def _rollout_bid(env, policy, obs, images):
    """BID: forward contrast (far from the weak policy) plus backward coherence (close to the previous chunk)."""
    NB  = int(os.environ.get("BID_NCAND", "16"))
    NW  = int(os.environ.get("BID_NWEAK", "16"))
    LAM = float(os.environ.get("BID_LAMBDA", "1.0"))
    KB  = int(os.environ.get("BID_HORIZON", "5"))
    LIM = int(os.environ.get("CS_STEP_LIM", "300"))
    WURL = os.environ.get("BID_WEAK_URL", "")
    VERB = os.environ.get("BID_VERBOSE", "0") == "1"
    if not WURL:
        raise RuntimeError("BID_WEAK_URL not set")
    chunk, off, prev = None, 0, None
    n_replan = n_keep = 0
    for step in range(LIM):
        if True:
            _, exe_s, _ = policy.step_cands(obs, NB)
            _url = policy.url
            try:
                policy.url = WURL
                _, exe_w, _ = policy.step_cands(obs, NW)
            finally:
                policy.url = _url
            C = np.stack([np.asarray(c[:KB], np.float64).ravel() for c in exe_s])
            W = np.stack([np.asarray(c[:KB], np.float64).ravel() for c in exe_w])
            if os.environ.get("BID_CENTER", "1") == "1":
                mu0 = np.concatenate([C, W]).mean(0, keepdims=True)
                C = C - mu0; W = W - mu0
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
            fc = -(Cn @ Wn.T).mean(axis=1)
            if prev is not None:
                pv = np.asarray(prev, np.float64).ravel()
                pv = pv / (np.linalg.norm(pv) + 1e-12)
                bc = Cn @ pv
            else:
                bc = np.zeros(len(Cn))
            sc = bc + LAM * fc
            pick = int(np.argmax(sc))
            chunk = _to_rollout_action(exe_s[pick]); off = 0
            prev = np.asarray(exe_s[pick][1:1+KB], np.float64)
            n_replan += 1
            if VERB:
                print(f"[bid] step={step} pick={pick} bc={bc[pick]:.3f} fc={fc[pick]:.3f}", flush=True)
        else:
            n_keep += 1
        action = chunk[off]; off += 1
        _apply_perturb(env, _PSTEP[0]); _PSTEP[0] += 1
        env.take_action(action, action_type='ee')
        obs = env.get_obs()
        obs['endpose']['left_endpose'] = list(np.asarray(action[:7]).reshape(7,))
        obs['endpose']['right_endpose'] = list(np.asarray(action[8:-1]).reshape(7,))
        images.append(obs["observation"]["head_camera"]["rgb"])
        if env.check_success():
            print(f"\nsuccess! bid replan={n_replan} keep={n_keep}")
            env.suc += 1
            return 1, images
        if env.actor_pose == False:
            print("\nfail due to false actor_pose!")
            return 0, images
    print(f"\nfail! bid replan={n_replan} keep={n_keep}")
    return 0, images


def _rollout_csq(env, policy, obs, images):
    """Proposed: sample candidates every step and replan on the disagreement gap."""
    DELTA = float(os.environ.get("CS_DELTA", "0.20"))
    NC = int(os.environ.get("CS_NCAND", "16"))
    LIM = int(os.environ.get("CS_STEP_LIM", "300"))
    VERB = os.environ.get("CS_VERBOSE", "0") == "1"
    chunk, chunk_raw, off = None, None, 0
    n_replan = n_keep = 0
    for step in range(LIM):
        raw, exe, q = policy.step_cands(obs, NC)
        C = raw[:, 0, :].astype(np.float64)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        M = len(Cn)
        pw = Cn @ Cn.T
        self_cs = float(pw[~np.eye(M, dtype=bool)].mean())
        if chunk_raw is not None and off < len(chunk_raw):
            cu = np.asarray(chunk_raw[off], np.float64)
            cu = cu / (np.linalg.norm(cu) + 1e-12)
            gap = self_cs - float((Cn @ cu).mean())
        else:
            gap = 1e9
        if gap > DELTA or chunk is None or off >= len(chunk):
            b = int(np.argmax(q))
            chunk = _to_rollout_action(exe[b]); chunk_raw = raw[b]; off = 0
            n_replan += 1
            if VERB: print(f"[gap] step={step} gap={gap:.5f} self={self_cs:.5f} REPLAN pick={b}", flush=True)
        else:
            n_keep += 1
            if VERB: print(f"[gap] step={step} gap={gap:.5f} self={self_cs:.5f} keep", flush=True)
        action = chunk[off]; off += 1
        _apply_perturb(env, _PSTEP[0]); _PSTEP[0] += 1
        env.take_action(action, action_type='ee')
        obs = env.get_obs()
        obs['endpose']['left_endpose'] = list(action[:7].reshape(7,))
        obs['endpose']['right_endpose'] = list(action[8:-1].reshape(7,))
        images.append(obs["observation"]["head_camera"]["rgb"])
        if env.check_success():
            print(f"\nsuccess! replan={n_replan} keep={n_keep}")
            env.suc += 1
            return 1, images
        if env.actor_pose == False:
            print("\nfail due to false actor_pose!")
            return 0, images
    print(f"\nfail! replan={n_replan} keep={n_keep}")
    return 0, images


def _rollout(env, policy):
    success_flag = False
    error_flag = False
    env._update_render()
    if env.render_freq: env.viewer.render()
    env.actor_pose = True
    images = []
    idx = 0
    _PSTEP[0] = 0
    os.environ.pop('_PERT_TX', None)
    os.environ.pop('_PERT_DONE', None)
    _PSTEP[0] = 0
    os.environ.pop('_PERT_TX', None)
    os.environ.pop('_PERT_DONE', None)
    obs = env.get_obs()
    _MODE = os.environ.get("CHUNK_MODE", "fixed")
    if _MODE == "cs_q":
        return _rollout_csq(env, policy, obs, images)
    if _MODE == "h1":
        return _rollout_h1(env, policy, obs, images)
    if _MODE == "te":
        return _rollout_te(env, policy, obs, images)
    if _MODE == "aac":
        return _rollout_aac(env, policy, obs, images)
    if _MODE == "bid":
        return _rollout_bid(env, policy, obs, images)

    for j in range(10):
        actions = policy.step(obs)
        left_xyz = actions[:, :3]
        left_rotate6d = actions[:, 3:9]
        left_gripper = actions[:, 9:10]

        left_quat = rotate6D_to_quat(left_rotate6d)
        left_grip = 1 - 2 * (left_gripper > 0.7)

        left_new = np.concatenate([left_xyz, left_quat, left_grip], axis=1)

        right_xyz = actions[:, 10:13].reshape(-1, 3)
        right_rotate6d = actions[:, 13:19].reshape(-1, 6)
        right_quat = rotate6D_to_quat(right_rotate6d)
        right_gripper = actions[:, 19:20].reshape(-1, 1)

        right_grip = 1 - 2 * (right_gripper > 0.7)
        right_new = np.concatenate([right_xyz, right_quat, right_grip], axis=1)

        rollout_action = np.concatenate([left_new, right_new], axis=1)
        for action in tqdm(rollout_action):
            if _PSTEP[0] < 3 or _PSTEP[0] == 20:
                print(f"[dbg] step={_PSTEP[0]} RANDOM={os.environ.get('PERTURB_RANDOM')} ACTOR={os.environ.get('PERTURB_ACTOR')}", flush=True)
            _apply_perturb(env, _PSTEP[0]); _PSTEP[0] += 1
            env.take_action(action, action_type='ee')
            obs = env.get_obs()
            obs['endpose']['left_endpose'] = list(action[:7].reshape(7,))
            obs['endpose']['right_endpose'] = list(action[8:-1].reshape(7,))
            images.append(obs["observation"]["head_camera"]["rgb"] )
            idx += 1
            if env.check_success():
                success_flag = True
                break
            if env.actor_pose == False:
                print('false actor_pose')
                error_flag = True
                break
        if error_flag:
            print("\nfail due to false actor_pose!")
            return 0, images

        if success_flag:
            print("\nsuccess!")
            env.suc +=1
            return 1, images

        if env.actor_pose == False:
            print('false actor_pose2')
            break

        j += 1
        env._update_render()
    print("\nfail!")
    return 0, images


def eval_episodes(task_name, task_config, policy, test_num=10, seed=0, eval_log_dir=None, instruction_type=None):
    """
    rollout several episodes and log the mean episode return
    """
    if not os.path.exists(os.path.join(eval_log_dir, task_name)):
        print('save to', os.path.join(eval_log_dir, task_name))
        os.makedirs(os.path.join(eval_log_dir, task_name))

    TASK_ENV, args = load_env(task_name, task_config)

    st_seed = 2000 * (1 + seed)

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args['policy_name'] = 'V4'

    args["eval_mode"] = True
    args["render_freq"] = 0
    args['ckpt_setting'] = '60k'
    while succ_seed < test_num:
        render_freq = args["render_freq"]
        print('Running test', now_id)
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except Exception as e:

                print("Error: ", e)

                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        instruction = args["task_name"].replace('_', ' ')
        print('instruction:', instruction)
        policy.set_instruction(instruction=instruction)

        try:
            status, images = _rollout(TASK_ENV, policy)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("ROLLOUT FAILED:", e, flush=True)
            TASK_ENV.close_env()
            now_seed += 1
            args["render_freq"] = render_freq
            continue
        save_path = f'{eval_log_dir}/{task_name}/{now_id}_{status}.mp4'
        save_video(save_path, images)

        metrics = {f'sim/{task_name}': status}
        save_path = f'{eval_log_dir}/results.json'
        _log_results(metrics, save_path)
        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(f"Success rate: {round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%")
        now_seed += 1
        try:
            point = round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)
            test_done = True
        except Exception as e:
            test_done = False
            print('redo tests due to:', e)
    metrics = {f'sim/{task_name}': point}
    _log_results(metrics, f'{eval_log_dir}/perc.json')

    return now_seed, TASK_ENV.suc


def save_video(output_path, frames, fps=30):
    print('saving video to ', output_path)
    imageio.mimsave(output_path, frames, fps=fps)


def _log_results(metrics, log_path):
    with open(log_path, 'a+') as f:
        line = json.dumps(metrics)
        f.write(line+'\n')


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default='8001', help="Your client port")
    parser.add_argument("--eval_log_dir", default=os.environ.get('EVAL_LOG_DIR', './eval_logs'), type=str, help="Where to log the evaluation results.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    parser.add_argument("--num_episodes", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (envs.<task>)")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name (without .yml)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the output video")
    parser.add_argument("--instruction_type", type=str, required=False, help="Where to save the output video")
    args = parser.parse_args()
    kwargs = vars(args)

    model = ClientModel(host=kwargs['host'], port=kwargs['port'])
    if args.task_name == 'all':
        for task in tqdm(ALL_TASKS):
            print(f"Evaluating task {task} for {kwargs['num_episodes']} episodes...")
            rewards = eval_episodes(task_name=task,
                                    task_config=args.task_config,
                                    policy=model,
                                    seed=args.seed,
                                    test_num=kwargs['num_episodes'],
                                    eval_log_dir=kwargs['eval_log_dir'],
                                    instruction_type=args.instruction_type)
    else:
        print(f"Evaluating task {args.task_name} for {kwargs['num_episodes']} episodes...")
        rewards = eval_episodes(task_name=args.task_name,
                                task_config=args.task_config,
                                policy=model,
                                seed=args.seed,
                                test_num=kwargs['num_episodes'],
                                eval_log_dir=kwargs['eval_log_dir'],
                                instruction_type=args.instruction_type)
if __name__ == "__main__":
    main()
