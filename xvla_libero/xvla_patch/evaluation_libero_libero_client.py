from __future__ import annotations
import os
from pathlib import Path

"""
Refined Libero evaluation client.
- Stronger typing, docstrings, and logging
- Safer HTTP client with timeouts & error handling
- Cleaner action post-processing and 6D-rotation helpers
- Deterministic seeding and robust main() patterned after the provided example
"""

import argparse
import collections
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import imageio
import json_numpy
import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import robosuite.utils.transform_utils as T


EPS = 1e-6

LIBERO_DATASETS = {
    "libero_goal": ["libero_goal"],
    "libero_object": ["libero_object"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero30": ["libero_goal", "libero_object", "libero_spatial"],
    "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"],
}

LIBERO_DATASETS_HORIZON = {
    "libero_goal": 800,
    "libero_object": 800,
    "libero_spatial": 800,
    "libero_10": 900,
    "libero_90": 800,
    "libero30": 800,
    "libero130": 800,
}

benchmark_dict = benchmark.get_benchmark_dict()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    """Match original code behavior: vertical+horizontal flips."""
    return np.flip(np.flip(img, 0), 1)


class LiberoAbsActionProcessor:
    """Helpers to convert between 6D rotation (Zhou et al.) and axis-angle."""

    def Rotate6D_to_AxisAngle(self, r6d: np.ndarray) -> np.ndarray:
        """Convert 6D rotation representation to axis-angle.

        Args:
            r6d: array with shape (N, 6) or (6,)
        Returns:
            array with shape (N, 3) or (3,)
        """
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]


        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)


        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)


        b3 = np.cross(b1, b2, axis=-1)

        R = np.stack([b1, b2, b3], axis=-1)

        axis_angle_list: List[np.ndarray] = []
        for i in range(R.shape[0]):
            quat = T.mat2quat(R[i])
            axis_angle = T.quat2axisangle(quat)
            axis_angle_list.append(axis_angle)

        axis_angle_array = np.stack(axis_angle_list, axis=0)
        return axis_angle_array[0] if single else axis_angle_array

    def Mat_to_Rotate6D(self, R: np.ndarray) -> np.ndarray:
        if R.ndim == 2:
            return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
        elif R.ndim == 3:
            return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
        else:
            raise ValueError("Rotation matrix must be (...,3,3)")

    def AxisAngle_to_Rotate6D(self, aa: np.ndarray) -> np.ndarray:
        if aa.ndim == 1:
            return self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(aa)))
        else:
            raise ValueError("Only 1D axis-angle supported here")

    def action_6d_to_axisangle(self, action: np.ndarray) -> np.ndarray:
        """Convert action [..., 3(pos)+6(rot6d)+1(grip)] -> [..., 3(pos)+3(aa)+1(grip)]"""
        if action.ndim == 1:
            final_ori = self.Rotate6D_to_AxisAngle(action[3:9])
            return np.concatenate([action[0:3], final_ori, action[-1:]])
        elif action.ndim == 2:
            final_ori = self.Rotate6D_to_AxisAngle(action[:, 3:9])
            return np.concatenate([action[:, 0:3], final_ori, action[:, -1:]], axis=-1)
        else:
            raise ValueError("Unsupported action shape")


class ClientModel:
    """Thin HTTP client that queries a remote policy server and returns actions."""

    def __init__(self, host: str, port: int):
        self.url = f"http://{host}:{port}/act"
        self.processor = LiberoAbsActionProcessor()
        self.reset()

    def reset(self) -> None:
        self.proprio: Optional[np.ndarray] = None
        self.action_plan: Deque[List[float]] = collections.deque()

    def _format_query(self, obs: Dict, goal: str) -> Dict:
        main_view = _flip_agentview(obs["agentview_image"])
        wrist_view = obs["robot0_eye_in_hand_image"]

        closed_loop_proprio = np.concatenate([obs['robo_pos'], obs['robo_ori'], np.array([0.0])], axis=-1)
        closed_loop_proprio = np.concatenate([closed_loop_proprio, np.zeros_like(closed_loop_proprio)], axis=-1)
        if self.proprio is None:

            self.proprio = closed_loop_proprio

        return {
            "proprio": json_numpy.dumps(self.proprio),
            "language_instruction": goal,
            "image0": json_numpy.dumps(main_view),
            "image1": json_numpy.dumps(wrist_view),
            "domain_id": 3,
            "steps": 10,
        }

    def _post(self, payload: Dict) -> np.ndarray:
        try:
            resp = requests.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

        action = np.array(data["action"])
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
        return action

    def step_cands(self, obs: Dict, goal: str, n_cand: int):
        """Return M candidates and their Q values. raw=[M,T,10] (server output), exe=[M,T,7] (LIBERO actions)."""
        payload = self._format_query(obs, goal)
        payload["n_cand"] = int(n_cand)
        r = requests.post(self.url.replace("/act", "/act_cands"), json=payload)
        r.raise_for_status()
        d = r.json()
        raw  = np.asarray(d["raw"], np.float32)
        exe0 = np.asarray(d["exe"], np.float32)
        q    = np.asarray(d["q"], np.float64).reshape(-1)
        act = []
        for c in exe0:
            c = c[:, :10]
            aa = self.processor.Rotate6D_to_AxisAngle(c[:, 3:9])
            g  = np.where(c[:, 9:10] > 0.5, 1.0, -1.0)
            act.append(np.concatenate([c[:, :3], aa, g], axis=-1))
        self._last_exe = exe0
        return raw, np.asarray(act, np.float32), q

    def sync_proprio(self, idx: int) -> None:
        self.proprio[:9] = np.asarray(self._last_exe)[idx][-1, :9].copy()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            payload = self._format_query(obs, goal)
            action = self._post(payload)


            self.proprio[:9] = action[-1, :9].copy()


            target_eef = action[:, :3]
            target_axis = self.processor.Rotate6D_to_AxisAngle(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)


            for row in final_action.tolist():
                self.action_plan.append(row)

        action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)

        action_predict[-1] = 1.0 if action_predict[-1] > 0.5 else -1.0
        return action_predict


class LIBEROEval:
    def __init__(
        self,
        task_suite_name: str,
        eval_horizon: int = 600,
        act_type: str = "abs",
        num_episodes: int = 10,
        eval_freq: int = 10,
        init_seed: int = 42,
    ) -> None:
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[self.task_suite_name]
        self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
        self.eval_horizon = int(os.environ.get("EVAL_HORIZON", eval_horizon))
        self.num_episodes = num_episodes
        self.eval_freq = eval_freq
        self.init_seed = init_seed
        self.act_type = act_type
        self.processor = LiberoAbsActionProcessor()
        self.base_dir: Path = Path('.')


    def _make_dir(self, save_path: Path) -> None:
        path = save_path / self.task_suite_name
        _ensure_dir(path)
        self.base_dir = path

    def _init_env(self, task_suite, task_id: int = 0, ep: int = 0):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(
            f"[info] retrieving task {task_id} from suite {self.task_suite_name}, "
            f"language: {task_description}, bddl: {task_bddl_file}"
        )

        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
        env = OffScreenRenderEnv(**env_args)


        env.seed(self.init_seed + ep + 100)
        obs = env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        init_state_id = ep % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_id])


        for _ in range(10):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            obs, reward, done, info = env.step(action)

        if self.act_type == 'abs':
            for robot in env.env.robots:
                robot.controller.use_delta = False
        elif self.act_type == 'rel':
            pass
        else:
            raise ValueError("act_type must be 'abs' or 'rel'")

        return env, task_description, obs

    def _log_results(self, metrics: Dict) -> None:
        print(metrics)
        save_name = self.base_dir / 'results.json'
        with open(save_name, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + "\n")

    def _save_video(self, save_path: Path, images: List[np.ndarray], fps: int = 30) -> None:
        imageio.mimsave(save_path.as_posix(), images, fps=fps)

    def _rollout(self, task_suite, policy: ClientModel, task_id: int, ep: int) -> float:
        env, lang, obs = self._init_env(task_suite, task_id, ep)
        images: List[np.ndarray] = []

        done_flag = False
        import sys as _sy
        _sy.path.insert(0, os.environ.get("AAC_PERTURB", str(Path(__file__).resolve().parent)))
        from libero_perturb import apply_perturb, reset_perturb
        reset_perturb()
        _MODE = os.environ.get("CHUNK_MODE", "fixed")
        _te = None
        if _MODE == "te":
            import sys as _sy; _sy.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
            from aac_te import TemporalEnsemble
            _te = TemporalEnsemble(horizon=int(os.environ.get("XV_HORIZON", "30")))
            _te_t = 0
        _aac_h, _aac_off, _aac_ck = 0, 0, None
        if _MODE == "aac":
            import sys as _sy; _sy.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
            from aac_te import aac_chunk_size, aac_config
            _aac_cfg = aac_config()
        _bid_prev, _bid_ck, _bid_off = None, None, 0
        if _MODE == "bid":
            import requests as _rq
            _WP = int(os.environ.get("BID_WEAK_PORT", "8180"))
        _M    = int(os.environ.get("CS_NCAND", "16"))
        _DEL  = float(os.environ.get("CS_DELTA", "0.20"))
        _VERB = os.environ.get("CS_VERBOSE", "0") == "1"
        _ck, _ckraw, _off = None, None, 0
        _pstep, _nrep = 0, 0
        for _ in tqdm(range(self.eval_horizon), desc=f'{lang}'):
            robo_ori = self.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
            robo_pos = env.env.robots[0].controller.ee_pos
            obs['robo_ori'] = robo_ori
            obs['robo_pos'] = robo_pos

            if _MODE == "bid":
                _KB  = int(os.environ.get("BID_HORIZON", "5"))
                _LAM = float(os.environ.get("BID_LAMBDA", "1.0"))
                if True:
                    _NB = int(os.environ.get("BID_NCAND", "16"))
                    _NW = int(os.environ.get("BID_NWEAK", "16"))
                    _, _es, _ = policy.step_cands(obs, lang, _NB)
                    _pay = policy._format_query(obs, lang); _pay["n_cand"] = _NW
                    _rw = _rq.post(policy.url.replace("/act", "/act_cands").replace(
                            str(policy.url.split(":")[-1].split("/")[0]), str(_WP)), json=_pay)
                    _rw.raise_for_status()
                    _ew0 = np.asarray(_rw.json()["exe"], np.float64)[:, :, :10]
                    _aa = np.stack([policy.processor.Rotate6D_to_AxisAngle(c[:, 3:9]) for c in _ew0])
                    _ew = np.concatenate([_ew0[:, :, :3], _aa,
                          np.where(_ew0[:, :, 9:10] > 0.5, 1.0, -1.0)], axis=-1)
                    _es = np.asarray(_es, np.float64)
                    _C = np.stack([c[:_KB].ravel() for c in _es])
                    _W = np.stack([c[:_KB].ravel() for c in _ew])
                    if os.environ.get("BID_CENTER", "1") == "1":
                        _mu0 = np.concatenate([_C, _W]).mean(0, keepdims=True)
                        _C = _C - _mu0; _W = _W - _mu0
                    _Cn = _C / (np.linalg.norm(_C, axis=1, keepdims=True) + 1e-12)
                    _Wn = _W / (np.linalg.norm(_W, axis=1, keepdims=True) + 1e-12)
                    _fc = -(_Cn @ _Wn.T).mean(axis=1)
                    if _bid_prev is not None:
                        _pv = np.asarray(_bid_prev, np.float64).ravel()
                        _pv = _pv / (np.linalg.norm(_pv) + 1e-12)
                        _bc = _Cn @ _pv
                    else:
                        _bc = np.zeros(len(_Cn))
                    _pick = int(np.argmax(_bc + _LAM * _fc))
                    _bid_ck, _bid_off = _es[_pick], 0
                    _bid_prev = np.asarray(_es[_pick][1:1+_KB], np.float64)
                    policy.sync_proprio(_pick); _nrep += 1
                    if _VERB: print(f"[bid] step={_pstep} pick={_pick} "
                                    f"bc={_bc[_pick]:.3f} fc={_fc[_pick]:.3f}", flush=True)
                action = _bid_ck[0]
            elif _MODE == "aac":
                if _aac_ck is None or _aac_off >= _aac_h:
                    raw, exe, q = policy.step_cands(obs, lang, _M)
                    _aac_h, _info = aac_chunk_size(raw, _aac_cfg)
                    b = int(np.argmax(q))
                    _aac_ck, _aac_off = exe[b], 0
                    policy.sync_proprio(b); _nrep += 1
                    if _VERB: print(f"[aac] step={_pstep} h*={_aac_h} pick={b}", flush=True)
                action = _aac_ck[_aac_off]; _aac_off += 1
            elif _MODE == "te":
                raw, exe, q = policy.step_cands(obs, lang, 2)
                policy.sync_proprio(0)
                _ch = np.asarray(policy._last_exe)[0][:, :10].astype(np.float64)
                if _te_t < 3: print(f"[TE] t={_te_t} ch.shape={_ch.shape} ch[0]={np.round(_ch[0],3)}", flush=True)
                _te.add_chunk(_te_t, _ch)
                a = _te.action(_te_t)
                if a is None: a = _ch[0]
                a = np.asarray(a, np.float64).copy()
                _te_t += 1
                _u = a[3:6] / (np.linalg.norm(a[3:6]) + 1e-12)
                _v = a[6:9] - (_u @ a[6:9]) * _u
                _v = _v / (np.linalg.norm(_v) + 1e-12)
                _aa = policy.processor.Rotate6D_to_AxisAngle(
                        np.concatenate([_u, _v])[None, :].astype(np.float32))[0]
                action = np.concatenate([a[:3], _aa, [1.0 if a[9] > 0.5 else -1.0]])
            elif _MODE == "cs_q":
                raw, exe, q = policy.step_cands(obs, lang, _M)
                C  = raw[:, 0, :].astype(np.float64)
                Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                nn_ = len(Cn)
                self_cs = float((Cn @ Cn.T)[~np.eye(nn_, dtype=bool)].mean())
                if _ckraw is not None and _off < len(_ckraw):
                    cu = np.asarray(_ckraw[_off], np.float64)
                    cu = cu / (np.linalg.norm(cu) + 1e-12)
                    gap = self_cs - float((Cn @ cu).mean())
                else:
                    gap = 1e9
                if gap > _DEL or _ck is None or _off >= len(_ck):
                    b = int(np.argmax(q))
                    _ck, _ckraw, _off = exe[b], raw[b], 0
                    policy.sync_proprio(b); _nrep += 1
                    if _VERB: print(f"[gap] step={_pstep} gap={gap:.5f} self={self_cs:.5f} REPLAN pick={b}", flush=True)
                elif _VERB:
                    print(f"[gap] step={_pstep} gap={gap:.5f} self={self_cs:.5f} keep", flush=True)
                action = _ck[_off]; _off += 1
            elif _MODE == "h1":
                raw, exe, q = policy.step_cands(obs, lang, 1)
                policy.sync_proprio(0); action = exe[0][0]
            else:
                action = policy.step(obs, lang)

            images.append(_flip_agentview(obs['agentview_image']))
            apply_perturb(env, _pstep); _pstep += 1
            obs, reward, done, info = env.step(action)
            if done:
                done_flag = True
                break

        save_path = self.base_dir / f"{lang}_{ep}.mp4"
        self._save_video(save_path, images, fps=30)

        success = 1.0 if done_flag else 0.0
        metrics = {f'sim/{self.task_suite_name}/{lang}': success}
        self._log_results(metrics)

        env.close()
        return success


    def eval_episodes(self, policy: ClientModel, save_path: Path) -> float:
        self._make_dir(save_path)

        rews: List[float] = []
        for task_suite in self.task_suite_list:
            _sel = os.environ.get("TASK_IDS", "")
            _ids = [int(x) for x in _sel.split(",") if x.strip()] if _sel else list(range(len(task_suite.tasks)))
            _eps = os.environ.get("EPISODE_IDS", "")
            _eplist = [int(x) for x in _eps.split(",") if x.strip()] if _eps else None
            for task_id in tqdm(_ids, desc="Evaluating tasks"):
                for ep in range(self.num_episodes):
                    if _eplist is not None and ep not in _eplist:
                        continue
                    policy.reset()
                    rew = self._rollout(task_suite, policy, task_id, ep)
                    rews.append(rew)

        eval_rewards = float(sum(rews) / max(len(rews), 1))
        metrics = {f'sim_summary/{self.task_suite_name}/all': eval_rewards}
        self._log_results(metrics)
        return eval_rewards


def eval_libero(
    agent: ClientModel,
    save_path: Path,
    num_episodes: int = 10,
    init_seed: int = 42,
    act_type: str = 'abs',
    task_suites: Iterable[str] = ("libero_goal", "libero_spatial", "libero_10"),
) -> Dict[str, float]:
    result_dict: Dict[str, float] = {}
    for suite_name in task_suites:
        horizon = LIBERO_DATASETS_HORIZON[suite_name]
        evaluator = LIBEROEval(
            task_suite_name=suite_name,
            eval_horizon=horizon,
            act_type=act_type,
            num_episodes=num_episodes,
            init_seed=init_seed,
        )
        eval_rewards = evaluator.eval_episodes(agent, save_path=save_path)
        result_dict[suite_name] = eval_rewards


    with open((save_path / "results.json").as_posix(), "a+", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
        f.write("\n")
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")

    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server info.json (contains 'host' and 'port')")
    parser.add_argument("--server_ip", type=str, default=None,
                        help="Manual server IP (if not using --connection_info)")
    parser.add_argument("--server_port", type=int, default=None,
                        help="Manual server port (if not using --connection_info)")


    parser.add_argument("--output_dir", type=str, default="logs/",
                        help="Directory for saving evaluation videos and logs")
    parser.add_argument("--task_suites", nargs='+', default=["libero_10", "libero_spatial", "libero_goal", "libero_object"],
                        help="Libero suites to evaluate")
    parser.add_argument("--eval_time", type=int, default=50, help="Episodes per task")
    parser.add_argument("--init_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--act_type", type=str, default="abs", choices=["abs", "rel"], help="Action type")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    print("🚀 [Client] Starting LIBERO evaluation client...")


    if args.connection_info is not None:
        info_path = Path(args.connection_info)
        print(f"🔍 Waiting for connection info file: {info_path}")
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not info_path.exists():
            sys.stdout.write(f"\r{spinner[i % len(spinner)]} Waiting for server to start...")
            sys.stdout.flush()
            time.sleep(0.5)
            i += 1
        print("\n✅ Connection info file found!")
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                infos = json.load(f)
            host, port = infos["host"], int(infos["port"])
            print(f"🔗 Loaded server info: host={host}, port={port}")
        except Exception as e:
            print(f"❌ Failed to read connection info: {e}")
            sys.exit(1)
    else:
        if not args.server_ip or not args.server_port:
            print("❌ Must specify either --connection_info or both --server_ip and --server_port.")
            sys.exit(1)
        host, port = args.server_ip, int(args.server_port)
        print(f"🔗 Using manual server address: {host}:{port}")


    print(f"🛰️  Connecting to policy server at {host}:{port} ...")
    client = ClientModel(host, port)
    print("✅ Successfully initialized client!")


    print("🎯 Starting LIBERO policy evaluation...")
    print(f"📁 Results and videos will be saved to: {out_dir.resolve()}")
    print("-" * 88)
    print("init seed:", args.init_seed)
    print("task suites:", args.task_suites)
    print("episodes per task:", args.eval_time)
    print("action type:", args.act_type)
    print("-" * 88)

    try:
        eval_results = eval_libero(
            agent=client,
            save_path=out_dir,
            init_seed=args.init_seed,
            num_episodes=args.eval_time,
            task_suites=args.task_suites,
            act_type=args.act_type,
        )
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(2)

    print("\n✅ All evaluations completed successfully!")
    print(f"📊 Summary: {json.dumps(eval_results, indent=2)}")
