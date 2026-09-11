import collections
import os
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from libero_perturb import apply_perturb, reset_perturb

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:


    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5


    task_suite_name: str = (
        "libero_spatial"
    )
    num_steps_wait: int = 10
    num_trials_per_task: int = 50


    video_out_path: str = "data/libero/videos"

    seed: int = 7


def eval_libero(args: Args) -> None:

    np.random.seed(args.seed)


    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)


    total_episodes, total_successes = 0, 0
    _sel = os.environ.get("TASK_IDS", "")
    _ids = [int(x) for x in _sel.split(",") if x.strip()] if _sel else list(range(num_tasks_in_suite))
    for task_id in tqdm.tqdm(_ids):

        task = task_suite.get_task(task_id)


        initial_states = task_suite.get_task_init_states(task_id)


        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)


        task_episodes, task_successes = 0, 0
        _EP = os.environ.get("EPISODE_IDS", "")
        _eplist = [int(x) for x in _EP.split(",") if x.strip()] if _EP else None
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            if _eplist is not None and episode_idx not in _eplist:
                continue
            print(f"[case] === episode {episode_idx} start ===", flush=True)
            logging.info(f"\nTask: {task_description}")


            env.reset()
            action_plan = collections.deque()
            reset_perturb()
            _pstep = 0
            _cs_chunk, _cs_raw, _cs_off = None, None, 0
            _MODE = os.environ.get("CHUNK_MODE", "fixed")
            if _MODE == "bid" and "_wcli" not in dir():
                _WP = int(os.environ.get("BID_WEAK_PORT", "8030"))
                _wcli = _websocket_client_policy.WebsocketClientPolicy(args.host, _WP)
            _bid_prev = None
            _te = None
            if _MODE == "te":
                import sys as _sy; _sy.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
                from aac_te import TemporalEnsemble
                _te = TemporalEnsemble(horizon=int(os.environ.get("LB_HORIZON", "10")))
            _aac_h = 0
            _n_replan, _n_keep = 0, 0


            obs = env.set_init_state(initial_states[episode_idx])


            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:


                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue


                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))


                    replay_images.append(img)

                    _CS = _MODE == "cs_q"
                    _ADP = _MODE in ("cs_q", "te", "aac", "h1", "bid")
                    _need = (_MODE in ("cs_q", "te", "h1", "bid")) or (_MODE == "aac" and (_cs_chunk is None or _cs_off >= _aac_h))
                    if (_ADP and _need) or (not _ADP and not action_plan):


                        element = {
                            "observation/image":
                            img,
                            "observation/wrist_image":
                            wrist_img,
                            "observation/state":
                            np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt":
                            str(task_description),
                        }


                        if _MODE == "bid":
                            _NB = int(os.environ.get("BID_NCAND", "16"))
                            _NW = int(os.environ.get("BID_NWEAK", "16"))
                            _LAM = float(os.environ.get("BID_LAMBDA", "1.0"))
                            _KB = int(os.environ.get("BID_HORIZON", "5"))
                            _rs = client.infer({**element, "_n_cand": _NB})
                            _rw = _wcli.infer({**element, "_n_cand": _NW})
                            _es = np.asarray(_rs["exe"], np.float64)
                            _ew = np.asarray(_rw["exe"], np.float64)
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
                            _cs_chunk = np.asarray(_es[_pick], np.float32); _cs_off = 0
                            _bid_prev = np.asarray(_es[_pick][1:1+_KB], np.float64)
                            _n_replan += 1
                            if os.environ.get("BID_VERBOSE", "0") == "1":
                                print(f"[bid] step={_pstep} pick={_pick} bc={_bc[_pick]:.3f} fc={_fc[_pick]:.3f}", flush=True)
                            action_chunk = None
                        elif _MODE == "h1":
                            _r = client.infer({**element, "_n_cand": 1})
                            _exe = np.asarray(_r["exe"], np.float32)
                            _cs_chunk = np.asarray([_exe[0][0]], np.float32); _cs_off = 0
                            action_chunk = None
                        elif _MODE == "te":
                            _r = client.infer({**element, "_n_cand": 1})
                            _exe = np.asarray(_r["exe"], np.float32)
                            _te.add_chunk(_pstep, _exe[0])
                            _a = _te.action(_pstep)
                            _cs_chunk = np.asarray([_a if _a is not None else _exe[0][0]], np.float32)
                            _cs_off = 0
                            action_chunk = None
                        elif _MODE == "aac":
                            import sys as _sy2; _sy2.path.insert(0, os.environ.get("AAC_CORE", str(Path(__file__).resolve().parents[2] / "core")))
                            from aac_te import aac_chunk_size, aac_config
                            _M = int(os.environ.get("AAC_NCAND", "16"))
                            _r = client.infer({**element, "_n_cand": _M})
                            _raw = np.asarray(_r["raw"], np.float64)
                            _exe = np.asarray(_r["exe"], np.float32)
                            _q = np.asarray(_r["q"], np.float64).reshape(-1)
                            _aac_h, _info = aac_chunk_size(_raw, aac_config())
                            _b = int(np.argmax(_q))
                            _cs_chunk = _exe[_b]; _cs_raw = _raw[_b]; _cs_off = 0
                            _n_replan += 1
                            if os.environ.get("AAC_VERBOSE", "0") == "1":
                                print(f"[aac] step={_pstep} h*={_aac_h} pick={_b}", flush=True)
                            action_chunk = None
                        elif _CS:
                            _M = int(os.environ.get("CS_NCAND", "16"))
                            _DELTA = float(os.environ.get("CS_DELTA", "0.20"))
                            _r = client.infer({**element, "_n_cand": _M})
                            _raw = np.asarray(_r["raw"], np.float64)
                            _exe = np.asarray(_r["exe"], np.float32)
                            _q = np.asarray(_r["q"], np.float64).reshape(-1)
                            _C = _raw[:, 0, :]
                            _Cn = _C / (np.linalg.norm(_C, axis=1, keepdims=True) + 1e-12)
                            _nn = len(_Cn)
                            _pw = _Cn @ _Cn.T
                            _self = float(_pw[~np.eye(_nn, dtype=bool)].mean())
                            if _cs_raw is not None and _cs_off < len(_cs_raw):
                                _cu = np.asarray(_cs_raw[_cs_off], np.float64)
                                _cu = _cu / (np.linalg.norm(_cu) + 1e-12)
                                _gap = _self - float((_Cn @ _cu).mean())
                            else:
                                _gap = 1e9
                            if _gap > _DELTA or _cs_chunk is None or _cs_off >= len(_cs_chunk):
                                _b = int(np.argmax(_q))
                                _cs_chunk = _exe[_b]
                                _cs_raw = _raw[_b]
                                _cs_off = 0
                                _n_replan += 1
                                if os.environ.get("CS_VERBOSE", "0") == "1":
                                    print(f"[gap] step={_pstep} gap={_gap:.5f} self={_self:.5f} REPLAN pick={_b}", flush=True)
                            else:
                                _n_keep += 1
                                if os.environ.get("CS_VERBOSE", "0") == "1":
                                    print(f"[gap] step={_pstep} gap={_gap:.5f} self={_self:.5f} keep", flush=True)
                            action_chunk = None
                        else:
                            action_chunk = client.infer(element)["actions"]
                        if action_chunk is not None:
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[:args.replan_steps])

                    if _ADP:
                        action = _cs_chunk[_cs_off]
                        _cs_off += 1
                    else:
                        action = action_plan.popleft()


                    apply_perturb(env, _pstep); _pstep += 1
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1


            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )


            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")


        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):

        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
