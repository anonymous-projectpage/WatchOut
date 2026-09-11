"""
MuJoCo port of the RoboTwin _apply_perturb routine. Instead of the SAPIEN calls
set_linear_velocity / set_pose, this writes sim.data.qvel / qpos directly.

A single free joint occupies

    qpos[adr : adr+3]  = position
    qpos[adr+3: adr+7] = quaternion
    qvel[dof : dof+3]  = linear velocity
    qvel[dof+3: dof+6] = angular velocity

Environment variables (names kept identical to RoboTwin)
--------------------------------------------------------
PERTURB_RANDOM      "1" enables the perturbation
PERTURB_ACTOR       object name, e.g. porcelain_mug_1 (its joint is looked up as <name>_joint0)
PERTURB_SEED        seed used to draw the perturbation start step deterministically
PERTURB_STEP_RANGE  "15,30", range of the start step
PERTURB_DUR         number of steps the perturbation is applied for (default 100)
PERTURB_MIRROR      "1" pushes the object toward the mirrored position across the axis
PERTURB_AXIS        0=x, 1=y (default 1); left-right is the y axis in LIBERO-10
PERTURB_VEL         mirror-mode velocity in m/s (default 0.5)
PERTURB_MODE        velocity (default) | teleport
PERTURB_DIST        total displacement in teleport / non-mirror mode (default 0.10)
PERTURB_TOL         tolerance for deciding the target has been reached (default 0.01)
PERTURB_VERBOSE     "1" logs every step
"""
from __future__ import annotations

import os
import numpy as np


def _find_joint(sim, name: str):
    """Prefer <name>_joint0, fall back to name itself, otherwise None."""
    for cand in (f"{name}_joint0", name, f"{name}_joint"):
        try:
            return sim.model.joint_name2id(cand)
        except Exception:
            continue
    return None


def list_joints(sim):
    return [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]


def reset_perturb():
    """Call at the start of each episode to clear internal state."""
    for k in ("_PERT_STEP", "_PERT_TP", "_PERT_DONE"):
        os.environ.pop(k, None)


def apply_perturb(env, step: int):
    """Call every step, right before the action is executed."""
    if os.environ.get("PERTURB_RANDOM", "0") != "1":
        return

    sim = getattr(env, "sim", None)
    if sim is None:
        return

    name = os.environ.get("PERTURB_ACTOR", "")
    jid = _find_joint(sim, name)
    if jid is None:
        if step == 0:
            print(f"[perturb] joint for '{name}' not found. available:",
                  list_joints(sim), flush=True)
        return

    qadr = int(sim.model.jnt_qposadr[jid])
    vadr = int(sim.model.jnt_dofadr[jid])
    AX = int(os.environ.get("PERTURB_AXIS", "1"))


    if "_PERT_STEP" not in os.environ:
        sd = int(os.environ.get("PERTURB_SEED", "0"))
        lo, hi = [int(x) for x in
                  os.environ.get("PERTURB_STEP_RANGE", "15,30").split(",")]
        rg = np.random.RandomState(sd)
        st = int(rg.randint(lo, hi + 1))
        os.environ["_PERT_STEP"] = str(st)
        print(f"[perturb-rand] seed={sd} step={st} actor={name} axis={AX}",
              flush=True)

    st = int(os.environ["_PERT_STEP"])
    dur = int(os.environ.get("PERTURB_DUR", "100"))
    if step < st or step >= st + dur:
        return

    mode = os.environ.get("PERTURB_MODE", "velocity")
    tol = float(os.environ.get("PERTURB_TOL", "0.01"))
    p = float(sim.data.qpos[qadr + AX])


    if os.environ.get("PERTURB_MIRROR", "") == "1":
        tp_s = os.environ.get("_PERT_TP", "")
        if not tp_s:
            tp = -p
            os.environ["_PERT_TP"] = str(tp)
            print(f"[perturb] mirror start axis{AX}={p:+.3f} -> {tp:+.3f}",
                  flush=True)
        else:
            tp = float(tp_s)
    else:
        d = float(os.environ.get("PERTURB_DIST", "0.10"))
        tp_s = os.environ.get("_PERT_TP", "")
        if not tp_s:
            sd = int(os.environ.get("PERTURB_SEED", "0"))
            sgn = 1.0 if np.random.RandomState(sd + 1).rand() > 0.5 else -1.0
            tp = p + sgn * d
            os.environ["_PERT_TP"] = str(tp)
            print(f"[perturb] shift start axis{AX}={p:+.3f} -> {tp:+.3f}",
                  flush=True)
        else:
            tp = float(tp_s)


    if abs(p - tp) < tol:
        if os.environ.get("_PERT_DONE"):
            return
        if not os.environ.get("_PERT_DONE"):
            os.environ["_PERT_DONE"] = "1"
            print(f"[perturb] DONE step={step} axis{AX}={p:+.3f}", flush=True)
        return


    if os.environ.get("PERTURB_DRYRUN", "0") == "1":
        return
    if mode == "teleport":
        n = max(1, int(os.environ.get("MIRROR_STEPS", "40")))
        sim.data.qpos[qadr + AX] = p + (tp - p) / n

        if os.environ.get("PERTURB_LOCK", "0") == "1":
            sim.data.qvel[vadr:vadr + 6] = 0.0
        if os.environ.get("PERTURB_FWD", "0") == "1":
            sim.forward()
    else:
        v = float(os.environ.get("PERTURB_VEL", "0.5"))
        vel = [0.0, 0.0, 0.0]
        vel[AX] = v * (1.0 if tp > p else -1.0)
        sim.data.qvel[vadr:vadr + 3] = vel

    if os.environ.get("PERTURB_VERBOSE", "0") == "1":
        print(f"[perturb] step={step} {name} axis{AX}={p:+.3f} -> {tp:+.3f}",
              flush=True)
