"""
Adaptive Action Chunking (AAC) and Temporal Ensemble (ACT) baselines.

Default layout follows the RoboTwin aloha-agilex bimanual 14-dim joint space:
  dims 0-5   left arm joints    (continuous)
  dim  6     left gripper       (discrete)
  dims 7-12  right arm joints   (continuous)
  dim  13    right gripper      (discrete)

Environment variables
---------------------
AAC_NCAND      number of candidate samples N (default 20); must exceed the
               continuous-group dimension so the covariance is estimable
AAC_ALPHA      minimum action-magnitude threshold alpha (default 3.0); the scale
               differs from the paper because this is joint space
AAC_ALPHA_REL  relative alternative: alpha = AAC_ALPHA_REL * m[-1]
AAC_GRIP_TH    gripper-closed threshold (median over candidates when unset)
AAC_GRIP_MODE  discrete (default) | continuous | off
AAC_HMIN       lower bound on h* (default 1)
AAC_HMAX       upper bound on h* (default H)
AAC_CONT_DIMS  continuous groups, semicolon separated (default "0-5;7-12")
AAC_GRIP_DIMS  gripper dims, comma separated (default "6,13")
AAC_MAG_MODE   disp (default, |a_l - a_0|) | cumsum (accumulated deltas)
AAC_VERBOSE    1 prints h*, xi and the entropy curve on every replan

TE_M           exponential weighting coefficient m for Temporal Ensemble (default 0.01)
"""
import os
import numpy as np

_LOG2PIE = float(np.log(2.0 * np.pi * np.e))


def _parse_groups(spec):
    """'0-5;7-12' -> [[0..5], [7..12]]"""
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.append(list(range(int(a), int(b) + 1)))
        else:
            out.append([int(part)])
    return out


def _parse_dims(spec):
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def aac_config():
    return dict(
        alpha=float(os.environ.get("AAC_ALPHA", "3.0")),
        alpha_rel=(float(os.environ["AAC_ALPHA_REL"])
                   if os.environ.get("AAC_ALPHA_REL") else None),
        grip_th=(float(os.environ["AAC_GRIP_TH"])
                 if os.environ.get("AAC_GRIP_TH") else None),
        grip_mode=os.environ.get("AAC_GRIP_MODE", "discrete"),
        hmin=int(os.environ.get("AAC_HMIN", "1")),
        hmax=(int(os.environ["AAC_HMAX"]) if os.environ.get("AAC_HMAX") else None),
        cont_groups=_parse_groups(os.environ.get("AAC_CONT_DIMS", "0-5;7-12")),
        grip_dims=_parse_dims(os.environ.get("AAC_GRIP_DIMS", "6,13")),
        mag_mode=os.environ.get("AAC_MAG_MODE", "disp"),
        verbose=os.environ.get("AAC_VERBOSE", "0") == "1",
    )


def gaussian_entropy(X, jitter_rel=1e-6):
    """
    Eq.(3)  E = 0.5 * log[ (2*pi*e)^d * det(Sigma) ]
              = 0.5 * ( d*log(2*pi*e) + logdet(Sigma) )

    X: (N, d), the d-dimensional action of N candidates.
    When N <= d the covariance is singular and logdet becomes -inf; a diagonal jitter guards against this.
    """
    X = np.asarray(X, np.float64)
    N, d = X.shape
    if N < 2:
        return 0.0
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    tr = float(np.trace(S))
    eps = jitter_rel * (tr / max(d, 1)) if tr > 0 else 1e-12
    S = S + np.eye(d) * max(eps, 1e-12)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0 or not np.isfinite(logdet):
        return 0.0
    return 0.5 * (d * _LOG2PIE + float(logdet))


def discrete_entropy(v, th):
    """
    Eq.(2)  E = -sum p log p,  p = c/N  (c = number of closed states)
    v: (N,) gripper values
    """
    v = np.asarray(v, np.float64)
    N = len(v)
    if N == 0:
        return 0.0
    p = float(np.sum(v < th)) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def per_step_entropy(A, cfg):
    """
    A: (N, H, d) candidate actions
    Returns: (H,) total entropy per timestep (continuous groups + gripper).
    """
    N, H, d = A.shape
    th = cfg["grip_th"]
    if th is None and cfg["grip_dims"] and cfg["grip_mode"] == "discrete":
        gv = A[:, :, cfg["grip_dims"]]
        th = float(np.median(gv))

    E = np.zeros(H, np.float64)
    for i in range(H):
        e = 0.0
        for g in cfg["cont_groups"]:
            g = [x for x in g if x < d]
            if g:
                e += gaussian_entropy(A[:, i, g])
        if cfg["grip_mode"] == "discrete":
            for gd in cfg["grip_dims"]:
                if gd < d:
                    e += discrete_entropy(A[:, i, gd], th)
        elif cfg["grip_mode"] == "continuous":
            gd = [x for x in cfg["grip_dims"] if x < d]
            if gd:
                e += gaussian_entropy(A[:, i, gd])
        E[i] = e
    return E


def action_magnitude(a, cfg):
    """
    a: (H, d) mean action trajectory
    Returns: (H,) m(l) for l = 1..H

    disp   : assumes absolute joint targets. m(l) = ||a_l - a_0|| over continuous dims, plus gripper switches.
    cumsum : assumes delta actions. m(l) = ||sum_{i<l} da_i||
    """
    H, d = a.shape
    cont = [x for g in cfg["cont_groups"] for x in g if x < d]
    grip = [x for x in cfg["grip_dims"] if x < d]

    m = np.zeros(H, np.float64)
    if cont:
        if cfg["mag_mode"] == "cumsum":
            disp = np.cumsum(a[:, cont], axis=0)
        else:
            disp = a[:, cont] - a[0:1, cont]
        m += np.linalg.norm(disp, axis=1)

    if grip:
        th = cfg["grip_th"]
        if th is None:
            th = float(np.median(a[:, grip]))
        closed = (a[:, grip] < th).astype(np.float64)
        switched = np.zeros(H, np.float64)
        for k in range(len(grip)):
            ch = np.abs(np.diff(closed[:, k], prepend=closed[0, k]))
            switched += np.maximum.accumulate(ch)
        m += switched
    return m


def xi_from_magnitude(m, alpha):
    """Eq.(6)  xi = argmin_l ( m(l) > alpha ); returns H when no such l exists."""
    idx = np.nonzero(m > alpha)[0]
    return int(idx[0]) + 1 if len(idx) else len(m)


def aac_chunk_size(cands, cfg=None):
    """
    cands: (N, H, d), N action chunks sampled in parallel.
    Returns: (h_star, info_dict)

    Eq.(5)  h* = max( argmax_h( E_bar_{h+1} - E_bar_h ),  xi )
    """
    if cfg is None:
        cfg = aac_config()
    A = np.asarray(cands, np.float64)
    if A.ndim != 3:
        raise ValueError(f"cands must be (N,H,d), got {A.shape}")
    N, H, d = A.shape

    E = per_step_entropy(A, cfg)
    Ebar = np.cumsum(E) / np.arange(1, H + 1)
    diff = np.diff(Ebar)

    h_ent = int(np.argmax(diff)) + 1 if len(diff) else H

    mean_a = A.mean(axis=0)
    m = action_magnitude(mean_a, cfg)


    if cfg["alpha_rel"] is not None:
        alpha = cfg["alpha_rel"] * float(m[-1])
    else:
        alpha = cfg["alpha"]
    xi = xi_from_magnitude(m, alpha)

    h_star = max(h_ent, xi)
    hmax = cfg["hmax"] or H
    h_star = int(min(max(h_star, cfg["hmin"]), hmax, H))

    info = dict(h_ent=h_ent, xi=xi, h_star=h_star, alpha=alpha,
                E=E, Ebar=Ebar, mag=m, N=N, H=H)
    if cfg["verbose"]:
        print(f"[aac] h*={h_star} (h_ent={h_ent}, xi={xi})  "
              f"alpha={alpha:.3f} m[-1]={m[-1]:.3f}  "
              f"E {E[0]:.2f}->{E[-1]:.2f}", flush=True)
    return h_star, info


class TemporalEnsemble:
    """
    Samples a fresh chunk every step and exponentially averages all past chunks that predict the current step.

    As in ACT: w_i = exp(-m * i), where i=0 is the oldest prediction.
    Larger m reflects new observations faster. ACT uses m=0.01 by default.
    """

    def __init__(self, horizon, m=None):
        self.H = int(horizon)
        self.m = float(os.environ.get("TE_M", "0.01")) if m is None else float(m)
        self.buf = []

    def add_chunk(self, step, chunk):
        c = np.asarray(chunk, np.float64)
        self.buf.append((int(step), c))
        cutoff = step - self.H
        self.buf = [(s, a) for (s, a) in self.buf if s > cutoff]

    def action(self, step):
        """Exponentially weighted average over the chunks predicting this step."""
        preds = []
        for s, c in self.buf:
            k = step - s
            if 0 <= k < len(c):
                preds.append(c[k])
        if not preds:
            return None
        P = np.stack(preds)
        w = np.exp(-self.m * np.arange(len(P)))
        w = w / w.sum()
        return (P * w[:, None]).sum(axis=0)

    def reset(self):
        self.buf = []
