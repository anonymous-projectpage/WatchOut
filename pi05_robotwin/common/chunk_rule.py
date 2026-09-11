"""Disagreement-based adaptive chunk execution.

The policy server returns M candidate chunks per query. We measure how much the
candidates disagree with each other (self), and how much they disagree with the
chunk currently being executed (cross). When the gap between the two exceeds a
threshold, the current chunk has become stale and we replan.

    self  = mean pairwise cosine similarity among the first actions of the M candidates
    cross = mean cosine similarity between those first actions and the action the
            running chunk prescribes for the current offset
    gap   = self - cross
    replan if gap > delta

Cosine similarity is computed on the *normalized* action space (the raw model
output before un-normalization). Using un-normalized actions breaks the measure
whenever the action space has a large constant offset -- e.g. absolute joint
angles in degrees -- because every candidate then points in nearly the same
direction and the gap collapses to zero.
"""

import numpy as np

EPS = 1e-12


def self_similarity(first_actions):
    """Mean pairwise cosine similarity among candidates, excluding self-pairs.

    Args:
        first_actions: (M, D) first action of each candidate, normalized space.
    Returns:
        float in [-1, 1]; nan when M < 2.
    """
    C = np.asarray(first_actions, np.float64)
    M = len(C)
    if M < 2:
        return float("nan")
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + EPS)
    S = Cn @ Cn.T
    return float(S[~np.eye(M, dtype=bool)].mean())


def disagreement_gap(first_actions, running_action):
    """gap = self - cross.

    Args:
        first_actions:  (M, D) first action of each candidate, normalized space.
        running_action: (D,) action the running chunk prescribes for the current
                        offset, normalized space. None when no chunk is running.
    Returns:
        (gap, self_cs). gap is +inf when no chunk is running, which forces a replan.
    """
    C = np.asarray(first_actions, np.float64)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + EPS)
    M = len(Cn)
    self_cs = float((Cn @ Cn.T)[~np.eye(M, dtype=bool)].mean()) if M > 1 else float("nan")

    if running_action is None:
        return float("inf"), self_cs
    cu = np.asarray(running_action, np.float64)
    cu = cu / (np.linalg.norm(cu) + EPS)
    return self_cs - float((Cn @ cu).mean()), self_cs


class AdaptiveChunk:
    """Holds the running chunk and decides when to replan.

    Usage per control step:
        raw, exe, q = policy.sample_candidates(obs, M)   # raw: normalized, exe: executable
        action, info = rule.step(raw, exe, q)
        env.step(action)
    """

    def __init__(self, delta, act_dims=None):
        self.delta = float(delta)
        self.act_dims = act_dims

        self.reset()

    def reset(self):
        self.chunk = None
        self.chunk_raw = None
        self.offset = 0
        self.n_replan = 0
        self.step_idx = 0

    def step(self, raw, exe, q):
        raw = np.asarray(raw, np.float64)
        if self.act_dims is not None:
            raw = raw[:, :, self.act_dims]

        running = None
        if self.chunk_raw is not None and self.offset < len(self.chunk_raw):
            running = self.chunk_raw[self.offset]
        gap, self_cs = disagreement_gap(raw[:, 0, :], running)

        replan = (gap > self.delta) or (self.chunk is None) or (self.offset >= len(self.chunk))
        if replan:
            b = int(np.argmax(np.asarray(q).reshape(-1)))
            self.chunk, self.chunk_raw, self.offset = exe[b], raw[b], 0
            self.n_replan += 1
        else:
            b = None

        action = self.chunk[self.offset]
        self.offset += 1
        info = dict(step=self.step_idx, gap=gap, self_cs=self_cs,
                    replan=replan, pick=b, offset=self.offset)
        self.step_idx += 1
        return action, info
