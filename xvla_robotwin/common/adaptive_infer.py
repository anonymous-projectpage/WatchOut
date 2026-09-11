"""Inference utilities for adaptive action chunking."""
import numpy as np
import jax
import jax.numpy as jnp
import openpi.models.pi0 as _pi0


def _fix_rng(policy):
    """Fix the policy RNG when CS_FIXRNG is set (for reproducible case studies)."""
    import os as _o, jax as _j
    fx = _o.environ.get("CS_FIXRNG", "")
    if fx and not getattr(policy, "_rng_fixed", False):
        policy._rng = _j.random.PRNGKey(int(fx))
        policy._rng_fixed = True
        print(f"[rng] fixed to {fx}", flush=True)


def compute_prefix(model, observation):
    """obs -> (prefix_output, prefix_mask). Independent of the action, so computed once."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    attn = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_output, _sfx), _ = model.PaliGemma.llm(
        [prefix_tokens, None], mask=attn, positions=positions, adarms_cond=[None, None]
    )
    return prefix_output, prefix_mask


def compute_q(model, observation, actions, prefix=None):
    """actions: (B, H, A). returns min(q1,q2): (B,)"""
    if prefix is None:
        prefix_output, prefix_mask = compute_prefix(model, observation)
    else:
        prefix_output, prefix_mask = prefix
    q1, q2 = model.chunk_critic(prefix_output, prefix_mask, observation.state, actions)
    return jnp.minimum(q1, q2)


def compute_q_candidates(model, observation, cand_actions, prefix=None):
    """cand_actions: (N, H, A), N candidates for the same observation. Returns (N,)."""
    if prefix is None:
        prefix_output, prefix_mask = compute_prefix(model, observation)
    else:
        prefix_output, prefix_mask = prefix
    N = cand_actions.shape[0]
    po = jnp.repeat(prefix_output, N, axis=0)
    pm = jnp.repeat(prefix_mask, N, axis=0)
    st = jnp.repeat(observation.state, N, axis=0)
    q1, q2 = model.chunk_critic(po, pm, st, cand_actions)
    return jnp.minimum(q1, q2)


def make_observation(policy, obs_dict):
    """Build a transformed Observation through the same path as Policy.infer()."""
    import openpi.models.model as _model
    inputs = jax.tree.map(lambda x: x, obs_dict)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    return _model.Observation.from_dict(inputs)


def sample_and_rank(policy, obs_dict, n_cand, horizon):
    """Sample n candidates and rank them by Q. Returns (cands (N,H,A), q (N,), best_idx)."""
    observation = make_observation(policy, obs_dict)
    model = policy._model

    cands = []
    for _ in range(n_cand):
        _fix_rng(policy)
        policy._rng, rng = jax.random.split(policy._rng)
        a = policy._sample_actions(rng, observation, **policy._sample_kwargs)
        cands.append(np.asarray(a[0, :horizon]))
    cands = np.stack(cands).astype(np.float32)

    prefix = compute_prefix(model, observation)
    q = compute_q_candidates(model, observation, jnp.asarray(cands), prefix=prefix)
    q = np.asarray(q)
    return cands, q, int(np.argmax(q))


_JIT_CACHE = {}


def make_jitted_q(model):
    """JIT-compile (obs, cands) -> q with the model state held fixed."""
    import flax.nnx as nnx

    key = id(model)
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]

    graphdef, state = nnx.split(model)

    @jax.jit
    def _fn(state, observation, cand_actions):
        m = nnx.merge(graphdef, state)
        prefix_output, prefix_mask = compute_prefix(m, observation)
        N = cand_actions.shape[0]
        po = jnp.repeat(prefix_output, N, axis=0)
        pm = jnp.repeat(prefix_mask, N, axis=0)
        st = jnp.repeat(observation.state, N, axis=0)
        q1, q2 = m.chunk_critic(po, pm, st, cand_actions)
        return jnp.minimum(q1, q2)

    def wrapped(observation, cand_actions):
        return _fn(state, observation, cand_actions)

    _JIT_CACHE[key] = wrapped
    return wrapped


def sample_and_rank_jit(policy, obs_dict, n_cand, horizon):
    """Sequential version: samples candidates one at a time via policy._sample_actions."""
    observation = make_observation(policy, obs_dict)
    qfn = make_jitted_q(policy._model)

    import openpi.models.model as _model
    _inp = jax.tree.map(lambda x: x, obs_dict)
    _inp = policy._input_transform(_inp)
    _inp = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], _inp)

    raw, exe = [], []
    for _ in range(n_cand):
        _fix_rng(policy)
        policy._rng, rng = jax.random.split(policy._rng)
        a = policy._sample_actions(rng, observation, **policy._sample_kwargs)
        raw.append(np.asarray(a[0, :horizon]))
        _o = {"state": _inp["state"], "actions": a}
        _o = jax.tree.map(lambda x: np.asarray(x[0, ...]), _o)
        _o = policy._output_transform(_o)
        exe.append(np.asarray(_o["actions"])[:horizon])
    raw = np.stack(raw).astype(np.float32)
    exe = np.stack(exe).astype(np.float32)

    q = np.asarray(qfn(observation, jnp.asarray(raw)))
    return exe, raw, q, int(np.argmax(q))


def sample_and_rank_batch(policy, obs_dict, n_cand, horizon):
    """Batched version: replicates the observation n_cand times and samples in one pass."""
    observation = make_observation(policy, obs_dict)
    qfn = make_jitted_q(policy._model)

    _inp = jax.tree.map(lambda x: x, obs_dict)
    _inp = policy._input_transform(_inp)
    _inp = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], _inp)

    N = int(n_cand)
    obs_b = jax.tree.map(
        lambda x: jnp.repeat(x, N, axis=0) if hasattr(x, "ndim") and x.ndim >= 1 else x,
        observation,
    )
    policy._rng, rng = jax.random.split(policy._rng)
    a = policy._sample_actions(rng, obs_b, **policy._sample_kwargs)

    raw = np.asarray(a[:, :horizon]).astype(np.float32)

    exe = []
    st = np.asarray(_inp["state"][0])
    for i in range(N):
        _o = {"state": st, "actions": np.asarray(a[i])}
        _o = policy._output_transform(_o)
        exe.append(np.asarray(_o["actions"])[:horizon])
    exe = np.stack(exe).astype(np.float32)

    q = np.asarray(qfn(observation, jnp.asarray(raw)))
    return exe, raw, q, int(np.argmax(q))
