from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.shared import array_typing as at


class CriticLayerNorm(nnx.Module):
    """Small explicit LayerNorm to avoid depending on NNX LayerNorm API details."""

    def __init__(self, width: int):
        self.scale = nnx.Param(
            jnp.ones((width,), dtype=jnp.float32)
        )
        self.bias = nnx.Param(
            jnp.zeros((width,), dtype=jnp.float32)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(
            jnp.square(x - mean),
            axis=-1,
            keepdims=True,
        )

        normalized = (
            x - mean
        ) * jax.lax.rsqrt(
            variance + 1e-5
        )

        return (
            normalized * self.scale.value
            + self.bias.value
        )


class CriticTransformerBlock(nnx.Module):
    """Self-attention Transformer block for chunk-value estimation."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
    ):
        if width % num_heads != 0:
            raise ValueError(
                f"critic width {width} must be divisible "
                f"by num_heads {num_heads}"
            )

        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads

        self.norm1 = CriticLayerNorm(width)
        self.norm2 = CriticLayerNorm(width)

        self.q_proj = nnx.Linear(
            width,
            width,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            width,
            width,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            width,
            width,
            rngs=rngs,
        )
        self.attention_out = nnx.Linear(
            width,
            width,
            rngs=rngs,
        )

        self.ffn_in = nnx.Linear(
            width,
            4 * width,
            rngs=rngs,
        )
        self.ffn_out = nnx.Linear(
            4 * width,
            width,
            rngs=rngs,
        )

    def _split_heads(
        self,
        x: jax.Array,
    ) -> jax.Array:
        batch_size, sequence_length, _ = x.shape

        x = x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return jnp.transpose(
            x,
            (0, 2, 1, 3),
        )

    def _merge_heads(
        self,
        x: jax.Array,
    ) -> jax.Array:
        x = jnp.transpose(
            x,
            (0, 2, 1, 3),
        )

        batch_size, sequence_length, _, _ = x.shape

        return x.reshape(
            batch_size,
            sequence_length,
            self.width,
        )

    def __call__(
        self,
        x: jax.Array,
    ) -> jax.Array:
        residual = x
        normalized = self.norm1(x)

        query = self._split_heads(
            self.q_proj(normalized)
        )
        key = self._split_heads(
            self.k_proj(normalized)
        )
        value = self._split_heads(
            self.v_proj(normalized)
        )

        attention_logits = jnp.einsum(
            "bhqd,bhkd->bhqk",
            query,
            key,
        )

        attention_logits = (
            attention_logits
            / jnp.sqrt(
                jnp.asarray(
                    self.head_dim,
                    dtype=jnp.float32,
                )
            )
        )

        attention_weights = jax.nn.softmax(
            attention_logits.astype(jnp.float32),
            axis=-1,
        ).astype(query.dtype)

        attended = jnp.einsum(
            "bhqk,bhkd->bhqd",
            attention_weights,
            value,
        )

        attended = self._merge_heads(attended)

        x = residual + self.attention_out(attended)

        residual = x
        normalized = self.norm2(x)

        hidden = self.ffn_in(normalized)
        hidden = nnx.swish(hidden)
        hidden = self.ffn_out(hidden)

        return residual + hidden


class ChunkQNetwork(nnx.Module):
    """Transformer critic Q(s, action_chunk)."""

    def __init__(
        self,
        *,
        context_dim: int,
        action_dim: int,
        action_horizon: int,
        width: int,
        num_layers: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.action_horizon = action_horizon
        self.width = width
        self.num_layers = num_layers

        self.context_proj = nnx.Linear(
            context_dim,
            width,
            rngs=rngs,
        )
        self.state_proj = nnx.Linear(
            action_dim,
            width,
            rngs=rngs,
        )
        self.action_proj = nnx.Linear(
            action_dim,
            width,
            rngs=rngs,
        )

        # Sequence:
        # [Q token, VLM context token, state token, 50 action tokens]
        sequence_length = action_horizon + 3

        self.q_token = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (1, 1, width),
                dtype=jnp.float32,
            )
        )

        self.position_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (1, sequence_length, width),
                dtype=jnp.float32,
            )
        )

        self.blocks = nnx.Dict({
            f"block_{index}": CriticTransformerBlock(
                width,
                num_heads,
                rngs=rngs,
            )
            for index in range(num_layers)
        })

        self.final_norm = CriticLayerNorm(width)

        self.q_out = nnx.Linear(
            width,
            1,
            rngs=rngs,
        )

    def __call__(
        self,
        prefix_output: jax.Array,
        prefix_mask: jax.Array,
        state: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        # NOTE: do NOT cast the full [b, prefix_len, dim] tensor to f32.
        state = state.astype(jnp.float32)
        actions = actions.astype(jnp.float32)

        mask = prefix_mask.astype(
            prefix_output.dtype
        )[..., None]

        context_sum = jnp.sum(
            prefix_output * mask,
            axis=1,
        )

        context_count = jnp.maximum(
            jnp.sum(mask, axis=1),
            1.0,
        )

        pooled_context = (
            context_sum / context_count
        )

        context_token = self.context_proj(
            pooled_context
        )[:, None, :]

        state_token = self.state_proj(
            state
        )[:, None, :]

        action_tokens = self.action_proj(
            actions
        )

        batch_size = state.shape[0]

        q_token = jnp.broadcast_to(
            self.q_token.value,
            (
                batch_size,
                1,
                self.width,
            ),
        )

        tokens = jnp.concatenate(
            [
                q_token,
                context_token,
                state_token,
                action_tokens,
            ],
            axis=1,
        )

        tokens = (
            tokens
            + self.position_embedding.value[
                :, :tokens.shape[1], :
            ]
        )

        for index in range(
            self.num_layers
        ):
            tokens = self.blocks[
                f"block_{index}"
            ](tokens)

        tokens = self.final_norm(tokens)

        q_value = self.q_out(
            tokens[:, 0, :]
        )

        return q_value[:, 0]


class StateValueNetwork(nnx.Module):
    """Transformer value network V(s) without action input."""

    def __init__(
        self,
        *,
        context_dim: int,
        action_dim: int,
        width: int,
        num_layers: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.width = width
        self.num_layers = num_layers

        self.context_proj = nnx.Linear(
            context_dim,
            width,
            rngs=rngs,
        )
        self.state_proj = nnx.Linear(
            action_dim,
            width,
            rngs=rngs,
        )

        # Sequence: [V token, VLM context token, state token]
        self.v_token = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (1, 1, width),
                dtype=jnp.float32,
            )
        )
        self.position_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (1, 3, width),
                dtype=jnp.float32,
            )
        )

        self.blocks = nnx.Dict({
            f"block_{index}": CriticTransformerBlock(
                width,
                num_heads,
                rngs=rngs,
            )
            for index in range(num_layers)
        })

        self.final_norm = CriticLayerNorm(width)
        self.v_out = nnx.Linear(
            width,
            1,
            rngs=rngs,
        )

    def __call__(
        self,
        prefix_output: jax.Array,
        prefix_mask: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        # NOTE: do NOT cast the full [b, prefix_len, dim] tensor to f32.
        state = state.astype(jnp.float32)

        mask = prefix_mask.astype(prefix_output.dtype)[..., None]
        context_sum = jnp.sum(
            prefix_output * mask,
            axis=1,
            dtype=jnp.float32,
        )
        context_count = jnp.maximum(
            jnp.sum(mask, axis=1, dtype=jnp.float32),
            1.0,
        )
        pooled_context = context_sum / context_count

        context_token = self.context_proj(
            pooled_context
        )[:, None, :]
        state_token = self.state_proj(
            state
        )[:, None, :]

        batch_size = state.shape[0]
        v_token = jnp.broadcast_to(
            self.v_token.value,
            (
                batch_size,
                1,
                self.width,
            ),
        )

        tokens = jnp.concatenate(
            [
                v_token,
                context_token,
                state_token,
            ],
            axis=1,
        )
        tokens = tokens + self.position_embedding.value

        for index in range(self.num_layers):
            tokens = self.blocks[
                f"block_{index}"
            ](tokens)

        tokens = self.final_norm(tokens)
        value = self.v_out(tokens[:, 0, :])
        return value[:, 0]


class TwinChunkCritic(nnx.Module):
    """Two independent chunk critics to reduce value overestimation later."""

    def __init__(
        self,
        *,
        context_dim: int,
        action_dim: int,
        action_horizon: int,
        width: int,
        num_layers: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        kwargs = {
            "context_dim": context_dim,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "width": width,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }

        self.q1 = ChunkQNetwork(
            **kwargs,
            rngs=rngs,
        )

        self.q2 = ChunkQNetwork(
            **kwargs,
            rngs=rngs,
        )

    def __call__(
        self,
        prefix_output: jax.Array,
        prefix_mask: jax.Array,
        state: jax.Array,
        actions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        q1 = self.q1(
            prefix_output,
            prefix_mask,
            state,
            actions,
        )

        q2 = self.q2(
            prefix_output,
            prefix_mask,
            state,
            actions,
        )

        return q1, q2


@dataclasses.dataclass(frozen=True)
class Pi0IQLConfig(_pi0_config.Pi0Config):
    """π0.5 Flow Actor + Twin Chunk Q + State Value + AWR Flow loss."""

    pi05: bool = True
    use_iql: bool = True

    critic_width: int = 256
    critic_num_layers: int = 3
    critic_num_heads: int = 8

    value_width: int = 256
    value_num_layers: int = 3
    value_num_heads: int = 8

    critic_gamma: float = 0.99
    q_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    expectile: float = 0.9
    advantage_beta: float = 3.0
    max_advantage_weight: float = 20.0
    normalize_advantage_weights: bool = True

    def create(
        self,
        rng: at.KeyArrayLike,
    ) -> "Pi0IQL":
        return Pi0IQL(
            self,
            rngs=nnx.Rngs(rng),
        )


class Pi0IQL(_pi0.Pi0):
    """π0 flow policy jointly trained with Q1/Q2, V, and AWR weighting."""

    def __init__(
        self,
        config: Pi0IQLConfig,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            config,
            rngs=rngs,
        )

        paligemma_config = _gemma.get_config(
            config.paligemma_variant
        )

        self.q_loss_weight = config.q_loss_weight
        self.value_loss_weight = config.value_loss_weight
        import os as _os_env
        self.expectile = float(_os_env.environ.get("EXPECTILE", config.expectile))
        print(f"[iql] expectile={self.expectile}", flush=True)
        import os as _os
        self.advantage_beta = float(_os.environ.get("ADV_BETA", config.advantage_beta))
        self.adv_standardize = _os.environ.get("ADV_STANDARDIZE", "0") == "1"
        self.actor_weight = float(_os.environ.get("ACTOR_WEIGHT", "1.0"))
        print(f"[iql] beta={self.advantage_beta} standardize={self.adv_standardize} actor_weight={self.actor_weight}", flush=True)
        self.max_advantage_weight = config.max_advantage_weight
        self.normalize_advantage_weights = (
            config.normalize_advantage_weights
        )

        self.chunk_critic = TwinChunkCritic(
            context_dim=paligemma_config.width,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            width=config.critic_width,
            num_layers=config.critic_num_layers,
            num_heads=config.critic_num_heads,
            rngs=rngs,
        )

        self.value_network = StateValueNetwork(
            context_dim=paligemma_config.width,
            action_dim=config.action_dim,
            width=config.value_width,
            num_layers=config.value_num_layers,
            num_heads=config.value_num_heads,
            rngs=rngs,
        )

        import os as _os_bid
        if _os_bid.environ.get("_LOADING_BID_WEAK", "0") != "1":
            self.target_chunk_critic = TwinChunkCritic(
                context_dim=paligemma_config.width,
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                width=config.critic_width,
                num_layers=config.critic_num_layers,
                num_heads=config.critic_num_heads,
                rngs=rngs,
            )
        import os as _os_td
        self.use_td = _os_td.environ.get("USE_TD", "1") == "1"
        self.action_horizon_for_td = int(config.action_horizon)
        self.critic_gamma = float(config.critic_gamma)
        print("[iql] use_td=%s H=%d gamma=%s" % (
            self.use_td, self.action_horizon_for_td,
            self.critic_gamma), flush=True)

    def _compute_flow_loss_and_context(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool,
    ):
        (
            preprocess_rng,
            noise_rng,
            time_rng,
        ) = jax.random.split(
            rng,
            3,
        )

        observation = (
            _model.preprocess_observation(
                preprocess_rng,
                observation,
                train=train,
            )
        )

        batch_shape = actions.shape[:-2]

        noise = jax.random.normal(
            noise_rng,
            actions.shape,
        )

        time = (
            jax.random.beta(
                time_rng,
                1.5,
                1.0,
                batch_shape,
            )
            * 0.999
            + 0.001
        )

        time_expanded = time[
            ...,
            None,
            None,
        ]

        # This follows the convention already used in the local π0 code:
        # t=1 is noise and t=0 is expert action.
        x_t = (
            time_expanded * noise
            + (1.0 - time_expanded) * actions
        )

        target_velocity = (
            noise - actions
        )

        (
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
        ) = self.embed_prefix(
            observation
        )

        (
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
        ) = self.embed_suffix(
            observation,
            x_t,
            time,
        )

        input_mask = jnp.concatenate(
            [
                prefix_mask,
                suffix_mask,
            ],
            axis=1,
        )

        ar_mask = jnp.concatenate(
            [
                prefix_ar_mask,
                suffix_ar_mask,
            ],
            axis=0,
        )

        attention_mask = (
            _pi0.make_attn_mask(
                input_mask,
                ar_mask,
            )
        )

        positions = (
            jnp.cumsum(
                input_mask,
                axis=1,
            )
            - 1
        )

        (
            prefix_output,
            suffix_output,
        ), _ = self.PaliGemma.llm(
            [
                prefix_tokens,
                suffix_tokens,
            ],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        predicted_velocity = (
            self.action_out_proj(
                suffix_output[
                    :,
                    -self.action_horizon:,
                ]
            )
        )

        flow_loss_per_step = jnp.mean(
            jnp.square(
                predicted_velocity
                - target_velocity
            ),
            axis=-1,
        )

        return (
            flow_loss_per_step,
            prefix_output,
            prefix_mask,
            observation,
        )

    def compute_loss_iql(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        critic_target: jax.Array,
        *,
        next_observation=None,
        reward_cum=None,
        done_chunk=None,
        train: bool = False,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        (
            flow_loss_per_step,
            prefix_output,
            prefix_mask,
            processed_observation,
        ) = self._compute_flow_loss_and_context(
            rng,
            observation,
            actions,
            train=train,
        )

        # One scalar FM loss per dataset sample.
        flow_loss_per_sample = jnp.mean(
            flow_loss_per_step,
            axis=-1,
        )
        unweighted_actor_loss = jnp.mean(
            flow_loss_per_sample
        )

        # Q/V see the VLM representation, but their losses do not alter
        # the shared VLM/action actor through this context path.
        critic_context = jax.lax.stop_gradient(
            prefix_output
        )

        q1, q2 = self.chunk_critic(
            critic_context,
            prefix_mask,
            processed_observation.state,
            actions,
        )

        value = self.value_network(
            critic_context,
            prefix_mask,
            processed_observation.state,
        )

        critic_target = jnp.asarray(
            critic_target,
            dtype=jnp.float32,
        ).reshape((-1,))
        critic_target = jnp.clip(
            critic_target,
            0.0,
            1.0,
        )

        if (self.use_td and next_observation is not None
                and reward_cum is not None and done_chunk is not None):
            (_, _npo, _npm, _nproc) = self._compute_flow_loss_and_context(
                rng, next_observation, actions, train=False)
            v_next = self.value_network(
                jax.lax.stop_gradient(_npo), _npm, _nproc.state)
            _rc = jnp.asarray(reward_cum, dtype=jnp.float32).reshape((-1,))
            _dn = jnp.asarray(done_chunk, dtype=jnp.float32).reshape((-1,))
            _gh = jnp.float32(self.critic_gamma ** self.action_horizon_for_td)
            critic_target = jnp.clip(jax.lax.stop_gradient(
                _rc + _gh * (1.0 - _dn) * v_next), 0.0, 1.0)

        q1_loss = jnp.mean(
            jnp.square(q1 - critic_target)
        )
        q2_loss = jnp.mean(
            jnp.square(q2 - critic_target)
        )
        q_loss = 0.5 * (
            q1_loss + q2_loss
        )

        # IQL-style expectile value regression.
        # V loss is not allowed to update Q1 or Q2.
        q1_t, q2_t = self.target_chunk_critic(
            critic_context,
            prefix_mask,
            processed_observation.state,
            actions,
        )
        q_min = jnp.minimum(q1_t, q2_t)
        value_target = jax.lax.stop_gradient(
            q_min
        )
        value_error = value_target - value

        expectile_weight = jnp.where(
            value_error >= 0.0,
            self.expectile,
            1.0 - self.expectile,
        )
        value_loss = jnp.mean(
            expectile_weight
            * jnp.square(value_error)
        )

        # AWR weighting. Actor loss cannot update Q or V.
        advantage = jax.lax.stop_gradient(
            q_min - value
        )
        if self.adv_standardize:
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-6)
        log_weight = jnp.clip(
            self.advantage_beta * advantage,
            -20.0,
            20.0,
        )
        advantage_weight = jnp.exp(log_weight)
        advantage_weight = jnp.minimum(
            advantage_weight,
            self.max_advantage_weight,
        )

        if self.normalize_advantage_weights:
            advantage_weight = (
                advantage_weight
                / (
                    jnp.mean(advantage_weight)
                    + 1e-6
                )
            )

        advantage_weight = jax.lax.stop_gradient(
            advantage_weight
        )

        actor_loss = jnp.mean(
            advantage_weight
            * flow_loss_per_sample
        )

        total_loss = (
            self.actor_weight * actor_loss
            + self.q_loss_weight * q_loss
            + self.value_loss_weight * value_loss
        )

        weight_sum = jnp.sum(advantage_weight)
        weight_ess_ratio = (
            jnp.square(weight_sum)
            / (
                advantage_weight.shape[0]
                * jnp.sum(
                    jnp.square(advantage_weight)
                )
                + 1e-6
            )
        )

        _a = advantage.reshape(-1)
        _t = critic_target.reshape(-1)
        _ac = _a - jnp.mean(_a)
        _tc = _t - jnp.mean(_t)
        adv_corr = jnp.sum(_ac * _tc) / (
            jnp.sqrt(jnp.sum(_ac ** 2) * jnp.sum(_tc ** 2)) + 1e-8
        )
        _succ = (critic_target > 0).astype(q_min.dtype).reshape(-1)
        _qf = q_min.reshape(-1)
        q_succ = jnp.sum(_qf * _succ) / (jnp.sum(_succ) + 1e-6)
        q_fail = jnp.sum(_qf * (1.0 - _succ)) / (jnp.sum(1.0 - _succ) + 1e-6)
        v_std = jnp.std(value)
        metrics = {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "value_loss": value_loss,
            "flow_loss": jnp.mean(flow_loss_per_sample),
            "adv_mean": jnp.mean(advantage),
            "adv_std": jnp.std(advantage),
            "adv_min": jnp.min(advantage),
            "adv_max": jnp.max(advantage),
            "adv_weight_mean": jnp.mean(advantage_weight),
            "adv_weight_std": jnp.std(advantage_weight),
            "adv_weight_max": jnp.max(advantage_weight),
            "adv_weight_min": jnp.min(advantage_weight),
            "ess_ratio": weight_ess_ratio,
            "q_mean": jnp.mean(q_min),
            "q_std": jnp.std(q_min),
            "v_mean": jnp.mean(value),
            "v_std": v_std,
            "qmv_std_ratio": jnp.std(q_min - value) / (v_std + 1e-6),
            "target_mean": jnp.mean(critic_target),
            "target_std": jnp.std(critic_target),
            "adv_corr": adv_corr,
            "q_succ": q_succ,
            "q_fail": q_fail,
            "q_gap": q_succ - q_fail,
        }

        return total_loss, metrics
