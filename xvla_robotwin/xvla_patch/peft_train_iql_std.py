"""
X-VLA LoRA + AWR (IQL) training.

Same structure as peft_train.py, but the actor loss becomes an advantage-weighted
flow-matching loss and the Q1/Q2/V critics are trained jointly (same formulation as pi05_iql.compute_loss_iql).

model.forward() only returns a scalar loss, so it is not used here; instead
forward_vlm -> noise injection -> transformer -> per-sample loss is carried out inline.
(an inlined copy of the body of modeling_xvla.XVLA.forward)

The critic is a new module rather than a LoRA adapter, so it is trained fully, with its own
parameter group and learning rate (--critic_lr).

Example:
  EXPECTILE=0.7 CUDA_VISIBLE_DEVICES=4 accelerate launch --mixed_precision bf16 \
    --num_processes 1 peft_train_iql.py \
    --models 2toINF/X-VLA-RoboTwin2 \
    --train_metas_path ${XVLA_ROOT}/metas_mt4 \
    --output_dir runs/mt4_iql_e07 \
    --batch_size 16 --learning_rate 1e-4 --critic_lr 3e-4 \
    --iters 30000 --warmup_steps 500 --save_interval 3000
"""
from __future__ import annotations

import os
import sys
import time
import math
import logging
import argparse
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
from xvla_iql_std import XVLAIQL, flow_loss_per_sample


def get_logger(name="train", accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    if accelerator is None or accelerator.is_main_process:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)
    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runnings")
    p.add_argument("--train_metas_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_actions", type=int, default=30)
    p.add_argument("--num_views", type=int, default=3)
    p.add_argument("--action_mode", type=str, default="ee6d")

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--learning_coef", type=float, default=0.1,
                   help="LR multiplier for soft prompts")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--freeze_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=500)

    p.add_argument("--save_interval", type=int, default=3000)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def group_lr(step, args, base, warmup, frozen_until=0):
    """Constant after warmup; zero before frozen_until."""
    if step < frozen_until:
        return 0.0
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    return base


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers("XVLA-Training")
    logger = get_logger(__name__, accelerator)
    logger.info(f"Args: {args}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = accelerator.device

    # ---------------------------------------------------------------- model
    processor = XVLAProcessor.from_pretrained(args.models)
    model = XVLA.from_pretrained(args.models, trust_remote_code=True,
                                 torch_dtype=torch.float32)

    lora_config = LoraConfig(
        lora_alpha=16, r=8, bias="none", target_modules="all-linear",
        modules_to_save=["transformer.soft_prompt_hub",
                         "transformer.action_encoder",
                         "transformer.action_decoder"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    base = model.base_model.model if hasattr(model, "base_model") else model
    ctx_dim = base.vlm.language_model.model.encoder.config.d_model
    probe = next(iter(create_dataloader(
        batch_size=2, metas_path=args.train_metas_path,
        num_actions=args.num_actions, training=False,
        action_mode=args.action_mode)))
    state_dim = probe["proprio"].shape[-1]
    action_dim = probe["action"].shape[-1]
    logger.info(f"ctx_dim={ctx_dim} state_dim={state_dim} action_dim={action_dim}")

    iql = XVLAIQL(context_dim=ctx_dim, state_dim=state_dim,
                  action_dim=action_dim, action_horizon=args.num_actions)

    # ------------------------------------------------------------ optimizer
    critic_params = list(iql.parameters())
    sp, ah, core, vlm = [], [], [], []
    for n, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if "soft_prompt" in n:
            sp.append(prm)
        elif "action_encoder" in n or "action_decoder" in n:
            ah.append(prm)
        elif "vlm" in n:
            vlm.append(prm)
        else:
            core.append(prm)

    groups = [
        {"params": core, "lr": args.learning_rate, "name": "transformer_core"},
        {"params": vlm, "lr": args.learning_rate, "name": "vlm"},
        {"params": ah, "lr": args.learning_rate, "name": "action_heads"},
        {"params": sp, "lr": args.learning_rate * args.learning_coef, "name": "soft_prompts"},
        {"params": critic_params, "lr": args.critic_lr, "name": "critic"},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    optim = AdamW(groups, betas=tuple(args.betas), weight_decay=args.weight_decay)
    base_lrs = {g["name"]: g["lr"] for g in groups}

    train_dataloader = create_dataloader(
        batch_size=args.batch_size, metas_path=args.train_metas_path,
        num_actions=args.num_actions, training=True, action_mode=args.action_mode)

    model, iql, optim = accelerator.prepare(model, iql, optim)

    net = model.base_model.model if hasattr(model, "base_model") else model
    iql_mod = iql.module if hasattr(iql, "module") else iql

    logger.info(f"🚀 Start AWR training for {args.iters} iterations")
    pbar = tqdm(total=args.iters, disable=not accelerator.is_main_process,
                dynamic_ncols=True, desc="train")

    global_step, t0 = 0, time.time()
    for batch in train_dataloader:
        if global_step >= args.iters:
            break

        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}

        for g in optim.param_groups:
            fz = 0 if g["name"] in ("soft_prompts", "critic", "action_heads") \
                 else args.freeze_steps
            g["lr"] = group_lr(global_step, args, base_lrs[g["name"]],
                               args.warmup_steps, fz)

        input_ids = inputs["input_ids"]
        action = inputs["action"]
        proprio = inputs["proprio"]
        critic_target = inputs["critic_target"]

        # ---- inlined body of modeling_xvla.XVLA.forward ------------------
        enc = net.forward_vlm(input_ids, inputs["image_input"], inputs["image_mask"])
        B = input_ids.shape[0]
        t = (torch.rand(1, device=device) + torch.arange(B, device=device) / B) % (1 - 1e-5)
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) \
                       + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = net.action_space.preprocess(proprio, action_noisy)
        pred_action = net.transformer(
            domain_id=inputs["domain_id"], action_with_noise=action_noisy_m,
            t=t, proprio=proprio_m, **enc)
        # ---------------------------------------------------------------

        flow_ps, flow_logs = flow_loss_per_sample(net.action_space, pred_action, action)

        # ---- chunk-wise TD: context/proprio at s_{t+H} ----
        _nc = _np_ = _rc = _dn = None
        if getattr(iql_mod, "use_td", False) and "next_image_input" in inputs:
            with torch.no_grad():
                _nenc = net.forward_vlm(
                    input_ids, inputs["next_image_input"], inputs["image_mask"])
                _nc = _nenc["vlm_features"]
            _np_ = inputs["next_abs_trajectory"][:, 0, :]
            _rc = inputs["critic_reward_cum"]
            _dn = inputs["critic_done_chunk"]

        loss, iql_logs = iql_mod.losses(
            context=enc["vlm_features"], mask=None,
            state=proprio, actions=action,
            flow_per_sample=flow_ps, critic_target=critic_target,
            next_context=_nc, next_state=_np_,
            reward_cum=_rc, done_chunk=_dn)

        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(
                list(model.parameters()) + list(iql.parameters()), args.max_grad_norm)
        optim.step()
        optim.zero_grad()
        if getattr(iql_mod, "use_td", False):
            iql_mod.update_target()

        if global_step % args.log_interval == 0:
            logs = {k: float(v) for k, v in {**flow_logs, **iql_logs}.items()}
            logs["loss_total"] = float(loss.detach())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                dt = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] loss={logs['loss_total']:.4f} "
                    f"actor={logs['actor_loss']:.4f} q={logs['q_loss']:.4f} "
                    f"v={logs['value_loss']:.4f} adv={logs['adv_mean']:+.3f} "
                    f"({dt:.2f}s/it)")

        if accelerator.is_main_process:
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}")
            pbar.update(1)

        global_step += 1
        if global_step % args.save_interval == 0 or global_step == args.iters:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                d = os.path.join(args.output_dir, f"ckpt-{global_step}")
                os.makedirs(d, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(d)
                torch.save(accelerator.unwrap_model(iql).state_dict(),
                           os.path.join(d, "iql_critic.pt"))
                logger.info(f"💾 saved {d}")

    accelerator.end_training()
    logger.info("done")


if __name__ == "__main__":
    main()
