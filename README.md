# WatchOut

## Setup

Clone the upstream repo, copy the patch files over it, and set the env vars.

```bash
# X-VLA
git clone https://github.com/2toinf/X-VLA.git && cd X-VLA && git checkout 6bc2513
cp WatchOut/xvla_libero/xvla_patch/*.py .
mv evaluation_libero_libero_client.py evaluation/libero/libero_client.py
mv libero_perturb.py evaluation/libero/
export XVLA_ROOT=$PWD WATCHOUT_COMMON=WatchOut/xvla_libero/common RESULTS_DIR=$PWD/results

# RoboTwin
git clone https://github.com/RoboTwin-Platform/RoboTwin.git && cd RoboTwin && git checkout 13c3c47
export ROBOTWIN_ROOT=$PWD

# pi0.5 (openpi lives in RoboTwin/policy/pi05)
cd $ROBOTWIN_ROOT/policy/pi05 && W=WatchOut/pi05_libero
cp $W/openpi_patch/src_openpi_training/*.py src/openpi/training/
cp $W/openpi_patch/src_openpi_models/*.py   src/openpi/models/
cp $W/openpi_patch/scripts/*.py             scripts/
cp $W/main_adaptive.py $W/libero_perturb.py examples/libero/      # LIBERO
cp $W/deploy_policy.py $W/pi_model.py       ../pi05_adaptive/     # RoboTwin
export OPENPI_ROOT=$PWD WATCHOUT_COMMON=$W/common
```

Append `configs_to_add.txt` to the `_CONFIGS` list in
`src/openpi/training/config.py` and replace `YOUR_HF_USER` with your account.

## Train

```bash
python scripts/compute_norm_stats.py --config-name pi05_libero10_iql_std

EXPECTILE=0.7 ADV_BETA=3.0 USE_TD=1 CRITIC_GAMMA=0.99 TARGET_TAU=0.005 \
python scripts/train_iql_std.py pi05_libero10_iql_std --exp-name=RUN --overwrite

EXPECTILE=0.7 ADV_BETA=3.0 USE_TD=1 CRITIC_GAMMA=0.99 TARGET_TAU=0.005 \
python peft_train_iql_std.py --models 2toINF/X-VLA-Libero \
  --train_metas_path $XVLA_ROOT/metas --output_dir $XVLA_ROOT/runs/RUN \
  --batch_size 16 --num_actions 30 --num_views 2 \
  --learning_rate 1e-4 --iters 30000 --save_interval 3000 --seed 42
```

## Serve

Candidate sampling must be on, otherwise the client gets a 404.

```bash
# pi0.5
SERVE_CANDS=1 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false PYTHONPATH=$PWD/src \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero10_iql_std --policy.dir=$PWD/checkpoints/.../30000

# X-VLA
CK=$XVLA_ROOT/runs/RUN/ckpt-30000
IQL_CRITIC_PATH=$CK/iql_critic.pt python -m deploy \
  --model_path 2toINF/X-VLA-Libero --LoRA_path $CK --output_dir logs/x_8000 --port 8000
```

## Evaluate

```bash
# X-VLA + LIBERO        <task_id> <port> <gpu> <tag>
CHUNK_MODE=cs_q CS_NCAND=16 PERTURB_ACTOR=alphabet_soup_1 \
./run_libero.sh 0 8000 0 mytag

# X-VLA + RoboTwin      <task> <actor> <tag> <n_parallel>
S0=1000 S1=1059 CMODE=cs_q CVD=0 PORT=8000 \
./run_robotwin.sh place_empty_cup cup mytag 6

# pi0.5 + LIBERO        <task_id> <port> <gpu> <seed> <tag>
CHUNK_MODE=cs_q CS_NCAND=16 PERTURB_ACTOR=porcelain_mug_1 \
./run_libero.sh 4 8000 0 1000 mytag

# pi0.5 + RoboTwin      <gpu> <task> <actor> <tag> <seed0> <mode> <n_parallel>
./run_robotwin.sh 0 stack_bowls_two bowl1 mytag 1000 pr 6
```
