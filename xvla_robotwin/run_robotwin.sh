#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate xvla2
cd ${XVLA_ROOT:?set XVLA_ROOT}/evaluation/robotwin-2.0
TASK=$1; ACTOR=$2; TAG=$3; NPAR=${4:-4}; DELTA=${5:-0.20}
SEEDS=$(seq ${S0:-1000} ${S1:-1059}); n=0
OUT=${RESULTS_DIR:-./results}/$TAG
mkdir -p $OUT
echo "[$TAG] start $(date +%H:%M) task=$TASK actor=$ACTOR npar=$NPAR delta=$DELTA"
for S in $SEEDS; do
  if [ "$TASK" = "beat_block_hammer" ]; then
    PERT="PERTURB_MODE=teleport PERTURB_DUR=30 PERTURB_DIST_RANGE=0.04,0.06 PERTURB_ANGLE_RANGE=150,210"
  else
    PERT="PERTURB_MODE=velocity PERTURB_MIRROR=1 PERTURB_VEL=0.5 PERTURB_DUR=100"
  fi
  ( env CHUNK_MODE=${CMODE:-cs_q} CS_NCAND=16 CS_DELTA=$DELTA CS_VERBOSE=1 CS_STEP_LIM=300 AAC_VERBOSE=1 AAC_ALPHA=${AAC_ALPHA:-3.0} AAC_ALPHA_REL=$AAC_ALPHA_REL AAC_CONT_DIMS=$AAC_CONT_DIMS AAC_GRIP_DIMS=$AAC_GRIP_DIMS \
      PERTURB_RANDOM=1 $PERT PERTURB_ACTOR=$ACTOR \
      PERTURB_STEP_RANGE=15,30 PERTURB_SEED=$S \
      CUDA_VISIBLE_DEVICES=${CVD:-2} python client.py \
      --host 0.0.0.0 --port ${PORT:-8010} --device 0 \
      --task_name $TASK --task_config demo_randomized \
      --num_episodes 1 --seed $S \
      --output_path $OUT/s$S --eval_log_dir $OUT/logs \
      > /tmp/${TAG}_s$S.log 2>&1 ) &
  n=$((n+1))
  if [ $((n % NPAR)) -eq 0 ]; then
    wait; K=0; N=0
    for T in $SEEDS; do
      v=$(grep -a "Success rate" /tmp/${TAG}_s$T.log 2>/dev/null | tail -1 | grep -oP '[\d.]+(?=%)')
      [ -n "$v" ] && { [ "${v%.*}" -ge 50 ] && K=$((K+1)); N=$((N+1)); }
    done
    echo "[$TAG] $(date +%H:%M)  $K/$N  (${n}/60 launched)"
  fi
done
wait
echo "[$TAG] DONE $(date +%H:%M)"
