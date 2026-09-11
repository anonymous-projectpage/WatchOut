#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate robotwin
export RT=${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT}
cd $RT/policy/pi05_adaptive
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
G=$1; TASK=$2; ACTOR=$3; TAG=$4; S0=$5; MODE=$6; NPAR=${7:-6}
SEEDS=$(seq $S0 $((S0+59))); n=0
case $MODE in
  te)  OPT="CHUNK_MODE=te TE_M=${TE_M:-0.05}" ;;
  pr)  OPT="CHUNK_MODE=cs_q CS_GAP=1 CS_KCHECK=1 CS_DELTA=0.20 CS_NCAND=16 CS_PROFILE=1" ;;
  aac) OPT="CHUNK_MODE=aac AAC_NCAND=16 AAC_ALPHA=3.0" ;;
  h1)  OPT="CHUNK_MODE=fixed FIXED_H=1" ;;
  bid) OPT="CHUNK_MODE=bid BID_NCAND=16 BID_NWEAK=16 BID_WEAK_CONFIG=pi05_mt50_off2on BID_WEAK_MODEL=adaptive_base BID_WEAK_CKPT=30000" ;;
  *)   echo "unknown mode: $MODE"; exit 1 ;;
esac
echo "[$TAG] start $(date +%H:%M) task=$TASK gpu=$G mode=$MODE seed=$S0~$((S0+59))"
for S in $SEEDS; do
  if [ "$TASK" = "beat_block_hammer" ]; then
    PERT="PERTURB_MODE=teleport PERTURB_DUR=30 PERTURB_DIST_RANGE=0.04,0.06 PERTURB_ANGLE_RANGE=150,210"
  else
    PERT="PERTURB_MODE=velocity PERTURB_MIRROR=1 PERTURB_VEL=0.5 PERTURB_DUR=100"
  fi
  ( env $OPT PERTURB_RANDOM=1 $PERT PERTURB_ACTOR=$ACTOR \
      PERTURB_STEP_RANGE=15,30 PERTURB_SEED=$S CS_NORM_CONFIG=pi05_mt4_iql \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${MF:-0.05} \
      CKPT_OVERRIDE=30000 ROBOTWIN_TEST_NUM=1 \
      bash eval.sh $TASK demo_randomized pi05_mt4_iql mt4_td_e07_s42 $S $G \
      > /tmp/${TAG}_s$S.log 2>&1 ) &
  n=$((n+1))
  if [ $((n % NPAR)) -eq 0 ]; then
    wait; K=0; N=0
    for T in $SEEDS; do
      v=$(grep -a "Success rate" /tmp/${TAG}_s$T.log 2>/dev/null | tail -1 | grep -oP '\d+(?=/1)')
      [ -n "$v" ] && { K=$((K+v)); N=$((N+1)); }
    done
    echo "[$TAG] $(date +%H:%M)  $K/$N  (${n}/60 launched)"
  fi
done
wait
K=0; N=0; D=""
for T in $SEEDS; do
  v=$(grep -a "Success rate" /tmp/${TAG}_s$T.log 2>/dev/null | tail -1 | grep -oP '\d+(?=/1)')
  if [ -n "$v" ]; then K=$((K+v)); N=$((N+1)); else D="$D $T"; fi
done
echo "[$TAG] DONE $(date +%H:%M)  FINAL $K/$N   DEAD:$D"
