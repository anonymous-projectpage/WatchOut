#!/bin/bash
# usage: CHUNK_MODE=cs_q CS_DELTA=0.25 ./run_libero.sh <TASK_ID> <PORT> <GPU> <TAG>
set -e
TASK=${1:?task id}; PORT=${2:-8170}; GPU=${3:-0}; TAG=${4:-run}
ACTOR=${PERTURB_ACTOR:?set PERTURB_ACTOR}
A="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"
B="25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"
cd ${XVLA_ROOT:?set XVLA_ROOT}/evaluation/libero
i=0
for EPS in "$A" "$B"; do
  env -u _PERT_TP -u _PERT_DONE -u _PERT_STEP \
    CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
    TASK_IDS=$TASK EPISODE_IDS=$EPS EVAL_HORIZON=${EVAL_HORIZON:-600} \
    CHUNK_MODE=${CHUNK_MODE:-cs_q} CS_DELTA=${CS_DELTA:-0.25} \
    CS_NCAND=${CS_NCAND:-16} CS_VERBOSE=${CS_VERBOSE:-1} \
    PERTURB_RANDOM=1 PERTURB_MODE=${PERTURB_MODE:-teleport} \
    PERTURB_DIST=${PERTURB_DIST:-0.10} PERTURB_DUR=${PERTURB_DUR:-30} \
    PERTURB_ACTOR=$ACTOR \
    nohup python libero_client.py --task_suites libero_10 \
      --server_ip 0.0.0.0 --server_port $((PORT+i)) --eval_time 50 \
      > ${RESULTS_DIR:-./results}/${TAG}_t${TASK}_$((PORT+i)).log 2>&1 &
  i=$((i+1)); sleep 5
done
echo "[$TAG] launched task=$TASK ports=$PORT,$((PORT+1))"
