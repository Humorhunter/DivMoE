#!/usr/bin/env bash
#
# Stage 1: domain-specialist dense CPT (one run per domain).
#
# Paper-faithful hyperparameters (DivMoE, Stage 1):
#   optimizer AdamW(betas=0.9,0.95), weight_decay 0.1, bf16 (all hardcoded in script)
#   lr 5e-5, min_lr 5e-6 (cosine), warmup 50 steps
#   seq_len 4096, budget 10B tokens per domain, global batch ~2M tokens/step
#
# Global batch is fixed at GLOBAL_BATCH_TOKENS (~2^21 = 2097152); this script
# derives gradient_accumulation_steps = GLOBAL_BATCH_TOKENS / (PER_DEVICE_BATCH * NUM_GPUS * BLOCK_SIZE).
#
# Output format (HuggingFace, loadable by create_divmoe.py via AutoModelForCausalLM):
#   $OUT_ROOT/stage1_<domain>/model.safetensors, config.json, tokenizer files, training_args.json
# plus DeepSpeed checkpoints:
#   $OUT_ROOT/stage1_<domain>/checkpoint-<step>/   (for --resume_from_checkpoint)
#   $OUT_ROOT/stage1_<domain>/model-<step>/        (HF snapshots)
#
# Usage:
#   bash scripts/stage1.sh
#   NUM_GPUS=8 PER_DEVICE_BATCH=16 bash scripts/stage1.sh
#   DOMAINS="math code" GRAD_CKPT=1 bash scripts/stage1.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- configurable knobs (override via env) ----
BASE="${BASE:-$HOME/models/Qwen3-0.6B-Base}"
DATA_ROOT="${DATA_ROOT:-/data/DivMoE/data}"
OUT_ROOT="${OUT_ROOT:-$HOME/DivMoE/output}"
DOMAINS="${DOMAINS:-math code academic web}"

NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-4096}"
GLOBAL_BATCH_TOKENS="${GLOBAL_BATCH_TOKENS:-2097152}"   # ~2M tokens/step (paper)
TRAIN_TOKENS="${TRAIN_TOKENS:-10000000000}"             # 10B tokens per domain (paper)

LR="${LR:-5e-5}"
MIN_LR="${MIN_LR:-5e-6}"
WARMUP="${WARMUP:-50}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SAVE_EVERY="${SAVE_EVERY:-500}"
ZERO_STAGE="${ZERO_STAGE:-2}"
GRAD_CKPT="${GRAD_CKPT:-0}"                             # 1 to enable gradient checkpointing
TRACKING="${TRACKING:-0}"                               # 1 to enable tensorboard tracking
SWANLAB="${SWANLAB:-1}"                                 # 1 to log to SwanLab (implies tracking)
SWANLAB_PROJECT="${SWANLAB_PROJECT:-DivMoE}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_MODE="${SWANLAB_MODE:-}"                        # cloud | local | offline | disabled

# ---- derive grad accumulation and step count ----
TOKENS_PER_ITER=$(( PER_DEVICE_BATCH * NUM_GPUS * BLOCK_SIZE ))
if (( GLOBAL_BATCH_TOKENS % TOKENS_PER_ITER != 0 )); then
  echo "ERROR: GLOBAL_BATCH_TOKENS=$GLOBAL_BATCH_TOKENS not divisible by" >&2
  echo "       PER_DEVICE_BATCH*NUM_GPUS*BLOCK_SIZE = $TOKENS_PER_ITER" >&2
  echo "       Adjust PER_DEVICE_BATCH or NUM_GPUS." >&2
  exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BATCH_TOKENS / TOKENS_PER_ITER ))
MAX_STEPS=$(( (TRAIN_TOKENS + GLOBAL_BATCH_TOKENS - 1) / GLOBAL_BATCH_TOKENS ))

EXTRA_FLAGS=""
[[ "$GRAD_CKPT" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --gradient_checkpointing"
[[ "$TRACKING" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --with_tracking"
if [[ "$SWANLAB" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --use_swanlab --swanlab_project $SWANLAB_PROJECT"
  [[ -n "$SWANLAB_WORKSPACE" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --swanlab_workspace $SWANLAB_WORKSPACE"
  [[ -n "$SWANLAB_MODE" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --swanlab_mode $SWANLAB_MODE"
fi

echo "=============================================================="
echo " Stage 1 dense specialist CPT"
echo "   base            : $BASE"
echo "   data root       : $DATA_ROOT/stage1/<domain>"
echo "   output root     : $OUT_ROOT/stage1_<domain>"
echo "   domains         : $DOMAINS"
echo "   num_gpus        : $NUM_GPUS"
echo "   per_device_batch: $PER_DEVICE_BATCH"
echo "   grad_accum      : $GRAD_ACCUM"
echo "   block_size      : $BLOCK_SIZE"
echo "   global batch    : $GLOBAL_BATCH_TOKENS tokens/step ($(( GLOBAL_BATCH_TOKENS / BLOCK_SIZE )) sequences)"
echo "   max_steps       : $MAX_STEPS  (~$(( TRAIN_TOKENS / 1000000000 ))B tokens/domain)"
echo "   lr/min_lr/warmup: $LR / $MIN_LR / $WARMUP"
echo "   weight_decay    : $WEIGHT_DECAY"
echo "   zero_stage      : $ZERO_STAGE"
echo "   swanlab         : $SWANLAB (project=$SWANLAB_PROJECT)"
echo "=============================================================="

for domain in $DOMAINS; do
  echo ""
  echo ">>> [Stage 1] domain=$domain"

  deepspeed --num_gpus "$NUM_GPUS" run_clm_no_trainer.py \
    --model_name_or_path "$BASE" \
    --olmoe_data_dir "$DATA_ROOT/stage1/$domain" \
    --output_dir "$OUT_ROOT/stage1_$domain" \
    --block_size "$BLOCK_SIZE" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --min_lr "$MIN_LR" \
    --lr_scheduler_type cosine_with_min_lr \
    --num_warmup_steps "$WARMUP" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_train_steps "$MAX_STEPS" \
    --checkpointing_steps "$SAVE_EVERY" \
    --zero_stage "$ZERO_STAGE" \
    $EXTRA_FLAGS

  echo ">>> [Stage 1] domain=$domain done -> $OUT_ROOT/stage1_$domain"
done

echo ""
echo "All Stage 1 specialists finished."
