#!/bin/bash
set -euo pipefail
#
# End-to-end segmentation training pipeline.
# Runs overnight: collect GT data → generate pseudo-labels → train model.
#
# Usage:
#   # Full pipeline (all 9 tracks × 4 speeds × 3 lighting = 108 runs, ~3 hours):
#   ./tools/segmentation/train_pipeline.sh
#
#   # Quick pipeline (2 tracks × 2 speeds × 1 lighting = 4 runs, ~10 min):
#   ./tools/segmentation/train_pipeline.sh --quick
#
#   # Skip collection (already have recordings, just prepare + train):
#   ./tools/segmentation/train_pipeline.sh --skip-collect
#
#   # Custom tracks/speeds:
#   ./tools/segmentation/train_pipeline.sh --tracks "s_loop hill_highway highway_65" --speeds "8 12"
#
# Output:
#   Recordings:  data/recordings/seg_training/
#   Dataset:     data/segmentation_dataset/
#   Checkpoint:  data/segmentation_dataset/checkpoints/segnet_best.pt
#   Log:         data/segmentation_dataset/training_pipeline.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
RECORDINGS_DIR="data/recordings/seg_training"
DATASET_DIR="data/segmentation_dataset"
EPOCHS=20
SKIP_COLLECT=false
QUICK=false
CUSTOM_TRACKS=""
CUSTOM_SPEEDS=""

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-collect) SKIP_COLLECT=true; shift ;;
        --quick) QUICK=true; shift ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --tracks) CUSTOM_TRACKS="$2"; shift 2 ;;
        --speeds) CUSTOM_SPEEDS="$2"; shift 2 ;;
        --recordings-dir) RECORDINGS_DIR="$2"; shift 2 ;;
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOG_FILE="${DATASET_DIR}/training_pipeline.log"
mkdir -p "$DATASET_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== Segmentation Training Pipeline ==="
log "Started at $(date)"
log ""

# ── Step 1: Collect GT recordings ─────────────────────────────────────────────
if $SKIP_COLLECT; then
    log "STEP 1: SKIPPED (--skip-collect)"
    existing=$(find "$RECORDINGS_DIR" -name "*.h5" 2>/dev/null | wc -l | tr -d ' ')
    log "  Using $existing existing recordings in $RECORDINGS_DIR"
else
    log "STEP 1: Collecting ground truth recordings..."

    COLLECT_ARGS=()
    if $QUICK; then
        COLLECT_ARGS+=(--tracks s_loop hill_highway --speeds 8 12)
        log "  Quick mode: 2 tracks × 2 speeds × 3 lighting = 12 runs"
    fi
    if [[ -n "$CUSTOM_TRACKS" ]]; then
        COLLECT_ARGS+=(--tracks $CUSTOM_TRACKS)
    fi
    if [[ -n "$CUSTOM_SPEEDS" ]]; then
        COLLECT_ARGS+=(--speeds $CUSTOM_SPEEDS)
    fi

    if ! ./tools/segmentation/collect_training_data.sh ${COLLECT_ARGS[@]+"${COLLECT_ARGS[@]}"} 2>&1 | tee -a "$LOG_FILE"; then
        log "ERROR: Collection failed. Check log above."
        exit 1
    fi

    collected=$(find "$RECORDINGS_DIR" -name "*.h5" 2>/dev/null | wc -l | tr -d ' ')
    log "  Collection complete: $collected recordings"
fi
log ""

# ── Step 2: Generate training dataset ─────────────────────────────────────────
log "STEP 2: Generating training dataset from pseudo-labels..."

if ! "$PYTHON_BIN" training/scripts/prepare_segmentation_dataset.py \
    --recordings "$RECORDINGS_DIR" \
    --output-dir "$DATASET_DIR" \
    --stride 3 \
    --no-overlays \
    2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Dataset preparation failed."
    exit 1
fi

# Count generated images
n_images=$(find "$DATASET_DIR/images" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
n_masks=$(find "$DATASET_DIR/masks" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
log "  Dataset ready: $n_images images, $n_masks masks"

if [[ "$n_images" -lt 100 ]]; then
    log "WARNING: Only $n_images images — model may underfit. Consider collecting more data."
fi
log ""

# ── Step 3: Train model ──────────────────────────────────────────────────────
log "STEP 3: Training SimpleSegNet ($EPOCHS epochs)..."

if ! "$PYTHON_BIN" training/scripts/train_segmentation.py \
    --data-dir "$DATASET_DIR" \
    --epochs "$EPOCHS" \
    --batch-size 8 \
    --learning-rate 1e-3 \
    2>&1 | tee -a "$LOG_FILE"; then
    log "ERROR: Training failed."
    exit 1
fi

CHECKPOINT="$DATASET_DIR/checkpoints/segnet_best.pt"
if [[ -f "$CHECKPOINT" ]]; then
    size=$(du -h "$CHECKPOINT" | cut -f1)
    log "  Checkpoint saved: $CHECKPOINT ($size)"
else
    log "ERROR: No checkpoint found after training!"
    exit 1
fi
log ""

# ── Summary ───────────────────────────────────────────────────────────────────
log "=== Pipeline Complete ==="
log "  Recordings: $RECORDINGS_DIR"
log "  Dataset:    $DATASET_DIR ($n_images images)"
log "  Checkpoint: $CHECKPOINT"
log "  Epochs:     $EPOCHS"
log "  Finished:   $(date)"
log ""
log "Next steps:"
log "  1. Validate on a recording:"
log "     python tools/segmentation/preview_segmentation_masks.py \\"
log "         --recording data/recordings/<recent>.h5 --frames 50-100 --segmentation --save_overlay"
log ""
log "  2. Run E2E test:"
log "     ./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \\"
log "         --track-yaml tracks/s_loop.yml --duration 60"
log ""
log "  3. Analyze:"
log "     python tools/analyze/analyze_drive_overall.py --latest"
