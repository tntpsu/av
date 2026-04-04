#!/bin/bash
set -euo pipefail
#
# Collect ground-truth recordings across all tracks and speed conditions
# for segmentation model training.
#
# Usage:
#   ./tools/segmentation/collect_training_data.sh                    # all tracks, default speeds
#   ./tools/segmentation/collect_training_data.sh --tracks hill_highway s_loop
#   ./tools/segmentation/collect_training_data.sh --speeds 6 10 14
#   ./tools/segmentation/collect_training_data.sh --duration 90 --dry-run
#
# After collection, run the dataset preparation:
#   python training/scripts/prepare_segmentation_dataset.py \
#       --recordings data/recordings/seg_training/ --output data/segmentation_dataset/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
TRACKS=(
    oval
    s_loop
    circle
    hairpin_15
    highway_65
    mixed_radius
    sweeping_highway
    hill_highway
    autobahn_30
)

SPEEDS=(6 8 10 12)        # m/s — covers slow curves to highway speed
DURATION=60               # seconds per run
OUTPUT_DIR="$PROJECT_ROOT/data/recordings/seg_training"
DRY_RUN=false
SKIP_EXISTING=true

# Lighting presets: "name:altitude:azimuth"
LIGHTING_PRESETS=(
    "noon:80:180"
    "afternoon:45:220"
    "dawn:15:90"
)

# ── Parse args ────────────────────────────────────────────────────────────────
parse_tracks=false
parse_speeds=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tracks)
            TRACKS=()
            parse_tracks=true
            parse_speeds=false
            shift
            ;;
        --speeds)
            SPEEDS=()
            parse_speeds=true
            parse_tracks=false
            shift
            ;;
        --duration)
            parse_tracks=false
            parse_speeds=false
            DURATION="$2"
            shift 2
            ;;
        --output-dir)
            parse_tracks=false
            parse_speeds=false
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            parse_tracks=false
            parse_speeds=false
            DRY_RUN=true
            shift
            ;;
        --no-skip)
            parse_tracks=false
            parse_speeds=false
            SKIP_EXISTING=false
            shift
            ;;
        *)
            if $parse_tracks; then
                TRACKS+=("$1")
            elif $parse_speeds; then
                SPEEDS+=("$1")
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_RUNS=$(( ${#TRACKS[@]} * ${#SPEEDS[@]} * ${#LIGHTING_PRESETS[@]} ))
TOTAL_TIME=$(( TOTAL_RUNS * (DURATION + 15) ))  # +15s for startup/shutdown

echo "=== Segmentation Training Data Collection ==="
echo "Tracks:    ${TRACKS[*]}"
echo "Speeds:    ${SPEEDS[*]} m/s"
echo "Lighting:  ${LIGHTING_PRESETS[*]}"
echo "Duration:  ${DURATION}s per run"
echo "Total:     ${TOTAL_RUNS} runs (~$((TOTAL_TIME / 60))min estimated)"
echo "Output:    ${OUTPUT_DIR}"
echo ""

mkdir -p "$OUTPUT_DIR"

if $DRY_RUN; then
    echo "[DRY RUN] Would execute:"
    for track in "${TRACKS[@]}"; do
        for speed in "${SPEEDS[@]}"; do
            for preset in "${LIGHTING_PRESETS[@]}"; do
                IFS=':' read -r light_name sun_alt sun_az <<< "$preset"
                echo "  GT run: track=${track} speed=${speed}m/s light=${light_name} duration=${DURATION}s"
            done
        done
    done
    echo ""
    echo "Total: ${TOTAL_RUNS} runs"
    exit 0
fi

# ── Collection loop ───────────────────────────────────────────────────────────
COMPLETED=0
FAILED=0
SKIPPED=0
LOG_FILE="${OUTPUT_DIR}/collection_log.txt"

echo "Collection started at $(date)" | tee "$LOG_FILE"

for track in "${TRACKS[@]}"; do
    track_yaml="$PROJECT_ROOT/tracks/${track}.yml"
    if [[ ! -f "$track_yaml" ]]; then
        echo "[WARN] Track YAML not found: ${track_yaml}, skipping"
        ((FAILED++))
        continue
    fi

    for speed in "${SPEEDS[@]}"; do
        # Compute safe duration: one lap minus 5% margin to stop before track-end seam
        safe_dur=$("$PYTHON_BIN" -c "
from tools.segmentation.track_length import one_lap_duration
from pathlib import Path
print(f'{one_lap_duration(Path(\"${track_yaml}\"), ${speed}):.0f}')
" 2>/dev/null || echo "$DURATION")
        # Use the shorter of user duration and safe one-lap duration
        if (( safe_dur < DURATION )); then
            run_dur=$safe_dur
        else
            run_dur=$DURATION
        fi

        for preset in "${LIGHTING_PRESETS[@]}"; do
            IFS=':' read -r light_name sun_alt sun_az <<< "$preset"
            run_name="gt_${track}_${speed}mps_${light_name}"

            # Skip if recording already exists
            if $SKIP_EXISTING; then
                existing=$(find "$OUTPUT_DIR" -name "${run_name}_*.h5" 2>/dev/null | head -1)
                if [[ -n "$existing" ]]; then
                    echo "[SKIP] ${run_name} — already exists: $(basename "$existing")"
                    ((SKIPPED++))
                    continue
                fi
            fi

            echo ""
            echo "── Run $((COMPLETED + FAILED + SKIPPED + 1))/${TOTAL_RUNS}: ${run_name} ──"
            echo "  Track: ${track}  Speed: ${speed} m/s  Light: ${light_name} (alt=${sun_alt}° az=${sun_az}°)  Duration: ${run_dur}s (1 lap)"

            # Run GT follower
            if "$PROJECT_ROOT/start_ground_truth.sh" \
                --track-yaml "$track_yaml" \
                --speed "$speed" \
                --duration "$run_dur" \
                --sun-altitude "$sun_alt" \
                --sun-azimuth "$sun_az" \
                --build-unity-player \
                --skip-unity-build-if-clean; then

                # Move the latest recording to output dir with descriptive name
                latest=$(ls -t "$PROJECT_ROOT/data/recordings/"recording_*.h5 2>/dev/null | head -1)
                if [[ -n "$latest" ]]; then
                    timestamp=$(date +%Y%m%d_%H%M%S)
                    dest="${OUTPUT_DIR}/${run_name}_${timestamp}.h5"
                    mv "$latest" "$dest"
                    echo "  -> Saved: $(basename "$dest")"
                    echo "OK  ${run_name} -> $(basename "$dest")" >> "$LOG_FILE"
                    ((COMPLETED++))
                else
                    echo "  [WARN] No recording found after run"
                    echo "FAIL ${run_name} — no recording" >> "$LOG_FILE"
                    ((FAILED++))
                fi
            else
                echo "  [FAIL] GT run failed for ${run_name}"
                echo "FAIL ${run_name} — run error" >> "$LOG_FILE"
                ((FAILED++))
            fi
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Collection Complete ==="
echo "  Completed: ${COMPLETED}"
echo "  Skipped:   ${SKIPPED}"
echo "  Failed:    ${FAILED}"
echo "  Recordings in: ${OUTPUT_DIR}"
echo ""

if [[ $COMPLETED -gt 0 ]]; then
    echo "Next steps:"
    echo "  1. Review sample frames:"
    echo "     python tools/segmentation/preview_segmentation_masks.py \\"
    echo "         --recording ${OUTPUT_DIR}/<recording>.h5 --frames 0-10 --save_overlay"
    echo ""
    echo "  2. Generate training dataset:"
    echo "     python training/scripts/prepare_segmentation_dataset.py \\"
    echo "         --recordings ${OUTPUT_DIR} --output data/segmentation_dataset/"
    echo ""
    echo "  3. Train model:"
    echo "     python training/scripts/train_segmentation.py"
fi

echo "Collection finished at $(date)" >> "$LOG_FILE"
