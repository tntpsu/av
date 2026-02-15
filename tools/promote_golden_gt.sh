#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RECORDINGS_DIR="$REPO_ROOT/data/recordings"

SOURCE=""
TRACK=""
DURATION=""
SYNC_POLICY="latest"
DATE_TAG="$(date +%Y%m%d)"
MODE="copy"
FORCE=false

print_usage() {
    echo "Usage: $0 --source FILE --track TRACK --duration TAG [options]"
    echo ""
    echo "Promote a GT recording to a canonical golden filename."
    echo ""
    echo "Required:"
    echo "  --source FILE            Source .h5 recording path (absolute or repo-relative)"
    echo "  --track TRACK            Track tag (e.g., oval, sloop)"
    echo "  --duration TAG           Duration tag (e.g., 20s, 45s, 60s)"
    echo ""
    echo "Options:"
    echo "  --sync-policy MODE       Sync policy tag in filename (default: latest)"
    echo "  --date YYYYMMDD          Date tag in filename (default: today)"
    echo "  --mode MODE              copy or move (default: copy)"
    echo "  --force                  Overwrite existing destination file"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Output filename format:"
    echo "  golden_gt_<date>_<track>_<sync-policy>_<duration>.h5"
    echo ""
    echo "Example:"
    echo "  $0 --source data/recordings/recording_20260214_114904.h5 --track sloop --duration 45s"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --track)
            TRACK="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --sync-policy)
            SYNC_POLICY="$2"
            shift 2
            ;;
        --date)
            DATE_TAG="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$SOURCE" || -z "$TRACK" || -z "$DURATION" ]]; then
    echo "Missing required arguments."
    print_usage
    exit 1
fi

if [[ "$MODE" != "copy" && "$MODE" != "move" ]]; then
    echo "Invalid --mode: $MODE (expected: copy or move)"
    exit 1
fi

if [[ ! "$DATE_TAG" =~ ^[0-9]{8}$ ]]; then
    echo "Invalid --date: $DATE_TAG (expected: YYYYMMDD)"
    exit 1
fi

sanitize_tag() {
    local raw="$1"
    raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
    raw="$(echo "$raw" | tr -c 'a-z0-9._-' '_')"
    raw="$(echo "$raw" | sed -E 's/_+/_/g; s/^_+//; s/_+$//')"
    echo "$raw"
}

TRACK_TAG="$(sanitize_tag "$TRACK")"
DURATION_TAG="$(sanitize_tag "$DURATION")"
SYNC_TAG="$(sanitize_tag "$SYNC_POLICY")"

if [[ -z "$TRACK_TAG" || -z "$DURATION_TAG" || -z "$SYNC_TAG" ]]; then
    echo "Invalid tag values after sanitization."
    exit 1
fi

if [[ "$SOURCE" = /* ]]; then
    SOURCE_PATH="$SOURCE"
else
    SOURCE_PATH="$REPO_ROOT/$SOURCE"
fi

if [[ ! -f "$SOURCE_PATH" ]]; then
    echo "Source file not found: $SOURCE_PATH"
    exit 1
fi

mkdir -p "$RECORDINGS_DIR"
DEST_PATH="$RECORDINGS_DIR/golden_gt_${DATE_TAG}_${TRACK_TAG}_${SYNC_TAG}_${DURATION_TAG}.h5"

if [[ -f "$DEST_PATH" && "$FORCE" != true ]]; then
    echo "Destination already exists: $DEST_PATH"
    echo "Re-run with --force to overwrite."
    exit 1
fi

if [[ "$MODE" == "move" ]]; then
    if [[ "$FORCE" == true ]]; then
        mv -f "$SOURCE_PATH" "$DEST_PATH"
    else
        mv "$SOURCE_PATH" "$DEST_PATH"
    fi
else
    if [[ "$FORCE" == true ]]; then
        cp -f "$SOURCE_PATH" "$DEST_PATH"
    else
        cp "$SOURCE_PATH" "$DEST_PATH"
    fi
fi

echo "Golden GT file ready:"
echo "$DEST_PATH"
