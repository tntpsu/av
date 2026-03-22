# Trimmed hill_highway recording (fixture)

- **File:** `hill_highway_trim_400f_no_camera.h5` (~**10 MiB** on disk)
- **Source:** `data/recordings/recording_20260321_215414.h5` (trimmed; not committed under `data/`)
- **Contents:** first **400** frames of time-series + control/vehicle/perception/ground-truth; **no** `camera/images` or `camera/topdown_images` (those dominate size).

**GitHub:** ~10 MiB is below the **100 MiB** blob limit and fine for normal repos (optional: stay &lt;50 MiB policy — this qualifies).

**Regenerate:**

```bash
python tools/trim_recording_h5.py \
  data/recordings/<your_recording>.h5 \
  tests/fixtures/hill_highway_trim_400f_no_camera.h5 \
  --max-frames 400
```

Use `--keep-camera` only if you need pixels (file grows toward **hundreds of MiB** quickly).
