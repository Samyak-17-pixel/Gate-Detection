# Local datasets

Keep large image and video sets **out of git**. Place them at the repo root (these paths are already in `.gitignore`):

| Path | Typical contents |
|------|------------------|
| `images/` | `.png` frames, e.g. `images/image_navigation_01/`, `images/image_qualification_01/` |
| `air_dataset/` | Optional extra stills |
| `gopro_videos/` | Source video (not committed) |

Batch scripts resolve `--folder` relative to the **repository root**, not the `scripts/` directory.

```bash
python3 scripts/navigation_gate_detector.py --folder images/image_navigation_01 --no-gui
```
