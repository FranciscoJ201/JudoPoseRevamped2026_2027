# Change Log

## 2026-10-20: Reorganization and cleanup
**Moves (`git mv`, history preserved, contents unchanged unless listed below)**
| Old | New |
|---|---|
| `DualRealSenseRecord.py` | `capture/dual_realsense_record.py` |
| `RealSenseExtrinsicCapture.py` | `capture/realsense_calib_capture.py` |
| `CompressPostRecord.py` | `capture/compress_post_record.py` |
| `IntrinsicAutomated.py` | `calibration/extract_intrinsic_frames.py` |
| `CalculateIntrinsics.py` | `calibration/calculate_intrinsics.py` |
| `RealsenseTesting/SimpleExtrinsicTest.py` | `calibration/stereo_extrinsics.py` |
| `ResolutionTester.py` | `tools/list_realsense_profiles.py` |
| `RealsenseTesting/pointcloudtesting.py` | `experiments/pointcloudtesting.py` |
| `RealsenseTesting/prelimExtrinsicCheck.py` | `experiments/prelimExtrinsicCheck.py` |
| `CalculateExtrinsics.py` | `archive/CalculateExtrinsics.py` (superseded by stereo extrinsics, not deleted) |
| `gopro_intrinsics.json` (untracked) | `calibration/results/gopro_intrinsics.json` |
| `currentProgress.md`, `PhysPose_Detailed_Notes.md` (untracked) | `docs/` |

The empty `RealsenseTesting/` folder was removed. `.gitignore` is unchanged.

**New files**
- `common/__init__.py`: marks `common` as a package.
- `common/intrinsics_io.py`: `load_intrinsics(json_path)` returns (K, D, model, width, height); `resolve_intrinsics_path(explicit_path, cam_dir)` returns a validated intrinsics path.
- `common/charuco.py`: `get_aruco_dictionary(dict_name)`; `build_charuco_detector(squares_x, squares_y, square_len, marker_len, dict_name)` returns (board, detector).
- `ChangeLog.md`: this file.

**Edited files**
- `calibration/stereo_extrinsics.py`
  - Imports the shared helpers from `common/` and removes its local copies of `load_intrinsics`, `resolve_intrinsics_path`, `get_aruco_dictionary`.
  - Adds a `sys.path` insert of the repo root so `common` imports resolve when run as a script.
  - Removes the `t1_world_global` global; `plot_calibration_verification_3d` now takes a `t1_world` argument.
  - Shared-corner matching per frame pair now uses `np.intersect1d(..., return_indices=True)` and array indexing instead of dictionaries and list comprehensions. Output ordering is unchanged (sorted corner ids).
- `capture/compress_post_record.py`: `main()` now uses argparse (`root` positional, `--keep_loose` flag) instead of `sys.argv`. Default behaviour is unchanged.
- `docs/currentProgress.md`: rewritten as the tentative protocol (stage status, folder purposes, run order with inputs and outputs, known issues, environments).

**Environment and remote changes (same day)**
- `~/.ssh/config`: appended `vrpc` and `empire` Host entries (backup at `~/.ssh/config.bak-<date>`).
- VR PC: created the empty folder `~/JudoPoseRevamped2026_2027`.

**Verification**
`python -m py_compile` passes on all moved and edited scripts. `common` imports and loading `gopro_intrinsics.json` were tested. `stereo_extrinsics.py` has not been rerun end to end on real images.
