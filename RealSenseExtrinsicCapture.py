import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import sys
import json

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# Scaled down to 640x360 to match production capture resolution exactly
W, H       = 640, 360
TARGET_FPS = 30          # 30fps is plenty for a still-capture tool

# Leave empty to auto-detect all connected RealSense devices.
SERIAL_NUMBERS = []

# Root output folder — one subfolder per camera will be created here.
# e.g.  calib_capture/realsense_cam_0/board_images/top_left/
OUTPUT_ROOT = 'calib_capture'

# ─────────────────────────────────────────────
#  POSITIONS AND DISPLAY LAYOUT
#
#  These match the flat-board calibration protocol in extrinsic.py.
#  The operator selects a position with number keys, then captures
#  with SPACE. Images are saved to:
#      <OUTPUT_ROOT>/realsense_cam_<N>/board_images/<position>/
#
#  No facing subfolders — the flat-board calibration code doesn't
#  need them. Position subfolders give you coverage QC at a glance.
# ─────────────────────────────────────────────

POSITIONS = [
    'top_left',
    'top_right',
    'bottom_right',
    'bottom_left',
    'centre',
]

# Key bindings: 1–5 select position, SPACE captures, Q quits, A takes anchor shot
POSITION_KEYS = {
    ord('1'): 0,
    ord('2'): 1,
    ord('3'): 2,
    ord('4'): 3,
    ord('5'): 4,
}

# Overlay colours
COLOR_ACTIVE   = (0,   220,  0)    # green  — current position label
COLOR_INACTIVE = (160, 160, 160)   # grey   — other positions in HUD
COLOR_CAPTURE  = (0,   200, 255)   # yellow — flash on capture
COLOR_ANCHOR   = (255, 120,   0)   # orange — anchor shot indicator
COLOR_COUNT    = (255, 255, 255)   # white  — shot counter


# ─────────────────────────────────────────────
#  OUTPUT DIRECTORY HELPERS
# ─────────────────────────────────────────────

def make_output_dirs(cam_labels):
    """
    Creates the full folder tree for all cameras and positions.

    Structure:
        calib_capture/
            realsense_cam_0/
                anchor/
                board_images/
                    top_left/
                    top_right/
                    bottom_right/
                    bottom_left/
                    centre/
            realsense_cam_1/
                ...
    """
    dirs = {}
    for label in cam_labels:
        cam_root  = os.path.join(OUTPUT_ROOT, label)
        anchor_dir = os.path.join(cam_root, 'anchor')
        os.makedirs(anchor_dir, exist_ok=True)

        position_dirs = {}
        for pos in POSITIONS:
            pos_dir = os.path.join(cam_root, 'board_images', pos)
            os.makedirs(pos_dir, exist_ok=True)
            position_dirs[pos] = pos_dir

        dirs[label] = {
            'root':      cam_root,
            'anchor':    anchor_dir,
            'positions': position_dirs,
        }

    return dirs


def count_existing_shots(dirs):
    """Returns a dict of {cam_label: {position: count}} for the HUD."""
    counts = {}
    for label, cam_dirs in dirs.items():
        counts[label] = {}
        for pos, pos_dir in cam_dirs['positions'].items():
            n = len([f for f in os.listdir(pos_dir)
                     if f.lower().endswith(('.jpg', '.png'))])
            counts[label][pos] = n
    return counts


# ─────────────────────────────────────────────
#  DEVICE DISCOVERY
# ─────────────────────────────────────────────

def discover_devices():
    """
    Queries the RealSense context for connected physical devices.
    
    Returns:
        list: Serial numbers of detected cameras.
    """
    ctx     = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        raise RuntimeError("No RealSense devices found. Check USB connections.")

    serials = []
    print(f"\nDetected {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name   = dev.get_info(rs.camera_info.name)
        print(f"  [{i}] {name}  serial: {serial}")
        serials.append(serial)

    return serials


# ─────────────────────────────────────────────
#  INTRINSICS SAVE
# ─────────────────────────────────────────────

def save_intrinsics(profile, cam_label, serial, cam_root):
    """
    Saves camera intrinsics JSON alongside the calibration images.
    
    Arguments:
        profile: Active pipeline profile.
        cam_label (str): Label assigned to this camera.
        serial (str): Camera serial number.
        cam_root (str): Root destination folder for camera outputs.
    """
    color_intr = (profile
                  .get_stream(rs.stream.color)
                  .as_video_stream_profile()
                  .get_intrinsics())

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    intrinsics = {
        "serial":            serial,
        "cam_label":         cam_label,
        "depth_scale":       depth_scale,
        "fx":                color_intr.fx,
        "fy":                color_intr.fy,
        "cx":                color_intr.ppx,
        "cy":                color_intr.ppy,
        "width":             color_intr.width,
        "height":            color_intr.height,
        "distortion_model":  str(color_intr.model),
        "distortion_coeffs": list(color_intr.coeffs),
    }

    intr_path = os.path.join(cam_root, 'intrinsics.json')
    with open(intr_path, 'w') as f:
        json.dump(intrinsics, f, indent=4)
    print(f"  [{cam_label}] Intrinsics saved → {intr_path}")


# ─────────────────────────────────────────────
#  HUD OVERLAY
# ─────────────────────────────────────────────

def draw_hud(img, cam_label, current_pos_idx, shot_counts, flash_alpha,
             anchor_count, flash_is_anchor=False):
    """
    Draws the position menu and shot counter onto the preview image in-place.

    Layout (top-left corner):
        [cam label]
        1: top_left      (N shots)
        2: top_right     (N shots)
        ...
        A: anchor        (N shots)
        SPACE to capture | Q to quit
    """
    overlay = img.copy()

    # Dark background panel for readability
    panel_w = 320
    panel_h = 30 + len(POSITIONS) * 26 + 60
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    y = 22
    cv2.putText(img, f'[ {cam_label} ]', (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ACTIVE, 1, cv2.LINE_AA)
    y += 24

    for i, pos in enumerate(POSITIONS):
        is_active = (i == current_pos_idx)
        count     = shot_counts.get(cam_label, {}).get(pos, 0)
        color     = COLOR_ACTIVE if is_active else COLOR_INACTIVE
        prefix    = '►' if is_active else ' '
        label     = f"{prefix} {i+1}: {pos:<14} {count} shot{'s' if count != 1 else ''}"
        cv2.putText(img, label, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 26

    # Anchor line
    anchor_color = COLOR_ANCHOR if anchor_count > 0 else COLOR_INACTIVE
    cv2.putText(img, f'  A: anchor          {anchor_count} shot{"s" if anchor_count != 1 else ""}',
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, anchor_color, 1, cv2.LINE_AA)
    y += 26

    cv2.putText(img, 'SPACE: capture  Q: quit', (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_INACTIVE, 1, cv2.LINE_AA)

    # Capture flash — brief colour tint on the full frame
    if flash_alpha > 0:
        flash_color = COLOR_ANCHOR if flash_is_anchor else COLOR_CAPTURE
        tint        = np.full_like(img, flash_color, dtype=np.uint8)
        cv2.addWeighted(tint, flash_alpha, img, 1.0 - flash_alpha, 0, img)


# ─────────────────────────────────────────────
#  CAPTURE HELPER
# ─────────────────────────────────────────────

def capture_frames(pipelines, cam_labels, dirs, current_pos_idx,
                   shot_counts, is_anchor=False):
    """
    Grabs the latest frame from every pipeline simultaneously and saves them.
    Returns the number of cameras successfully saved.
    """
    saved = 0
    timestamp = int(time.time() * 1000)

    for (label, pipeline) in pipelines:
        try:
            frames      = pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if not color_frame:
                print(f"  [{label}] Warning: no colour frame — skipping this camera.")
                continue

            img = np.asanyarray(color_frame.get_data())

            if is_anchor:
                save_dir  = dirs[label]['anchor']
                filename  = f'anchor_{timestamp}.jpg'
            else:
                position  = POSITIONS[current_pos_idx]
                save_dir  = dirs[label]['positions'][position]
                filename  = f'{position}_{timestamp}.jpg'

            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if is_anchor:
                dirs[label]['_anchor_count'] = dirs[label].get('_anchor_count', 0) + 1
            else:
                shot_counts[label][position] = shot_counts[label].get(position, 0) + 1

            saved += 1

        except Exception as e:
            print(f"  [{label}] Capture error: {e}")

    return saved


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    # ── Device discovery ─────────────────────────────────────────────────────
    serials = SERIAL_NUMBERS if SERIAL_NUMBERS else discover_devices()

    if not serials:
        print("No cameras found. Exiting.")
        sys.exit(1)

    cam_labels = [f'realsense_cam_{i}' for i in range(len(serials))]

    # ── Output directories ───────────────────────────────────────────────────
    dirs        = make_output_dirs(cam_labels)
    shot_counts = count_existing_shots(dirs)
    anchor_counts = {label: len([f for f in os.listdir(dirs[label]['anchor'])
                                  if f.lower().endswith(('.jpg', '.png'))])
                     for label in cam_labels}

    print(f"\nOutput root : {os.path.abspath(OUTPUT_ROOT)}")
    for label in cam_labels:
        print(f"  {label}/board_images/  — {sum(shot_counts[label].values())} existing shots")

    # ── Start pipelines ──────────────────────────────────────────────────────
    pipelines = []
    for serial, label in zip(serials, cam_labels):
        print(f"\n  [{label}] Starting pipeline (serial: {serial})...")
        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, TARGET_FPS)
        profile  = pipeline.start(config)

        save_intrinsics(profile, label, serial, dirs[label]['root'])
        pipelines.append((label, pipeline))

    print(f"\n{'='*55}")
    print(f"  EXTRINSIC CALIBRATION CAPTURE")
    print(f"{'='*55}")
    print(f"  1-5  : select board position")
    print(f"  A    : take anchor shot (clean mat, no board)")
    print(f"  SPACE: capture from all {len(pipelines)} camera(s) simultaneously")
    print(f"  Q    : quit and save")
    print(f"{'='*55}\n")

    # ── State ────────────────────────────────────────────────────────────────
    current_pos_idx  = 0
    flash_alpha      = 0.0       # decays each frame to create the capture flash
    flash_is_anchor  = False
    last_capture_msg = ''

    try:
        while True:
            # ── Grab preview frames from all cameras ─────────────────────────
            preview_frames = []
            for label, pipeline in pipelines:
                try:
                    frames      = pipeline.wait_for_frames(timeout_ms=200)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        img = np.asanyarray(color_frame.get_data()).copy()
                    else:
                        img = np.zeros((H, W, 3), dtype=np.uint8)
                except Exception:
                    img = np.zeros((H, W, 3), dtype=np.uint8)

                # Draw HUD onto each camera's preview
                anchor_count = anchor_counts.get(label, 0)
                draw_hud(img, label, current_pos_idx, shot_counts,
                         flash_alpha, anchor_count, flash_is_anchor)
                preview_frames.append(img)

            # ── Composite all cameras side by side ───────────────────────────
            composite = np.hstack(preview_frames)

            # Status bar at the bottom of the composite
            status_h  = 36
            status_bar = np.zeros((status_h, composite.shape[1], 3), dtype=np.uint8)
            pos_name   = POSITIONS[current_pos_idx]
            total_shots = sum(sum(shot_counts[l].values()) for l in cam_labels)
            status_text = (f"Position: {pos_name}  |  "
                           f"Total shots: {total_shots}  |  "
                           f"{last_capture_msg}")
            cv2.putText(status_bar, status_text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_COUNT, 1, cv2.LINE_AA)
            composite = np.vstack([composite, status_bar])

            cv2.imshow("Extrinsic Capture — SPACE: capture  1-5: position  A: anchor  Q: quit",
                       composite)

            # Decay flash
            flash_alpha = max(0.0, flash_alpha - 0.08)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:   # Q or ESC
                break

            elif key in POSITION_KEYS:
                current_pos_idx = POSITION_KEYS[key]
                last_capture_msg = f"Selected: {POSITIONS[current_pos_idx]}"
                print(f"  Position → {POSITIONS[current_pos_idx]}")

            elif key == ord('a'):
                # Anchor shot — clean mat, no board
                saved = capture_frames(pipelines, cam_labels, dirs,
                                       current_pos_idx, shot_counts,
                                       is_anchor=True)
                for label in cam_labels:
                    anchor_counts[label] = anchor_counts.get(label, 0) + (1 if saved else 0)
                flash_alpha     = 0.45
                flash_is_anchor = True
                last_capture_msg = f"Anchor saved ({saved}/{len(pipelines)} cameras)"
                print(f"  Anchor captured — {saved}/{len(pipelines)} cameras saved.")

            elif key == ord(' '):
                # Board shot at current position
                saved = capture_frames(pipelines, cam_labels, dirs,
                                       current_pos_idx, shot_counts,
                                       is_anchor=False)
                pos_name = POSITIONS[current_pos_idx]
                total_at_pos = shot_counts[cam_labels[0]].get(pos_name, 0)
                flash_alpha     = 0.45
                flash_is_anchor = False
                last_capture_msg = (f"Saved {pos_name} "
                                    f"({total_at_pos} shots here, "
                                    f"{saved}/{len(pipelines)} cams)")
                print(f"  Captured [{pos_name}] — "
                      f"{total_at_pos} shots at this position, "
                      f"{saved}/{len(pipelines)} cameras saved.")

    finally:
        cv2.destroyAllWindows()
        for _, pipeline in pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass

        # ── Summary ──────────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"  CAPTURE SUMMARY")
        print(f"{'='*55}")
        for label in cam_labels:
            print(f"\n  {label}:")
            for pos in POSITIONS:
                n = shot_counts[label].get(pos, 0)
                bar   = '█' * n + '░' * max(0, 5 - n)
                flag  = '  ✓' if n >= 3 else ('  ⚠ aim for 3+' if n > 0 else '  ✗ MISSING')
                print(f"    {pos:<16} {bar}  {n:2d} shots{flag}")
            anch = anchor_counts.get(label, 0)
            flag = '  ✓' if anch >= 1 else '  ✗ MISSING — needed for Phase 1 mat corner click'
            print(f"    {'anchor':<16} {'█' * min(anch, 5)}  {anch:2d} shots{flag}")

        print(f"\n  Output saved to: {os.path.abspath(OUTPUT_ROOT)}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()