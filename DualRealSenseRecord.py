import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
import queue
import sys
import os
import csv
import json
import tarfile
import shutil

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# Scaled down to 640x360 to respect USB 3.0 bandwidth limits across dual cameras
COLOR_W, COLOR_H = 640, 360
DEPTH_W, DEPTH_H = 640, 360

TARGET_FPS = 60

# Leave as empty list to auto-detect all connected RealSense devices.
# Or specify serial numbers explicitly: ['123456789', '987654321']
SERIAL_NUMBERS = []

WARMUP_SECONDS = 5

# ─────────────────────────────────────────────
#  PER-CAMERA OUTPUT STRUCTURE
# ─────────────────────────────────────────────

def make_camera_output_dir(cam_label):
    """Creates and returns the output directory for one camera."""
    out_dir = f'realsense_{cam_label}'
    os.makedirs(os.path.join(out_dir, 'depth_frames'), exist_ok=True)
    return out_dir


# ─────────────────────────────────────────────
#  WRITER THREADS
# ─────────────────────────────────────────────

def depth_writer_thread(depth_queue, output_folder, stop_flag):
    """
    Consumes (frame_index, uint16 depth array) tuples and saves them
    as raw .npy files to bypass the severe CPU bottleneck of PNG compression.
    
    Sample in post:  z_m = np.load('depth_XXXXXX.npy')[v, u] / 1000.0
    """
    while not stop_flag.is_set() or not depth_queue.empty():
        try:
            frame_idx, depth_arr = depth_queue.get(timeout=0.1)
            path = os.path.join(output_folder, f'depth_{frame_idx:06d}.npy')
            np.save(path, depth_arr)
            depth_queue.task_done()
        except queue.Empty:
            continue


def color_writer_thread(frame_queue, writer, stop_flag):
    """Consumes BGR numpy arrays and encodes them to the MP4 file."""
    while not stop_flag.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.1)
            writer.write(frame)
            frame_queue.task_done()
        except queue.Empty:
            continue


# ─────────────────────────────────────────────
#  TIMESTAMP HELPER
# ─────────────────────────────────────────────

def get_frame_timestamps(color_frame):
    """
    Returns (sensor_timestamp_ms, time_of_arrival_ms, host_time_s).
    """
    sensor_ts  = color_frame.get_timestamp()
    arrival_ts = color_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
    host_ts    = time.time()
    return sensor_ts, arrival_ts, host_ts


# ─────────────────────────────────────────────
#  CAMERA WORKER
# ─────────────────────────────────────────────

def camera_worker(serial, cam_label, ready_event, global_stop):
    """
    Full recording lifecycle for one isolated RealSense camera thread.
    """
    out_dir    = make_camera_output_dir(cam_label)
    depth_dir  = os.path.join(out_dir, 'depth_frames')
    video_path = os.path.join(out_dir, 'color.mp4')
    csv_path   = os.path.join(out_dir, 'timestamps.csv')
    intr_path  = os.path.join(out_dir, 'intrinsics.json')

    print(f"  [{cam_label}] Initialising pipeline (serial: {serial})...")

    # ── Pipeline targeted to this specific serial number ─────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    
    # Native bgr8 format with lower resolution to prevent USB bandwidth saturation
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, TARGET_FPS)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  TARGET_FPS)

    profile = pipeline.start(config)
    align   = rs.align(rs.stream.color)

    # ── Depth Sensor & Depth Scale ───────────────────────────────────────────
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    # ── Intrinsics ───────────────────────────────────────────────────────────
    color_intr = (profile
                  .get_stream(rs.stream.color)
                  .as_video_stream_profile()
                  .get_intrinsics())

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
    with open(intr_path, 'w') as f:
        json.dump(intrinsics, f, indent=4)
    print(f"  [{cam_label}] Intrinsics & depth scale ({depth_scale}m) saved → {intr_path}")

    # ── Video writer & Threads ───────────────────────────────────────────────
    fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
    color_writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (COLOR_W, COLOR_H))

    color_queue = queue.Queue(maxsize=300)   
    depth_queue = queue.Queue(maxsize=300)
    local_stop  = threading.Event()          

    t_color = threading.Thread(target=color_writer_thread, args=(color_queue, color_writer, local_stop), daemon=True)
    t_depth = threading.Thread(target=depth_writer_thread, args=(depth_queue, depth_dir, local_stop), daemon=True)
    
    t_color.start()
    t_depth.start()

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame_index',
        'sensor_timestamp_ms',   
        'time_of_arrival_ms',    
        'host_time_s',
        'color_dropped',
        'depth_dropped',
    ])

    # ── Warm-up ──────────────────────────────────────────────────────────────
    print(f"  [{cam_label}] Warming up for {WARMUP_SECONDS}s...")
    warmup_end = time.time() + WARMUP_SECONDS
    while time.time() < warmup_end:
        pipeline.wait_for_frames()
    print(f"  [{cam_label}] Warm-up complete — recording started.")

    ready_event.set()

    # ── Main recording loop ──────────────────────────────────────────────────
    frame_count = 0
    start_time  = time.time()

    try:
        while not global_stop.is_set():
            
            # --- THE SAFETY NET ---
            try:
                # 1 second timeout prevents the thread from hanging completely if USB disconnects
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                
                # Verify hardware actually provided both frames before math
                if not frames.get_color_frame() or not frames.get_depth_frame():
                    continue

                aligned = align.process(frames)

            except RuntimeError as e:
                # Catch the dreaded USB bandwidth hiccup without killing the thread
                print(f"  [{cam_label}] USB bandwidth hiccup — skipping corrupted frame.")
                continue
            except Exception as e:
                print(f"  [{cam_label}] Unexpected error: {e}")
                break
            # ------------------------

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Timestamps
            sensor_ts, arrival_ts, host_ts = get_frame_timestamps(color_frame)

            # Color → queue
            color_dropped = 0
            color_image = np.asanyarray(color_frame.get_data())
            try:
                color_queue.put(color_image, block=False)
            except queue.Full:
                color_dropped = 1
                print(f"  [{cam_label}] Warning: color queue full at frame {frame_count} — dropping frame.")

            # Depth (.npy) → queue
            depth_dropped = 0
            depth_image = np.asanyarray(depth_frame.get_data())
            try:
                depth_queue.put((frame_count, depth_image), block=False)
            except queue.Full:
                depth_dropped = 1
                print(f"  [{cam_label}] Warning: depth queue full at frame {frame_count} — dropping depth frame.")

            csv_writer.writerow([frame_count, sensor_ts, arrival_ts, host_ts, color_dropped, depth_dropped])

            frame_count += 1

            if frame_count % (TARGET_FPS * 10) == 0:
                elapsed = time.time() - start_time
                print(f"  [{cam_label}] {frame_count} frames ({elapsed:.1f}s, {frame_count/elapsed:.1f} fps actual)")

    finally:
        local_stop.set()
        t_color.join()
        t_depth.join()

        elapsed    = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0

        pipeline.stop()
        color_writer.release()
        csv_file.close()

        print(f"\n  [{cam_label}] Finished recording — {frame_count} frames @ {actual_fps:.2f} fps actual")
        print(f"    Color video  : {video_path}")
        print(f"    Depth frames : {depth_dir}/")
        print(f"    Timestamps   : {csv_path}")

        # Package loose depth .npy files into an uncompressed tar archive


# ─────────────────────────────────────────────
#  DEVICE DISCOVERY & MAIN
# ─────────────────────────────────────────────

def discover_devices():
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


def main():
    if SERIAL_NUMBERS:
        serials = SERIAL_NUMBERS
        print(f"\nUsing configured serial numbers: {serials}")
    else:
        serials = discover_devices()

    if len(serials) == 0:
        print("No cameras to record. Exiting.")
        sys.exit(1)

    if len(serials) > 2:
        print(f"Warning: Using first 2 of {len(serials)} detected cameras.")
        serials = serials[:2]

    cam_labels = [f'cam_{i}' for i in range(len(serials))]

    # ── Preview ──────────────────────────────────────────────────────────────
    print(f"\nOpening preview for {len(serials)} camera(s).")
    print("Press 'c' to start recording, 'q' to quit.")

    preview_pipelines = []
    for serial, label in zip(serials, cam_labels):
        p   = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, TARGET_FPS)
        p.start(cfg)
        preview_pipelines.append((label, p))

    while True:
        imgs = []
        for label, p in preview_pipelines:
            frames = p.wait_for_frames()
            cf     = frames.get_color_frame()
            if cf:
                img = np.asanyarray(cf.get_data())
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                imgs.append(img)

        if imgs:
            cv2.imshow("Preview — 'c' record, 'q' quit", np.hstack(imgs))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            for _, p in preview_pipelines:
                p.stop()
            sys.exit(0)

    for _, p in preview_pipelines:
        p.stop()
    time.sleep(1.0)

    # ── Launch Threads ───────────────────────────────────────────────────────
    global_stop  = threading.Event()
    ready_events = [threading.Event() for _ in serials]
    cam_threads  = []

    for serial, label, ready_ev in zip(serials, cam_labels, ready_events):
        t = threading.Thread(
            target=camera_worker,
            args=(serial, label, ready_ev, global_stop),
            daemon=True
        )
        t.start()
        cam_threads.append(t)

    # ── Wait for Warmup ──────────────────────────────────────────────────────
    print(f"\nWaiting for all {len(serials)} cameras to complete warm-up...")
    for ev, label in zip(ready_events, cam_labels):
        ev.wait()
        print(f"  {label} — ready")

    print("\nAll cameras recording simultaneously.")
    print("Press Ctrl+C to stop all cameras at once.\n")

    # ── Hold main thread ─────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C — stopping all cameras simultaneously...")
        global_stop.set()

    for t in cam_threads:
        t.join()

    print("\nAll cameras stopped cleanly.")

if __name__ == "__main__":
    main()