import cv2
import os
import json
import numpy as np

def is_blurry(image, threshold=100.0):
    """
    Computes the Laplacian variance of the image.
    Low variance indicates high blur / motion drag.
    
    Arguments:
        image (np.ndarray): Grayscale or BGR image frame.
        threshold (float): Variance threshold below which frame is flagged blurry.
        
    Returns:
        tuple: (bool, float) -> (is_blurry_flag, variance_score)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold, var


def extract_gopro_intrinsic_frames(video_path, output_dir, board, dictionary, 
                                   stride=15, min_corners=6, blur_threshold=100.0, max_frames=50):
    """
    Intelligent extraction for GoPro Stage 2 Intrinsic Calibration.
    Filters out motion blur, detects ChArUco corners, verifies geometric coverage,
    and saves only high-quality calibration candidates.
    
    Arguments:
        video_path (str): Path to the GoPro intrinsic recording (.mp4).
        output_dir (str): Directory where candidate calibration frames are written.
        board (cv2.aruco.CharucoBoard): Configured ChArUco board object.
        dictionary (cv2.aruco.Dictionary): Configured ArUco dictionary.
        stride (int): Frame inspection step size (e.g. 15 = check every 15th frame).
        min_corners (int): Minimum valid interpolated corners required.
        blur_threshold (float): Minimum Laplacian variance for motion sharpness.
        max_frames (int): Target ceiling for calibration frames.
        
    Returns:
        saved_count (int): Number of high-quality frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"[VideoSplitter] Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[VideoSplitter] Scanning {video_path} for GoPro intrinsics ({total_frames} total frames)...")

    # Set up OpenCV ArUco detector parameters
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    frame_idx = 0
    saved_count = 0
    last_saved_corners_mean = None

    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break

        if frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Motion blur rejection
            blurry, var_score = is_blurry(gray, blur_threshold)
            if blurry:
                frame_idx += 1
                continue

            # 2. ArUco marker detection
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)

            if marker_ids is not None and len(marker_ids) > 0:
                # 3. Interpolate ChArUco corners
                ret_val, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board
                )

                if ret_val is not None and ret_val >= min_corners:
                    # 4. Angular/spatial diversity gate (prevent saving identical static frames)
                    current_mean = np.mean(charuco_corners, axis=0)
                    if last_saved_corners_mean is not None:
                        dist = np.linalg.norm(current_mean - last_saved_corners_mean)
                        # If the board centroid moved less than 25 pixels, skip to preserve diversity
                        if dist < 25.0:
                            frame_idx += 1
                            continue

                    # Frame is sharp, board is detected, and position is distinct
                    out_path = os.path.join(output_dir, f"calib_gopro_{saved_count:04d}_f{frame_idx:06d}.jpg")
                    cv2.imwrite(out_path, frame)
                    last_saved_corners_mean = current_mean
                    saved_count += 1
                    print(f"  [Saved] Frame {frame_idx:06d} | Corners: {ret_val} | Sharpness: {var_score:.1f} -> {out_path}")

        frame_idx += 1

    cap.release()
    print(f"[VideoSplitter] GoPro Intrinsic Extraction complete. Saved {saved_count} sharp, diverse frames to: {output_dir}\n")
    return saved_count


def split_video_for_calibration(video_path, output_dir, frame_skip=1):
    """
    Standard blind extraction used ONLY for Stage 2 Calibration (ChArUco boards).
    Extracts every Nth frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_skip == 0:
            out_name = os.path.join(output_dir, f"calib_frame_{saved:06d}.jpg")
            cv2.imwrite(out_name, frame)
            saved += 1
            
        count += 1
        
    cap.release()
    print(f"[VideoSplitter] Calibration split complete. Saved {saved} frames to {output_dir}")


def extract_synced_frames(video_path, synced_json_path, output_dir):
    """
    Extracts perfectly synchronized frames by following the master map 
    created by align_jsons.py. It renames the frames to the unified timeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the synced map
    with open(synced_json_path, 'r') as f:
        synced_data = json.load(f)
        
    # Create a dictionary mapping the original hardware frame to the new unified frame
    # e.g., { 10: 10, 11: 11, 13: 12 } <- Frame 12 was dropped by hardware
    frame_map = {item['orig_hardware_index']: item['frame_index'] for item in synced_data}
    
    cap = cv2.VideoCapture(video_path)
    
    print(f"\n[VideoSplitter] Extracting synced frames using map: {synced_json_path}")
    current_hw_index = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 2. Check if this hardware frame survived the sync cuts
        if current_hw_index in frame_map:
            unified_index = frame_map[current_hw_index]
            
            # 3. Save it under the new unified timeline name
            out_name = os.path.join(output_dir, f"frame_{unified_index:06d}.jpg")
            cv2.imwrite(out_name, frame)
            saved_count += 1
            
        current_hw_index += 1
        
    cap.release()
    print(f"[VideoSplitter] Success. Extracted {saved_count} perfectly synced frames to {output_dir}")


if __name__ == "__main__":
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 8), 0.10795, 0.08128, aruco_dict)
    extract_gopro_intrinsic_frames(
        video_path="test.mp4",
        output_dir="gopro_0_intrinsic_frames",
        board=board,
        dictionary=aruco_dict,
        stride=1,
        min_corners=1,
        blur_threshold=8,
        max_frames=100
    )
    