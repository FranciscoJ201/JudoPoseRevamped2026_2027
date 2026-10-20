import argparse
import glob
import json
import os
import sys
import cv2
import numpy as np

# Make the repo root importable so `common` resolves when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.charuco import build_charuco_detector
from common.intrinsics_io import load_intrinsics, resolve_intrinsics_path


def parse_arguments():
    """
    Parses CLI arguments for multi-pair ChArUco stereo calibration.
    """
    parser = argparse.ArgumentParser(
        description="Compute stereo camera extrinsics by pooling multi-view ChArUco image pairs across subfolders."
    )
    # Camera directories
    parser.add_argument(
        "--cam1_dir", type=str, required=True,
        help="Root directory for camera 1 (e.g. calib_capture/realsense_cam_0)."
    )
    parser.add_argument(
        "--cam2_dir", type=str, required=True,
        help="Root directory for camera 2 (e.g. calib_capture/realsense_cam_1)."
    )
    # Intrinsics paths (auto-resolved from directories if omitted)
    parser.add_argument(
        "--intrinsics1", "-i1", type=str, default=None,
        help="Path to camera 1 intrinsics JSON (defaults to <cam1_dir>/intrinsics.json)."
    )
    parser.add_argument(
        "--intrinsics2", "-i2", type=str, default=None,
        help="Path to camera 2 intrinsics JSON (defaults to <cam2_dir>/intrinsics.json)."
    )
    parser.add_argument(
        "--cam1_id", type=str, default="cam_1",
        help="Identifier label for camera 1."
    )
    parser.add_argument(
        "--cam2_id", type=str, default="cam_2",
        help="Identifier label for camera 2."
    )
    # ChArUco board dimensions (800x600 board, 60mm checker, 45mm marker, 9x12 squares)
    parser.add_argument(
        "--squares_x", type=int, default=12,
        help="Number of chessboard squares along X dimension."
    )
    parser.add_argument(
        "--squares_y", type=int, default=9,
        help="Number of chessboard squares along Y dimension."
    )
    parser.add_argument(
        "--square_len", type=float, default=0.060,
        help="Physical square side length in meters."
    )
    parser.add_argument(
        "--marker_len", type=float, default=0.045,
        help="Physical ArUco marker side length in meters."
    )
    parser.add_argument(
        "--dictionary", type=str, default="DICT_5X5_50",
        help="ArUco predefined dictionary identifier."
    )
    # Operational thresholds
    parser.add_argument(
        "--min_corners", type=int, default=5,
        help="Minimum number of shared corners required per frame pair."
    )
    parser.add_argument(
        "--min_pairs", type=int, default=1,
        help="Minimum number of valid synchronized pairs required to proceed."
    )
    parser.add_argument(
        "--output_json", "-o", type=str, default="extrinsics_stereo.json",
        help="Output path for computed extrinsic calibration JSON."
    )
    # Visualization options
    parser.add_argument(
        "--plot", action="store_true",
        help="Display interactive matplotlib 3D geometry and residual scatter plots."
    )
    parser.add_argument(
        "--plot_output", type=str, default=None,
        help="Optional file path to save verification plot image (e.g. calib_verification.png)."
    )
    parser.add_argument(
        "--plot_pair_idx", type=int, default=None,
        help="Specific synchronized pair index (0 to N-1) to visualize in 3D. Defaults to highest corner count."
    )
    return parser.parse_args()


def find_synchronized_pairs(cam1_dir, cam2_dir):
    """
    Recursively scans two camera directories and matches frames by identical filename.

    Arguments:
        cam1_dir (str): Root directory for camera 1.
        cam2_dir (str): Root directory for camera 2.

    Returns:
        list of tuples: [(cam1_img_path, cam2_img_path, filename), ...]
    """
    valid_exts = ('*.jpg', '*.jpeg', '*.png')
    cam1_files = {}

    for ext in valid_exts:
        for p in glob.glob(os.path.join(cam1_dir, "**", ext), recursive=True):
            # Exclude anchor frames as they contain no board
            if "anchor" in p.lower():
                continue
            cam1_files[os.path.basename(p)] = p

    matched_pairs = []
    for ext in valid_exts:
        for p in glob.glob(os.path.join(cam2_dir, "**", ext), recursive=True):
            if "anchor" in p.lower():
                continue
            fname = os.path.basename(p)
            if fname in cam1_files:
                matched_pairs.append((cam1_files[fname], p, fname))

    matched_pairs.sort(key=lambda x: x[2])
    return matched_pairs


def detect_charuco_corners(img_path, detector):
    """
    Detects ChArUco corners on an image.

    Arguments:
        img_path (str): Path to image file.
        detector (cv2.aruco.CharucoDetector): ChArUco detector instance.

    Returns:
        tuple: (charuco_corners, charuco_ids) or (None, None).
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    return corners, ids


def plot_calibration_verification_3d(obj_points_list, img_points1_list, img_points2_list, 
                                     P1, P2, cam1_pos, cam2_pos, R1, R2, K1, D1, K2, D2, 
                                     t1_world, pair_idx=0, pair_fname="", save_path=None):
    """
    Plots an interactive 3D scene of camera poses and triangulated 3D points
    for a SINGLE simultaneous time period, alongside reprojection error residuals.

    Arguments:
        obj_points_list (list): List of (N, 3) true board points per pair.
        img_points1_list (list): List of (N, 1, 2) image points for camera 1.
        img_points2_list (list): List of (N, 1, 2) image points for camera 2.
        P1 (np.ndarray): 3x4 projection matrix for camera 1.
        P2 (np.ndarray): 3x4 projection matrix for camera 2.
        cam1_pos (np.ndarray): (3,) world coordinates of camera 1 center.
        cam2_pos (np.ndarray): (3,) world coordinates of camera 2 center.
        R1 (np.ndarray): 3x3 rotation matrix for camera 1.
        R2 (np.ndarray): 3x3 rotation matrix for camera 2.
        K1 (np.ndarray): 3x3 intrinsic matrix for camera 1.
        D1 (np.ndarray): Distortion vector for camera 1.
        K2 (np.ndarray): 3x3 intrinsic matrix for camera 2.
        D2 (np.ndarray): Distortion vector for camera 2.
        t1_world (np.ndarray): (3, 1) camera 1 translation vector from the anchor-pair PnP solve.
        pair_idx (int): Specific synchronized pair index to visualize.
        pair_fname (str): Filename of the frame pair being visualized.
        save_path (str, optional): Destination file path to save the generated plot.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 6))

    # --- Subplot 1: 3D Scene View for This Specific Time Period ---
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # 1. Plot Camera Centers
    ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], color='blue', s=80, label='Camera 1')
    ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], color='orange', s=80, label='Camera 2')

    # Baseline connection
    baseline_dist = np.linalg.norm(cam2_pos - cam1_pos)
    ax.plot([cam1_pos[0], cam2_pos[0]], 
            [cam1_pos[1], cam2_pos[1]], 
            [cam1_pos[2], cam2_pos[2]], 
            color='purple', linestyle='--', linewidth=1.5,
            label=f'Baseline: {baseline_dist:.3f}m')

    # Draw coordinate orientation triads for each camera (0.12m axis length)
    axis_len = 0.12
    for pos, R in [(cam1_pos, R1), (cam2_pos, R2)]:
        x_dir = R.T @ np.array([axis_len, 0, 0])
        y_dir = R.T @ np.array([0, axis_len, 0])
        z_dir = R.T @ np.array([0, 0, axis_len])
        ax.quiver(pos[0], pos[1], pos[2], x_dir[0], x_dir[1], x_dir[2], color='red', alpha=0.8)
        ax.quiver(pos[0], pos[1], pos[2], y_dir[0], y_dir[1], y_dir[2], color='green', alpha=0.8)
        ax.quiver(pos[0], pos[1], pos[2], z_dir[0], z_dir[1], z_dir[2], color='blue', alpha=0.8)

    # 2. Triangulate Points for this SAME Synchronized Timestamp
    pts1_undist = cv2.undistortPoints(img_points1_list[pair_idx], K1, D1, P=K1).reshape(-1, 2).T
    pts2_undist = cv2.undistortPoints(img_points2_list[pair_idx], K2, D2, P=K2).reshape(-1, 2).T

    pts_4d = cv2.triangulatePoints(P1, P2, pts1_undist, pts2_undist)
    triangulated_3d = (pts_4d[:3] / pts_4d[3]).T  # (N, 3) in world coordinates

    # 3. Ground Truth Board Coordinates for This Exact Timestamp
    # True board coordinates are transformed into the world coordinate system defined by P1/P2
    true_3d_local = obj_points_list[pair_idx]
    
    # Solve where the board was relative to Cam 1 during this exact timestamp
    _, rvec_board, tvec_board = cv2.solvePnP(true_3d_local, img_points1_list[pair_idx], K1, D1)
    R_b, _ = cv2.Rodrigues(rvec_board)
    
    # Map board local coordinates into world frame: X_world = R1^T * (R_b * X_board + tvec_b - t1)
    # Since P1 world frame is aligned with Cam 1 at this anchor shot:
    true_3d_world = (R1.T @ (R_b @ true_3d_local.T + tvec_board - t1_world)).T

    ax.scatter(true_3d_world[:, 0], true_3d_world[:, 1], true_3d_world[:, 2], 
               color='black', s=30, label=f'True Board Pose (Z=0 plane)')
    ax.scatter(triangulated_3d[:, 0], triangulated_3d[:, 1], triangulated_3d[:, 2], 
               color='magenta', marker='^', s=40, label='Triangulated Shared Corners')

    # Draw residual error vectors between true and triangulated positions
    for p_true, p_tri in zip(true_3d_world, triangulated_3d):
        ax.plot([p_true[0], p_tri[0]], [p_true[1], p_tri[1]], [p_true[2], p_tri[2]], 
                color='gray', linestyle=':', linewidth=1.0)

    # 3D spatial triangulation error in millimeters
    triangulation_err_mm = np.linalg.norm(triangulated_3d - true_3d_world, axis=1) * 1000.0
    mean_err_mm = np.mean(triangulation_err_mm)

    ax.set_title(f"Synchronized Timestamp: {pair_fname}\nMean 3D Triangulation Error: {mean_err_mm:.2f} mm", fontsize=11)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.legend(loc='upper right', fontsize=8)

    # --- Subplot 2: Reprojection Residual Scatter (Px) Across All Pairs ---
    ax2 = fig.add_subplot(1, 2, 2)
    all_res_x1, all_res_y1 = [], []
    all_res_x2, all_res_y2 = [], []

    for obj_pts, pts1, pts2 in zip(obj_points_list, img_points1_list, img_points2_list):
        _, r1, t1 = cv2.solvePnP(obj_pts, pts1, K1, D1)
        _, r2, t2 = cv2.solvePnP(obj_pts, pts2, K2, D2)

        proj1, _ = cv2.projectPoints(obj_pts, r1, t1, K1, D1)
        proj2, _ = cv2.projectPoints(obj_pts, r2, t2, K2, D2)

        diff1 = (proj1.reshape(-1, 2) - pts1.reshape(-1, 2))
        diff2 = (proj2.reshape(-1, 2) - pts2.reshape(-1, 2))

        all_res_x1.extend(diff1[:, 0])
        all_res_y1.extend(diff1[:, 1])
        all_res_x2.extend(diff2[:, 0])
        all_res_y2.extend(diff2[:, 1])

    ax2.scatter(all_res_x1, all_res_y1, color='blue', alpha=0.6, label='Cam 1 Residuals')
    ax2.scatter(all_res_x2, all_res_y2, color='orange', alpha=0.6, label='Cam 2 Residuals')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_title("Reprojection Residuals Across All Multi-Pose Frames", fontsize=11)
    ax2.set_xlabel("Residual X (pixels)")
    ax2.set_ylabel("Residual Y (pixels)")
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[+] Diagnostic verification plot saved to: {save_path}")

    plt.show()


def main():
    args = parse_arguments()

    # 1. Resolve intrinsics paths
    intr1_path = resolve_intrinsics_path(args.intrinsics1, args.cam1_dir)
    intr2_path = resolve_intrinsics_path(args.intrinsics2, args.cam2_dir)

    print("=" * 65)
    print(" MULTI-VIEW CHARUCO STEREO EXTRINSIC SOLVER")
    print(f" Camera 1 Dir        : {args.cam1_dir}")
    print(f" Camera 2 Dir        : {args.cam2_dir}")
    print(f" Camera 1 Intrinsics : {intr1_path}")
    print(f" Camera 2 Intrinsics : {intr2_path}")
    print("=" * 65)

    # 2. Ingest camera intrinsics
    K1, D1, model1, w1, h1 = load_intrinsics(intr1_path)
    K2, D2, model2, w2, h2 = load_intrinsics(intr2_path)

    # 3. Discover synchronized image pairs
    pairs = find_synchronized_pairs(args.cam1_dir, args.cam2_dir)
    print(f"[+] Discovered {len(pairs)} synchronized frame pairs across subfolders.")

    if not pairs:
        raise FileNotFoundError("[-] No matching image filenames found between camera directories.")

    # 4. Initialize board and detector
    board, detector = build_charuco_detector(
        args.squares_x, args.squares_y, args.square_len, args.marker_len, args.dictionary
    )
    all_obj_points = board.getChessboardCorners()

    # 5. Extract shared corner correspondences across all valid pairs
    obj_points_list = []  # 3D points in board frame (N, 3)
    img_points1_list = [] # 2D points in Cam 1 (N, 1, 2)
    img_points2_list = [] # 2D points in Cam 2 (N, 1, 2)
    valid_pair_records = []

    for img1_path, img2_path, fname in pairs:
        corners1, ids1 = detect_charuco_corners(img1_path, detector)
        corners2, ids2 = detect_charuco_corners(img2_path, detector)

        if corners1 is None or corners2 is None or ids1 is None or ids2 is None:
            continue

        ids1_flat = ids1.flatten()
        ids2_flat = ids2.flatten()
        common_ids, idx1, idx2 = np.intersect1d(ids1_flat, ids2_flat, return_indices=True)

        if len(common_ids) < args.min_corners:
            continue

        shared_pts_3d = all_obj_points[common_ids].astype(np.float32)
        shared_pts_2d_1 = corners1.reshape(-1, 2)[idx1].astype(np.float32).reshape(-1, 1, 2)
        shared_pts_2d_2 = corners2.reshape(-1, 2)[idx2].astype(np.float32).reshape(-1, 1, 2)

        obj_points_list.append(shared_pts_3d)
        img_points1_list.append(shared_pts_2d_1)
        img_points2_list.append(shared_pts_2d_2)

        valid_pair_records.append({
            "pair_idx": len(valid_pair_records),
            "filename": fname,
            "img1_path": img1_path,
            "img2_path": img2_path,
            "shared_corners": len(common_ids),
            "common_ids": common_ids.tolist()
        })
        print(f"  [✓] Pair #{len(valid_pair_records)-1}: {fname:<30} -> {len(common_ids)} shared corners")

    print(f"\n[+] Valid pairs retained for stereo calibration: {len(valid_pair_records)} / {len(pairs)}")

    if len(valid_pair_records) < args.min_pairs:
        raise RuntimeError(f"[-] Retained pairs ({len(valid_pair_records)}) below --min_pairs ({args.min_pairs}).")

    # 6. Multi-View Stereo Calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    rmse_stereo, _, _, _, _, R_rel, t_rel, E, F = cv2.stereoCalibrate(
        obj_points_list,
        img_points1_list,
        img_points2_list,
        K1,
        D1,
        K2,
        D2,
        (w1, h1),
        flags=flags,
        criteria=criteria
    )

    print("=" * 65)
    print(f"[+] Stereo Optimization Converged.")
    print(f"    Global Stereo Reprojection RMSE: {rmse_stereo:.4f} px")
    print(f"    Baseline Distance (Norm): {float(np.linalg.norm(t_rel)):.4f} meters")
    print("=" * 65)

    # 7. World Pose Anchor Selection
    # Choose pair index to define world origin
    if args.plot_pair_idx is not None and 0 <= args.plot_pair_idx < len(valid_pair_records):
        target_pair_idx = args.plot_pair_idx
    else:
        target_pair_idx = int(np.argmax([r["shared_corners"] for r in valid_pair_records]))

    anchor_record = valid_pair_records[target_pair_idx]
    anchor_obj_pts = obj_points_list[target_pair_idx]
    anchor_img_pts1 = img_points1_list[target_pair_idx]

    _, rvec1, tvec1 = cv2.solvePnP(
        anchor_obj_pts,
        anchor_img_pts1,
        K1,
        D1,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    R1_world, _ = cv2.Rodrigues(rvec1)
    t1_world = tvec1

    # Camera 2 world pose derived via the stereo transformation
    R2_world = R_rel @ R1_world
    t2_world = R_rel @ t1_world + t_rel

    cam1_pos_world = (-R1_world.T @ t1_world).flatten()
    cam2_pos_world = (-R2_world.T @ t2_world).flatten()

    P1 = K1 @ np.hstack((R1_world, t1_world))
    P2 = K2 @ np.hstack((R2_world, t2_world))

    # 8. Export structured results
    payload = {
        "calibration_metadata": {
            "total_pairs_discovered": len(pairs),
            "valid_pairs_optimized": len(valid_pair_records),
            "global_stereo_rmse_px": round(float(rmse_stereo), 4),
            "baseline_meters": round(float(np.linalg.norm(t_rel)), 4),
            "anchor_pair_filename": anchor_record["filename"],
            "board_parameters": {
                "squares_x": args.squares_x,
                "squares_y": args.squares_y,
                "square_len_m": args.square_len,
                "marker_len_m": args.marker_len,
                "dictionary": args.dictionary
            }
        },
        "relative_extrinsics_cam1_to_cam2": {
            "rotation_matrix": R_rel.tolist(),
            "translation_vector": t_rel.flatten().tolist(),
            "essential_matrix": E.tolist(),
            "fundamental_matrix": F.tolist()
        },
        "cameras": {
            args.cam1_id: {
                "intrinsics_file": intr1_path,
                "camera_model": model1,
                "rotation_matrix": R1_world.tolist(),
                "translation_vector": t1_world.flatten().tolist(),
                "projection_matrix": P1.tolist(),
                "camera_position_world": cam1_pos_world.tolist()
            },
            args.cam2_id: {
                "intrinsics_file": intr2_path,
                "camera_model": model2,
                "rotation_matrix": R2_world.tolist(),
                "translation_vector": t2_world.flatten().tolist(),
                "projection_matrix": P2.tolist(),
                "camera_position_world": cam2_pos_world.tolist()
            }
        },
        "evaluated_pairs": valid_pair_records
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(payload, f, indent=4)

    print(f"[+] Output JSON saved to: {args.output_json}")
    print(f"    {args.cam1_id} Position: X={cam1_pos_world[0]:.3f}m, Y={cam1_pos_world[1]:.3f}m, Z={cam1_pos_world[2]:.3f}m")
    print(f"    {args.cam2_id} Position: X={cam2_pos_world[0]:.3f}m, Y={cam2_pos_world[1]:.3f}m, Z={cam2_pos_world[2]:.3f}m")

    # 9. Visual 3D Verification Plot
    if args.plot or args.plot_output:
        plot_calibration_verification_3d(
            obj_points_list,
            img_points1_list,
            img_points2_list,
            P1,
            P2,
            cam1_pos_world,
            cam2_pos_world,
            R1_world,
            R2_world,
            K1,
            D1,
            K2,
            D2,
            t1_world=t1_world,
            pair_idx=target_pair_idx,
            pair_fname=anchor_record["filename"],
            save_path=args.plot_output
        )


if __name__ == "__main__":
    main()