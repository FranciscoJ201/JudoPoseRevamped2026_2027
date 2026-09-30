import argparse
import glob
import json
import os
import sys

# Headless backend for HPC cluster environments without X11
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import cv2


def parse_arguments():
    """
    Parses CLI arguments for camera intrinsic calibration.
    """
    parser = argparse.ArgumentParser(
        description="Compute GoPro intrinsic calibration parameters from ChArUco targets for HPC deployment."
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, default="gopro_0_intrinsic_frames",
        help="Path to folder containing extracted calibration frames (.jpg, .png)."
    )
    parser.add_argument(
        "--output_json", "-o", type=str, default="gopro_intrinsics.json",
        help="Output path for intrinsic parameter JSON."
    )
    parser.add_argument(
        "--model", "-m", type=str, choices=["rational", "fisheye", "standard"], default="fisheye",
        help="Distortion model: 'rational' (CALIB_RATIONAL_MODEL), 'fisheye' (cv2.fisheye), or 'standard'."
    )
    parser.add_argument(
        "--squares_x", type=int, default=5,
        help="Number of chessboard squares along X dimension."
    )
    parser.add_argument(
        "--squares_y", type=int, default=8,
        help="Number of chessboard squares along Y dimension."
    )
    parser.add_argument(
        "--square_len", type=float, default=0.04,
        help="Physical chessboard square side length in meters."
    )
    parser.add_argument(
        "--marker_len", type=float, default=0.03,
        help="Physical ArUco marker side length in meters."
    )
    parser.add_argument(
        "--min_corners", type=int, default=12,
        help="Minimum valid ChArUco corners required to retain a frame."
    )
    parser.add_argument(
        "--outlier_thresh", type=float, default=1.0,
        help="Per-view RMSE threshold (pixels) for iterative outlier frame pruning."
    )
    parser.add_argument(
        "--save_plots", type=str, default='IntrinsicImages',
        help="File path to save the 3-panel calibration diagnostic dashboard image."
    )
    return parser.parse_args()


def plot_calibration_graphs(objpoints, rvecs, tvecs, raw_imgpoints, imgpoints, image_size, save_path):
    """
    Plots a 3-panel diagnostic dashboard: 3D board poses, raw corner density, and filtered corner density.
    Saves figure directly to disk (headless mode).
    
    Arguments:
        objpoints (list): 3D coordinates in target frame.
        rvecs (list): Rotation vectors per view.
        tvecs (list): Translation vectors per view.
        raw_imgpoints (list): Initial corner detections.
        imgpoints (list): Filtered corner detections used in final solver pass.
        image_size (tuple): Image resolution (width, height).
        save_path (str): Destination path for plot figure.
    """
    fig = plt.figure(figsize=(24, 8))

    # --- Panel 1: 3D Target Poses Relative to Camera Coordinate System ---
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(0, 0, 0, c='black', marker='^', s=200, label='Camera Lens')

    for i in range(len(objpoints)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i]
        pts = objpoints[i].reshape(-1, 3)
        transformed_pts = (R @ pts.T) + t
        transformed_pts = transformed_pts.T

        label_name = f'View {i+1}' if i < 15 else ""
        ax1.scatter(
            transformed_pts[:, 0], transformed_pts[:, 1], transformed_pts[:, 2],
            alpha=0.6, s=15, label=label_name
        )

    ax1.set_xlabel('X Axis (Meters)')
    ax1.set_ylabel('Y Axis (Meters)')
    ax1.set_zlabel('Z Axis (Meters Depth)')
    ax1.set_title('3D Target Orientation Distribution')
    handles, labels = ax1.get_legend_handles_labels()
    if len(handles) > 15:
        ax1.legend(handles[:15], labels[:15])
    else:
        ax1.legend()

    # --- Helper Function for Density Heatmaps ---
    def draw_heatmap(ax, points, title):
        all_x, all_y = [], []
        for corners in points:
            for corner in corners:
                all_x.append(corner[0][0])
                all_y.append(corner[0][1])

        width, height = image_size
        h = ax.hist2d(
            all_x, all_y,
            bins=[max(10, int(width / 30)), max(10, int(height / 30))],
            range=[[0, width], [0, height]],
            cmap='inferno',
            norm=LogNorm()
        )
        fig.colorbar(h[3], ax=ax, label='Logarithmic Detection Density')
        ax.invert_yaxis()
        ax.set_xlabel('Image Width (Pixels)')
        ax.set_ylabel('Image Height (Pixels)')
        ax.set_title(title)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    # --- Panel 2: Initial Detection Heatmap ---
    ax2 = fig.add_subplot(132)
    draw_heatmap(ax2, raw_imgpoints, f'Input Frame Coverage ({len(raw_imgpoints)} Views)')

    # --- Panel 3: Post-Pruning Detection Heatmap ---
    ax3 = fig.add_subplot(133)
    draw_heatmap(ax3, imgpoints, f'Optimized Solver Coverage ({len(imgpoints)} Views)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[+] Diagnostic calibration dashboard saved to: {save_path}")


def compute_per_view_errors(objpoints, imgpoints, rvecs, tvecs, K, D, model):
    """
    Computes Euclidean reprojection root-mean-square error (RMSE) for each individual calibration image.
    
    Arguments:
        objpoints (list): List of (N_i, 3) arrays of 3D object points.
        imgpoints (list): List of (N_i, 1, 2) arrays of observed 2D image coordinates.
        rvecs (list): Rotation vectors for each image view.
        tvecs (list): Translation vectors for each image view.
        K (np.ndarray): 3x3 intrinsic camera matrix.
        D (np.ndarray): Distortion coefficients array.
        model (str): Camera projection model ('fisheye', 'rational', 'standard').

    Returns:
        list: Float RMSE error for each corresponding image view.
    """
    errors = []
    for i in range(len(objpoints)):
        pts3d = objpoints[i]
        pts2d_obs = imgpoints[i]

        if model == "fisheye":
            # Reshape for cv2.fisheye.projectPoints
            pts3d_in = pts3d.reshape(-1, 1, 3).astype(np.float64)
            rvec_in = rvecs[i].reshape(3, 1).astype(np.float64)
            tvec_in = tvecs[i].reshape(3, 1).astype(np.float64)
            pts2d_proj, _ = cv2.fisheye.projectPoints(pts3d_in, rvec_in, tvec_in, K, D)
        else:
            pts2d_proj, _ = cv2.projectPoints(pts3d, rvecs[i], tvecs[i], K, D)

        diff = pts2d_obs.reshape(-1, 2) - pts2d_proj.reshape(-1, 2)
        view_rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        errors.append(float(view_rmse))

    return errors


def solve_intrinsics(objpoints, imgpoints, image_size, model):
    """
    Executes OpenCV calibration routines according to the selected mathematical model.
    
    Arguments:
        objpoints (list): List of 3D object points.
        imgpoints (list): List of 2D corner image points.
        image_size (tuple): Sensor resolution (width, height).
        model (str): One of 'rational', 'fisheye', or 'standard'.

    Returns:
        tuple: (rms, K, D, rvecs, tvecs)
    """
    if model == "fisheye":
        # Format object points and image points for fisheye module: (N, 1, 3) and (N, 1, 2)
        fish_obj = [p.reshape(-1, 1, 3).astype(np.float64) for p in objpoints]
        fish_img = [p.reshape(-1, 1, 2).astype(np.float64) for p in imgpoints]

        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objectPoints=fish_obj,
            imagePoints=fish_img,
            image_size=image_size,
            K=K,
            D=D,
            rvecs=None,
            tvecs=None,
            flags=flags,
            criteria=criteria
        )
    elif model == "rational":
        flags = cv2.CALIB_RATIONAL_MODEL
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags
        )
    else:  # Standard pinhole
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

    return rms, K, D, rvecs, tvecs


def calibrate(args):
    """
    Orchestrates target corner detection, solver convergence, outlier pruning, and export.
    
    Arguments:
        args (argparse.Namespace): Command-line configuration parameters.
    """
    print("=" * 60)
    print(f" GOPRO INTRINSIC CALIBRATION ENGINE")
    print(f" Image Directory : {args.input_dir}")
    print(f" Distortion Model: {args.model.upper()}")
    print("=" * 60)

    # 1. Initialize ChArUco Board Definition
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_len,
        args.marker_len,
        dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)
    board_3d_points = board.getChessboardCorners()

    # 2. Ingest Calibration Image Files
    image_patterns = [os.path.join(args.input_dir, ext) for ext in ("*.jpg", "*.jpeg", "*.png")]
    image_files = []
    for pat in image_patterns:
        image_files.extend(glob.glob(pat))
    image_files = sorted(image_files)

    if not image_files:
        raise FileNotFoundError(f"[-] No valid images found in: {args.input_dir}")

    print(f"[+] Processing {len(image_files)} calibration frames...")

    all_objpoints = []
    all_imgpoints = []
    raw_imgpoints = []
    retained_files = []
    image_size = None

    for fpath in image_files:
        img = cv2.imread(fpath)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_corners is not None and charuco_ids is not None:
            raw_imgpoints.append(charuco_corners)
            if len(charuco_corners) >= args.min_corners:
                ids = charuco_ids.flatten()
                pts3d = board_3d_points[ids]

                all_objpoints.append(pts3d.astype(np.float32))
                all_imgpoints.append(charuco_corners.astype(np.float32))
                retained_files.append(fpath)

    print(f"[+] Total frames meeting corner threshold (>={args.min_corners}): {len(all_imgpoints)}/{len(image_files)}")

    if len(all_imgpoints) < 8:
        raise RuntimeError("[-] Insufficient valid frames to constrain nonlinear distortion parameters. Exiting.")

    # 3. Initial Optimization Pass
    print(f"[+] Running initial {args.model.upper()} solver optimization pass...")
    rms, K, D, rvecs, tvecs = solve_intrinsics(all_objpoints, all_imgpoints, image_size, args.model)
    print(f"    Initial Optimization Global RMSE: {rms:.4f} pixels")

    # 4. Outlier Detection and Second Pass Optimization
    per_view_errors = compute_per_view_errors(all_objpoints, all_imgpoints, rvecs, tvecs, K, D, args.model)

    filtered_obj = []
    filtered_img = []
    filtered_rvecs = []
    filtered_tvecs = []
    pruned_count = 0

    for idx, err in enumerate(per_view_errors):
        if err <= args.outlier_thresh:
            filtered_obj.append(all_objpoints[idx])
            filtered_img.append(all_imgpoints[idx])
        else:
            pruned_count += 1
            print(f"    [Pruned] View {os.path.basename(retained_files[idx])} rejected (RMSE: {err:.3f} px > {args.outlier_thresh} px)")

    if pruned_count > 0 and len(filtered_obj) >= 8:
        print(f"[+] Re-optimizing parameters after pruning {pruned_count} high-residual views...")
        rms, K, D, rvecs, tvecs = solve_intrinsics(filtered_obj, filtered_img, image_size, args.model)
        final_obj = filtered_obj
        final_img = filtered_img
        print(f"    Final Refined Global RMSE: {rms:.4f} pixels")
    else:
        final_obj = all_objpoints
        final_img = all_imgpoints
        print(f"    No views pruned. Final Global RMSE: {rms:.4f} pixels")

    # 5. Export Calibrated Matrices to JSON
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    dist_flat = D.flatten().tolist()

    output_payload = {
        "camera_model": args.model,
        "width": image_size[0],
        "height": image_size[1],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist_flat,
        "global_reprojection_rmse": float(rms),
        "total_views_evaluated": len(image_files),
        "views_retained": len(final_obj)
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output_payload, f, indent=4)

    print(f"[+] Camera parameters written successfully to: {args.output_json}")
    print(f"    Principal Point (cx, cy): ({cx:.2f}, {cy:.2f})")
    print(f"    Focal Length (fx, fy)   : ({fx:.2f}, {fy:.2f})")

    # 6. Generate Diagnostic Visual Dashboard
    if args.save_plots:
        plot_calibration_graphs(
            final_obj, rvecs, tvecs, raw_imgpoints, final_img, image_size, args.save_plots
        )


if __name__ == "__main__":
    cli_args = parse_arguments()
    calibrate(cli_args)