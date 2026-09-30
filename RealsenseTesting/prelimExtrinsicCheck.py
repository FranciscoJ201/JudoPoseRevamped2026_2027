import glob
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# --- CONFIGURATION ---
CHECKBOARD_SIZE = (
    9,
    6,
)  # Internal corners of your checkerboard (columns - 1, rows - 1)
SQUARE_SIZE = 0.025  # Size of a square in meters (e.g., 25mm)
LEFT_IMAGES_PATH = "left_cam/*.png"
RIGHT_IMAGES_PATH = "right_cam/*.png"

# 1. Pull Intrinsics directly from RealSense Pipeline (Offline device read)
print("Pulling factory intrinsics from RealSense device...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# Construct OpenCV intrinsic matrix K
K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1],
], dtype=np.float32)

dist_coeffs = np.array(
    intrinsics.coeffs, dtype=np.float32
)  # Distortion parameters
pipeline.stop()
print("Successfully retrieved camera intrinsics!")

# 2. Prepare Object Points (3D coordinates of board grid)
objp = np.zeros(
    (CHECKBOARD_SIZE[0] * CHECKBOARD_SIZE[1], 3), np.float32
)
objp[:, :2] = np.mgrid[0 : CHECKBOARD_SIZE[0], 0 : CHECKBOARD_SIZE[1]].T.reshape(
    -1, 2
)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left camera view
imgpoints_right = []  # 2D points in right camera view

left_images = sorted(glob.glob(LEFT_IMAGES_PATH))
right_images = sorted(glob.glob(RIGHT_IMAGES_PATH))

print(
    f"Found {len(left_images)} left images and {len(right_images)} right"
    " images."
)

for img_l_path, img_r_path in zip(left_images, right_images):
  img_l = cv2.imread(img_l_path)
  img_r = cv2.imread(img_r_path)
  gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
  gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

  ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKBOARD_SIZE, None)
  ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKBOARD_SIZE, None)

  if ret_l and ret_r:
    objpoints.append(objp)
    # Refine corner locations for sub-pixel accuracy
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
    cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

    imgpoints_left.append(corners_l)
    imgpoints_right.append(corners_r)

print("Running Stereo Calibration between the two RealSense cameras...")
# 3. Stereo Calibration to find Relative Extrinsics (R, T) between Cam 1 and Cam 2
flags = cv2.CALIB_FIX_INTRINSIC  # Keep factory intrinsics fixed, solve only R, T
ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K,
    dist_coeffs,
    K,
    dist_coeffs,
    gray_l.shape[::-1],
    criteria=criteria,
    flags=flags,
)

print(f"Stereo Calibration Complete! RMS Error: {ret:.4f}")
print(f"Rotation Matrix R:\n{R}")
print(f"Translation Vector T (meters):\n{T}")

# 4. Build Projection Matrices for 3D Triangulation Test
# Camera 1 at origin
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
# Camera 2 transformed by stereo R and T
P2 = K @ np.hstack((R, T))

# Take the first detected board corner from both images to test triangulation
pt1 = imgpoints_left[0][0][0]
pt2 = imgpoints_right[0][0][0]

# Triangulate 3D point using OpenCV
pt4d = cv2.triangulatePoints(
    P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1)
)
point_3d = (pt4d[:3] / pt4d[3]).ravel()
print(f"Test Triangulated 3D Point (Board Corner): {point_3d} meters")

# 5. Visualize in 3D Space via Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Offline Stereo Rig 3D Test", width=800, height=600)

# Create Coordinate frames representing both cameras
cam1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
cam2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
# Apply transformation to Camera 2 frame based on stereo calibration output
cam2_frame.transform(
    np.vstack((np.hstack((R, T)), [0, 0, 0, 1]))
)  # Homogeneous matrix

# Create a sphere at the triangulated corner point
point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
point_sphere.paint_uniform_color([0.0, 1.0, 0.0])
point_sphere.translate(point_3d)

vis.add_geometry(cam1_frame)
vis.add_geometry(cam2_frame)
vis.add_geometry(point_sphere)

print("Displaying 3D Open3D verification window. Close window to exit.")
vis.run()
vis.destroy_window()