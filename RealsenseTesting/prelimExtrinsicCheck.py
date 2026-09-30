import glob
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# --- CONFIGURATION ---
LEFT_IMAGES_PATH = "calib_capture/left_cam/*.jpg"
RIGHT_IMAGES_PATH = "calib_capture/right_cam/*.jpg"

# Setup ChArUco Board Parameters (4x4 dictionary, e.g., 5x7 squares layout)
# Adjust squaresX and squaresY to match your physical board dimensions
ARUCO_DICT = cv2.aruco.DICT_4X4_50
squaresX = 5
squaresY = 7
squareLength = 0.04  # Size of checker square in meters (e.g., 40mm)
markerLength = 0.03  # Size of ArUco marker in meters (e.g., 30mm)

# Initialize ArUco dictionary and ChArUco board
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (squaresX, squaresY), squareLength, markerLength, aruco_dict
)
detector = cv2.aruco.CharucoDetector(board)

# 1. Pull Intrinsics directly from RealSense Pipeline (Offline device read)
print("Pulling factory intrinsics from RealSense device...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# Construct OpenCV intrinsic matrix K
K = np.array(
    [
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ],
    dtype=np.float32,
)
dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)
pipeline.stop()
print("Successfully retrieved camera intrinsics!")

objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left camera view
imgpoints_right = []  # 2D points in right camera view

left_images = sorted(glob.glob(LEFT_IMAGES_PATH))
right_images = sorted(glob.glob(RIGHT_IMAGES_PATH))

print(
    f"Found {len(left_images)} left images and {len(right_images)} right"
    " images."
)

image_shape = None
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for img_l_path, img_r_path in zip(left_images, right_images):
  img_l = cv2.imread(img_l_path)
  img_r = cv2.imread(img_r_path)
  gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
  gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

  # Detect ChArUco corners for both images
  charuco_corners_l, charuco_ids_l, _, _ = detector.detectBoard(gray_l)
  charuco_corners_r, charuco_ids_r, _, _ = detector.detectBoard(gray_r)

  # Check if both cameras successfully detected enough corners
  if (
      charuco_corners_l is not None
      and charuco_corners_r is not None
      and len(charuco_corners_l) > 4
      and len(charuco_corners_r) > 4
  ):

    # Match common corners found in both views
    # For robust stereo calibration, we match based on IDs
    common_ids, idx_l, idx_r = np.intersect1d(
        charuco_ids_l, charuco_ids_r, return_indices=True
    )

    if len(common_ids) > 4:
      matched_corners_l = charuco_corners_l[idx_l]
      matched_corners_r = charuco_corners_r[idx_r]

      # Get corresponding 3D object points from the board layout model
      objp = board.getChessboardCorners()[common_ids].reshape(-1, 3)

      objpoints.append(objp)
      imgpoints_left.append(matched_corners_l)
      imgpoints_right.append(matched_corners_r)

      image_shape = gray_l.shape[::-1]

if len(objpoints) == 0:
  raise ValueError(
      "No valid ChArUco corners found in any image pair! Check your board"
      " dimensions (squaresX/squaresY)."
  )

print("Running Stereo Calibration between the two RealSense cameras...")
# 2. Stereo Calibration using ChArUco matched points
flags = cv2.CALIB_FIX_INTRINSIC
ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K,
    dist_coeffs,
    K,
    dist_coeffs,
    image_shape,
    criteria=criteria,
    flags=flags,
)

print(f"Stereo Calibration Complete! RMS Error: {ret:.4f}")
print(f"Rotation Matrix R:\n{R}")
print(f"Translation Vector T (meters):\n{T}")

# 3. Build Projection Matrices for 3D Triangulation Test
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, T))

# Take the first matched ChArUco corner to test triangulation
pt1 = imgpoints_left[0][0][0]
pt2 = imgpoints_right[0][0][0]

pt4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
point_3d = (pt4d[:3] / pt4d[3]).ravel()
print(f"Test Triangulated 3D Point (ChArUco Corner): {point_3d} meters")

# 4. Visualize in 3D Space via Open3D
vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="Offline ChArUco Stereo Rig 3D Test", width=800, height=600
)

cam1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
cam2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
cam2_frame.transform(np.vstack((np.hstack((R, T)), [0, 0, 0, 1])))

point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
point_sphere.paint_uniform_color([0.0, 1.0, 0.0])
point_sphere.translate(point_3d)

vis.add_geometry(cam1_frame)
vis.add_geometry(cam2_frame)
vis.add_geometry(point_sphere)

print("Displaying 3D Open3D verification window. Close window to exit.")
vis.run()
vis.destroy_window()