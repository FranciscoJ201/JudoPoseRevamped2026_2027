import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def plot_floor_diagnostics(pcd_points, plane_model, inlier_indices):
  """Generates a diagnostic matplotlib dashboard for the extracted floor plane."""
  a, b, c, d = plane_model
  normal = np.array([a, b, c])
  points = np.array(pcd_points)

  # Calculate camera tilt angle relative to gravity/floor (assuming floor normal is roughly up)
  # Cos(theta) = dot(normal, camera_down_vector)
  camera_down = np.array([0, 1, 0])  # OpenCV Y-axis points down
  tilt_rad = np.arccos(
      np.clip(
          np.dot(normal, camera_down) / (np.linalg.norm(normal)), -1.0, 1.0
      )
  )
  tilt_deg = np.degrees(tilt_rad)

  # Distances of points from the fitted plane
  distances = (
      np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)
  ) * 100  # in cm

  fig = plt.figure(figsize=(14, 8))
  fig.suptitle(
      f"Floor Plane Calibration Diagnostics\nHeight: {abs(d):.3f}m | Camera"
      f" Pitch Tilt: {tilt_deg:.1f}°",
      fontsize=14,
      fontweight="bold",
  )

  # Subplot 1: 3D Point Cloud Scatter
  ax1 = fig.add_subplot(2, 2, 1, projection="3d")
  inliers_mask = np.zeros(len(points), dtype=bool)
  inliers_mask[inlier_indices] = True

  ax1.scatter(
      points[inliers_mask, 0],
      points[inliers_mask, 2],
      -points[inliers_mask, 1],
      c="g",
      s=2,
      alpha=0.6,
      label="Inlier Floor Points",
  )
  ax1.scatter(
      points[~inliers_mask, 0],
      points[~inliers_mask, 2],
      -points[~inliers_mask, 1],
      c="r",
      s=2,
      alpha=0.3,
      label="Outliers",
  )
  ax1.set_title("3D Extracted Floor Points (Camera Frame)")
  ax1.set_xlabel("X (m)")
  ax1.set_ylabel("Z / Depth (m)")
  ax1.set_zlabel("Y / Height (m)")
  ax1.legend(loc="upper right")

  # Subplot 2: Residual Error Histogram (Noise level of sensor on floor)
  ax2 = fig.add_subplot(2, 2, 2)
  ax2.hist(distances[inliers_mask], bins=30, color="green", alpha=0.7)
  ax2.set_title("Floor Surface Noise / Residuals (cm)")
  ax2.set_xlabel("Distance to Fitted Plane (cm)")
  ax2.set_ylabel("Point Count")
  ax2.axvline(
      np.mean(distances[inliers_mask]),
      color="black",
      linestyle="dashed",
      linewidth=1.5,
      label=f"Mean Error: {np.mean(distances[inliers_mask]):.2f}cm",
  )
  ax2.legend()

  # Subplot 3: Depth vs Elevation Profile
  ax3 = fig.add_subplot(2, 2, 3)
  ax3.scatter(
      points[inliers_mask, 2],
      points[inliers_mask, 1],
      c="g",
      s=2,
      alpha=0.5,
  )
  ax3.set_title("Side Profile: Depth (Z) vs Vertical Position (Y)")
  ax3.set_xlabel("Depth Z (m)")
  ax3.set_ylabel("Camera Y Axis (m)")
  ax3.invert_yaxis()  # Image Y goes down
  ax3.grid(True)

  # Subplot 4: Vector Parameters Summary Card
  ax4 = fig.add_subplot(2, 2, 4)
  ax4.axis("off")
  text_str = (
      f"Plane Model Parameters:\n"
      f"-----------------------\n"
      f"A (X Normal): {a:.4f}\n"
      f"B (Y Normal): {b:.4f}\n"
      f"C (Z Normal): {c:.4f}\n"
      f"D (Offset)  : {d:.4f}\n\n"
      f"Extracted Metrics:\n"
      f"-----------------------\n"
      f"True Camera Height (d): {abs(d):.3f} meters\n"
      f"Normal Vector (n)    : [{a:.2f}, {b:.2f}, {c:.2f}]\n"
      f"Floor Inlier Ratio   : {len(inlier_indices)/len(points)*100:.1f}%\n"
  )
  ax4.text(
      0.1,
      0.2,
      text_str,
      fontsize=12,
      family="monospace",
      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
  )

  plt.tight_layout()
  plt.show()


# 1. Initialize RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# Initialize Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="Open3D RealSense Floor Stream", width=640, height=480
)
pcd_geo = o3d.geometry.PointCloud()
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
vis.add_geometry(coordinate_frame)

print(
    "Point the RealSense camera at the floor. Press 'c' to capture & plot"
    " graphs, or 'q' to quit."
)

try:
  while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
      continue

    color_image = np.asanyarray(color_frame.get_data())

    # Sampling box setup
    h, w, _ = color_image.shape
    box_top, box_bottom = int(h * 0.65), int(h * 0.9)
    box_left, box_right = int(w * 0.25), int(w * 0.75)

    # Visual overlay on OpenCV stream
    cv2.rectangle(
        color_image,
        (box_left, box_top),
        (box_right, box_bottom),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        color_image,
        "Sampling Zone (Floor Target)",
        (box_left, box_top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        color_image,
        "Press 'c' to compute plane & generate graphs",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow("RealSense Feed", color_image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
      break
    elif key == ord("c"):
      depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
      pcd_points = []

      # Deproject bounding box region
      for v in range(box_top, box_bottom, 4):
        for u in range(box_left, box_right, 4):
          dist = depth_frame.get_distance(u, v)
          if 0.2 < dist < 4.0:
            point = rs.rs2_deproject_pixel_to_point(
                depth_intrinsics, [u, v], dist
            )
            pcd_points.append(point)

      if len(pcd_points) < 50:
        print("Not enough valid depth points sampled. Adjust position.")
        continue

      # Create Open3D PointCloud
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(np.array(pcd_points))

      # RANSAC Plane Fit
      plane_model, inliers = pcd.segment_plane(
          distance_threshold=0.015, init_n=3, num_iterations=1000
      )
      [a, b, c, d] = plane_model

      # Draw Floor Normal Vector Arrow in Open3D
      center_point = np.mean(np.array(pcd_points)[inliers], axis=0)
      normal_end = center_point + np.array([a, b, c]) * 0.5  # 50cm arrow length

      lines = [[0, 1]]
      colors = [[1, 0, 0]]  # Red arrow for normal vector
      line_set = o3d.geometry.LineSet(
          points=o3d.utility.Vector3dVector([center_point, normal_end]),
          lines=o3d.utility.Vector2iVector(lines),
      )
      line_set.colors = o3d.utility.Vector3dVector(colors)

      # Update 3D Open3D Window
      pcd_inliers = pcd.select_by_index(inliers)
      pcd_inliers.paint_uniform_color([0.0, 1.0, 0.0])  # Green = Floor
      vis.clear_geometries()
      vis.add_geometry(pcd_inliers)
      vis.add_geometry(line_set)
      vis.add_geometry(coordinate_frame)
      vis.poll_events()
      vis.update_renderer()

      # Launch Matplotlib Visual Graph Dashboard
      plot_floor_diagnostics(pcd_points, plane_model, inliers)

finally:
  pipeline.stop()
  cv2.destroyAllWindows()
  vis.destroy_window()