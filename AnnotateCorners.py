import json
import os
import cv2
import numpy as np

# ─────────────────────────────────────────────
#  FIELD LAPTOP CONFIGURATION
# ─────────────────────────────────────────────

# List all anchor images (one clean frame per camera showing the full tatami)
ANCHOR_IMAGES = [
    "anchor_realsense_cam_0.jpg",
    "anchor_realsense_cam_1.jpg",
    "anchor_gopro_0.jpg",
    "anchor_gopro_1.jpg",
]

# Physical square dimension of the inner combat tatami area (in meters)
MAT_SIZE_METERS = 8.0

# Destination path for the saved corner annotations
OUTPUT_JSON = "mat_corners.json"

# ─────────────────────────────────────────────
#  GLOBAL STATE FOR MOUSE CALLBACK
# ─────────────────────────────────────────────

clicked_points = []
current_display_img = None


def mouse_callback(event, x, y, flags, param):
    """
    Captures mouse clicks on the tatami corners in strict order:
    1: Top-Left, 2: Top-Right, 3: Bottom-Right, 4: Bottom-Left
    """
    global clicked_points, current_display_img
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([float(x), float(y)])
        current_display_img = param.copy()

        for idx, pt in enumerate(clicked_points):
            cv2.circle(current_display_img, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
            cv2.putText(
                current_display_img, str(idx + 1),
                (int(pt[0]) + 10, int(pt[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            if idx > 0:
                cv2.line(
                    current_display_img,
                    (int(clicked_points[idx - 1][0]), int(clicked_points[idx - 1][1])),
                    (int(pt[0]), int(pt[1])), (0, 255, 255), 2
                )

        if len(clicked_points) == 4:
            cv2.line(
                current_display_img,
                (int(clicked_points[3][0]), int(clicked_points[3][1])),
                (int(clicked_points[0][0]), int(clicked_points[0][1])),
                (0, 255, 255), 2
            )
            print("  -> 4 corners registered. Press SPACE or ENTER to accept, 'r' to reset.")

        cv2.imshow("Tatami Annotation", current_display_img)


def annotate_camera(image_path):
    """
    Loads one image and prompts the operator to click the 4 tatami corners.
    
    Arguments:
        image_path (str): Path to image file.

    Returns:
        list: 4-element list of [x, y] coordinates.
    """
    global clicked_points, current_display_img
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[-] Could not load image: {image_path}")

    window_name = "Tatami Annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        clicked_points = []
        current_display_img = img.copy()
        cv2.imshow(window_name, current_display_img)
        cv2.setMouseCallback(window_name, mouse_callback, img)

        print(f"\n[+] Annotating: {image_path}")
        print("    Click order: 1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")
        print("    Keys: [ENTER/SPACE] Accept | [r] Reset | [q] Cancel")

        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32) and len(clicked_points) == 4:  # Enter or Space
            break
        elif key == ord('r'):
            print("    [Reset] Clearing clicked points for this frame.")
            continue
        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Annotation cancelled by operator.")

    cv2.destroyAllWindows()
    return clicked_points


def main():
    half = MAT_SIZE_METERS / 2.0

    annotations = {
        "metadata": {
            "mat_size_meters": MAT_SIZE_METERS,
            "world_points": {
                "top_left": [-half, half, 0.0],
                "top_right": [half, half, 0.0],
                "bottom_right": [half, -half, 0.0],
                "bottom_left": [-half, -half, 0.0]
            }
        },
        "cameras": {}
    }

    print("=" * 60)
    print("  FIELD TATAMI CORNER ANNOTATION TOOL")
    print(f"  Target Mat Size: {MAT_SIZE_METERS:.1f}m x {MAT_SIZE_METERS:.1f}m")
    print(f"  Total Cameras  : {len(ANCHOR_IMAGES)}")
    print("=" * 60)

    for img_path in ANCHOR_IMAGES:
        if not os.path.exists(img_path):
            print(f"[-] Warning: File not found ({img_path}). Skipping.")
            continue

        # Derives camera ID from the filename (e.g. 'anchor_gopro_0.jpg' -> 'gopro_0')
        raw_name = os.path.splitext(os.path.basename(img_path))[0]
        cam_id = raw_name.replace("anchor_", "")

        pts = annotate_camera(img_path)
        annotations["cameras"][cam_id] = {
            "image_path": img_path,
            "corners_2d": {
                "top_left": pts[0],
                "top_right": pts[1],
                "bottom_right": pts[2],
                "bottom_left": pts[3]
            }
        }

    out_dir = os.path.dirname(os.path.abspath(OUTPUT_JSON))
    os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(annotations, f, indent=4)

    print("\n" + "=" * 60)
    print(f" SUCCESS: Annotations written to {OUTPUT_JSON}")
    print(f" Cameras saved: {list(annotations['cameras'].keys())}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()