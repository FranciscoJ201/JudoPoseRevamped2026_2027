import argparse
import json
import os
import cv2
import numpy as np


def parse_arguments():
    """
    Parses CLI arguments for interactive mat corner annotation.
    """
    parser = argparse.ArgumentParser(
        description="Interactive tool to select 4 tatami corners for world frame anchoring."
    )
    parser.add_argument(
        "--images", "-i", nargs="+", required=True,
        help="List of anchor image paths (e.g. gopro_0.jpg gopro_1.jpg realsense_cam_0.jpg)."
    )
    parser.add_argument(
        "--output_json", "-o", type=str, default="mat_corners.json",
        help="Output path for annotated corner coordinates."
    )
    parser.add_argument(
        "--mat_size", "-s", type=float, default=8.0,
        help="Physical length of combat area edge in meters (default: 8.0m)."
    )
    return parser.parse_args()


# Global state for OpenCV mouse callback
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
    Loads one image and prompts the user to click the 4 tatami corners.
    """
    global clicked_points, current_display_img
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[-] Could not load: {image_path}")

    window_name = "Tatami Annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        clicked_points = []
        current_display_img = img.copy()
        cv2.imshow(window_name, current_display_img)
        cv2.setMouseCallback(window_name, mouse_callback, img)

        print(f"\n[+] Annotating: {image_path}")
        print("    Click order: 1. Top-Left, 2. Top-Right, 3. Bottom-Right, 4. Bottom-Left")
        print("    Keys: [ENTER/SPACE] Accept | [r] Reset | [q] Cancel")

        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32) and len(clicked_points) == 4:  # Enter or Space
            break
        elif key == ord('r'):
            continue
        elif key == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Annotation cancelled by operator.")

    cv2.destroyAllWindows()
    return clicked_points


def main():
    args = parse_arguments()
    half = args.mat_size / 2.0

    annotations = {
        "metadata": {
            "mat_size_meters": args.mat_size,
            "world_points": {
                "top_left": [-half, half, 0.0],
                "top_right": [half, half, 0.0],
                "bottom_right": [half, -half, 0.0],
                "bottom_left": [-half, -half, 0.0]
            }
        },
        "cameras": {}
    }

    for img_path in args.images:
        cam_id = os.path.splitext(os.path.basename(img_path))[0]
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

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(annotations, f, indent=4)

    print(f"\n[+] Mat annotations saved successfully to: {args.output_json}")


if __name__ == "__main__":
    main()