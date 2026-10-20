import json
import os

import numpy as np


def load_intrinsics(json_path):
    """
    Ingests intrinsic parameters supporting both RealSense and GoPro output formats.

    Arguments:
        json_path (str): Path to intrinsic parameters JSON.

    Returns:
        tuple: (K, D, model, width, height)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    model = data.get("camera_model", data.get("distortion_model", "standard")).lower()
    width = int(data["width"])
    height = int(data["height"])

    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"], dtype=np.float64)
        D = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    else:
        fx, fy = data["fx"], data["fy"]
        cx, cy = data["cx"], data["cy"]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.array(data["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)

    return K, D, model, width, height


def resolve_intrinsics_path(explicit_path, cam_dir):
    """
    Resolves intrinsics JSON path from explicit CLI argument or default directory location.

    Arguments:
        explicit_path (str): Value passed via CLI.
        cam_dir (str): Camera root directory.

    Returns:
        str: Validated path to intrinsics JSON.
    """
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    if cam_dir:
        candidate = os.path.join(cam_dir, "intrinsics.json")
        if os.path.isfile(candidate):
            return candidate
        candidate_parent = os.path.join(os.path.dirname(cam_dir), "intrinsics.json")
        if os.path.isfile(candidate_parent):
            return candidate_parent

    raise FileNotFoundError(f"[-] Could not locate intrinsics JSON for '{explicit_path or cam_dir}'.")
