import os
import sys
import glob
import time
import zipfile
import shutil
from tqdm import tqdm

def compress_depth_directory(cam_dir, remove_loose=True):
    """
    Compresses loose .npy depth frames from a RealSense camera folder into a 
    single ZIP_DEFLATED archive. Verifies integrity before removing loose files.

    Arguments:
        cam_dir (str): Root camera directory containing 'depth_frames' (e.g. 'realsense_cam_0').
        remove_loose (bool): If True, safely deletes the loose depth folder upon verification.

    Returns:
        tuple: (zip_path, raw_mb, compressed_mb)
    """
    depth_dir = os.path.join(cam_dir, 'depth_frames')
    zip_path  = os.path.join(cam_dir, 'depth_frames.zip')

    if not os.path.isdir(depth_dir):
        print(f"[-] No 'depth_frames' directory found in {cam_dir}. Skipping.")
        return None, 0, 0

    npy_files = sorted(glob.glob(os.path.join(depth_dir, "depth_*.npy")))
    if not npy_files:
        print(f"[-] No .npy files found in {depth_dir}. Skipping.")
        return None, 0, 0

    print(f"\n[+] Processing {cam_dir}: Compressing {len(npy_files)} depth frames...")
    start_time = time.time()

    # Calculate raw uncompressed byte size
    raw_bytes = sum(os.path.getsize(f) for f in npy_files)
    raw_mb = raw_bytes / (1024 * 1024)

    # Stream frames into ZIP with Deflate compression
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fpath in tqdm(npy_files, desc=f"  Packing {os.path.basename(cam_dir)}", unit="frame"):
            arcname = os.path.basename(fpath)
            zf.write(fpath, arcname=arcname)

    compressed_bytes = os.path.getsize(zip_path)
    compressed_mb = compressed_bytes / (1024 * 1024)
    elapsed = time.time() - start_time
    ratio = (1 - (compressed_bytes / raw_bytes)) * 100 if raw_bytes > 0 else 0

    print(f"    Raw: {raw_mb:.1f} MB -> Compressed: {compressed_mb:.1f} MB ({ratio:.1f}% reduction in {elapsed:.1f}s)")

    # Integrity verification test
    print("    Verifying archive integrity...")
    with zipfile.ZipFile(zip_path, mode='r') as zf:
        test_result = zf.testzip()
        if test_result is not None:
            raise RuntimeError(f"Archive corruption detected in {zip_path} at entry {test_result}. Loose files retained.")
    print("    Integrity verified.")

    # Remove loose folder to reclaim storage and inodes
    if remove_loose:
        print(f"    Cleaning up loose directory: {depth_dir}")
        shutil.rmtree(depth_dir)

    return zip_path, raw_mb, compressed_mb


def main():
    """
    Scans the current working directory (or directory passed as CLI argument)
    for all 'realsense_cam_*' folders and compresses depth frames.
    """
    target_root = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Locate all realsense camera directories
    cam_dirs = sorted(glob.glob(os.path.join(target_root, "realsense_cam_*")))

    if not cam_dirs:
        print(f"No 'realsense_cam_*' directories detected in: {os.path.abspath(target_root)}")
        return

    print("=" * 60)
    print(f" POST-CAPTURE DEPTH COMPRESSION ENGINE")
    print(f" Found {len(cam_dirs)} camera folder(s)")
    print("=" * 60)

    total_raw = 0.0
    total_comp = 0.0

    for cdir in cam_dirs:
        _, raw_mb, comp_mb = compress_depth_directory(cdir, remove_loose=True)
        total_raw += raw_mb
        total_comp += comp_mb

    if total_raw > 0:
        overall_reduction = (1 - (total_comp / total_raw)) * 100
        print("\n" + "=" * 60)
        print(f" ALL CAMERAS COMPRESSED SUCCESSFULLY")
        print(f" Total Raw Size        : {total_raw / 1024:.2f} GB")
        print(f" Total Compressed Size : {total_comp / 1024:.2f} GB")
        print(f" Space Saved           : {overall_reduction:.1f}%")
        print(" Ready for HPC cluster transfer (rsync / scp)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()