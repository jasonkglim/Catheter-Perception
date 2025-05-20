import cv2
import numpy as np
import os
import sys
import time
import glob
import pickle


image_dir = ""

# Load Camera parameters
calib_id = "05-16-25"
with open(
    f"../camera_calibration/{calib_id}/camera_calibr_data.pkl", "rb"
) as f:
    camera_calib_data = pickle.load(f)
K = []
d = []
R = []
T = []
for cam_num in range(2):
    K.append(
        camera_calib_data[f"cam{cam_num}"]["intrinsics"]["K"]
    )  # Camera intrinsic matrix
    d.append(
        camera_calib_data[f"cam{cam_num}"]["intrinsics"]["d"]
    )  # Distortion coefficients
    R.append(
        camera_calib_data[f"cam{cam_num}"]["extrinsics"]["R"]
    )  # Extrinsic camera-world rotation matrix
    T.append(
        camera_calib_data[f"cam{cam_num}"]["extrinsics"]["T"]
    )  # Extrinsic camera-world translation vector

# Define voxel space and create lookup table
VOXEL_SIZE = 0.0005  # mm
N_X = 60
N_Y = 60
N_Z = 60
voxel_map = np.zeros((N_X, N_Y, N_Z), dtype=np.uint8)
# location of physical world frame origin in voxel space
origin = np.array([(N_X - 1) / 2, (N_Y - 1) / 2, 0]) * VOXEL_SIZE
# origin = np.array([0, 0, 0])
# Physical world frame coordinates of voxels
voxel_coordinates = (
    np.mgrid[0:N_X, 0:N_Y, 0:N_Z].reshape(3, -1).T * VOXEL_SIZE - origin
)
voxel_lookup_table = np.zeros((2, N_X * N_Y * N_Z, 2), dtype=np.float32)
for cam_num in range(2):
    rvec, _ = cv2.Rodrigues(R[cam_num])  # Rotation vector from rotation matrix
    image_coordinates, _ = cv2.projectPoints(
        voxel_coordinates, rvec, T[cam_num], K[cam_num], d[cam_num]
    )
    print(f"Min image coordinates: {np.min(image_coordinates, axis=0)}")
    print(f"Max image coordinates: {np.max(image_coordinates, axis=0)}")
    voxel_lookup_table[cam_num, :, :] = image_coordinates.reshape(-1, 2)

# Loop through images
num_images = len(os.listdir(f"{image_dir}/cam_0"))
if num_images != len(os.listdir(f"{image_dir}/cam_1")):
    print("Number of images in both cameras do not match")
    sys.exit(1)
for img_num in range(num_images):
    