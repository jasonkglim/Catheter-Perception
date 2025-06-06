# Runs catheter pose estimation on a full set of calibraiton images
import cv2
import numpy as np
import os
import sys
import time
import glob
import pickle
import pdb
import matplotlib.pyplot as plt
import segment_anything
from segment_anything import sam_model_registry, SamPredictor
import pandas as pd
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev


def visualize_results(img0, img1):
    # Visualize voxel map in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Set axes limits to match voxel grid limits
    x_min, y_min, z_min = voxel_coordinates.min(axis=0)
    x_max, y_max, z_max = voxel_coordinates.max(axis=0)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # Set labels and aspect ratio
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
    marker_size = (72 * VOXEL_SIZE / (x_max - x_min)) ** 2 * 10
    ax.scatter(
        occupied_points[:, 0],
        occupied_points[:, 1],
        occupied_points[:, 2],
        c="r",
        marker="s",
        s=marker_size,
        edgecolor="k",
    )
    plt.show()

    # Visualize Centerline
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot centerline
    ax.plot(
        centerline_points[:, 0],
        centerline_points[:, 1],
        centerline_points[:, 2],
        "b.-",
        label="Centerline",
    )

    # Plot base and tip
    ax.scatter(*base_coord, color="g", s=80, label="Base")
    ax.scatter(*tip_coord, color="r", s=80, label="Tip")

    # Set axes limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Catheter Centerline and Tip Pose")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Top-down view: projection onto the XY plane
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111)

    # Plot all occupied points as background
    ax2.scatter(
        occupied_points[:, 0],
        occupied_points[:, 1],
        c="lightgray",
        s=5,
        label="Occupied Voxels",
    )

    # Plot centerline
    ax2.plot(
        centerline_points[:, 0],
        centerline_points[:, 1],
        "b.-",
        label="Centerline",
    )

    # Plot base and tip
    ax2.scatter(base_coord[0], base_coord[1], color="g", s=80, label="Base")
    ax2.scatter(tip_coord[0], tip_coord[1], color="r", s=80, label="Tip")

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top-Down View (XY Projection)")
    ax2.set_aspect("equal")
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # Visualize centerline projection onto images
    def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs=None):
        """Project 3D points to 2D image coordinates using OpenCV."""
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1))
        points_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        return points_2d.squeeze()

    # Prepare 3D points for projection
    centerline_3d = centerline_points.astype(np.float32)
    base_3d = base_coord.reshape(1, 3).astype(np.float32)
    tip_3d = tip_coord.reshape(1, 3).astype(np.float32)

    # Project to cam0
    rvec0, _ = cv2.Rodrigues(R[0])
    centerline_img0 = project_points(centerline_3d, rvec0, T[0], K[0], d[0])
    base_img0 = project_points(base_3d, rvec0, T[1], K[0], d[0])
    tip_img0 = project_points(tip_3d, rvec0, T[1], K[0], d[0])

    # Project to cam1
    rvec1, _ = cv2.Rodrigues(R[1])
    centerline_img1 = project_points(centerline_3d, rvec1, T[1], K[1], d[1])
    base_img1 = project_points(base_3d, rvec1, T[1], K[1], d[1])
    tip_img1 = project_points(tip_3d, rvec1, T[1], K[1], d[1])

    # Visualize on cam0 image
    img0_vis = img0.copy()
    for pt in centerline_img0.astype(int):
        cv2.circle(img0_vis, tuple(pt), 2, (255, 0, 0), -1)
    cv2.circle(img0_vis, tuple(base_img0.astype(int)), 6, (0, 255, 0), -1)
    cv2.circle(img0_vis, tuple(tip_img0.astype(int)), 6, (0, 0, 255), -1)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img0_vis, cv2.COLOR_BGR2RGB))
    plt.title("Projection on Camera 0")
    plt.axis("off")
    plt.show()

    # Visualize on cam1 image
    img1_vis = img1.copy()
    for pt in centerline_img1.astype(int):
        cv2.circle(img1_vis, tuple(pt), 2, (255, 0, 0), -1)
    cv2.circle(img1_vis, tuple(base_img1.astype(int)), 6, (0, 255, 0), -1)
    cv2.circle(img1_vis, tuple(tip_img1.astype(int)), 6, (0, 0, 255), -1)

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
    plt.title("Projection on Camera 1")
    plt.axis("off")
    plt.show()

    return


if __name__ == "__main__":
    print("Starting Catheter Pose Estimation...")

    calib_id = "05-16-25"
    calib_base_dir = f"C:\\Users\\jlim\\Documents\\GitHub\\Catheter-Perception\\camera_calibration\\{calib_id}"
    image_dir = "C:\\Users\\jlim\\OneDrive - Cor Medical Ventures\\Documents\\Channel Robotics\\Catheter Calibration Data\\LC_v3_05_20_25_T1\\image_snapshots"

    print("Loading camera calibration data...")
    # Load pixel color classification model
    with open(
        "camera_calibration/naive_bayes_pixel_classifier.pkl", "rb"
    ) as f:
        pixel_classifier = pickle.load(f)

    # Load Camera parameters
    with open(
        f"camera_calibration/{calib_id}/camera_calib_data.pkl", "rb"
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

    print("Creating voxel space and lookup table...")
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
        rvec, _ = cv2.Rodrigues(
            R[cam_num]
        )  # Rotation vector from rotation matrix
        image_coordinates, _ = cv2.projectPoints(
            voxel_coordinates, rvec, T[cam_num], K[cam_num], d[cam_num]
        )
        # print(f"Min image coordinates: {np.min(image_coordinates, axis=0)}")
        # print(f"Max image coordinates: {np.max(image_coordinates, axis=0)}")
        voxel_lookup_table[cam_num, :, :] = image_coordinates.reshape(-1, 2)

    print("Loading SAM model...")
    # Load SAM model
    checkpoint_path = "C:\\Users\\jlim\\Documents\\GitHub\\segment-anything\\models\\sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    # sam.to("cuda" if torch.cuda.is_available() else "cpu")
    sam.to("cpu")
    sam_predictor = SamPredictor(sam)

    print("Looping over images...")
    # Load images
    cam0_image_files = [
        f
        for f in os.listdir(f"{image_dir}/cam_0")
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    cam1_image_files = [
        f
        for f in os.listdir(f"{image_dir}/cam_1")
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    # make sure image files are sorted by index
    cam0_image_files.sort(key=lambda name: int(name.split("_")[0]))
    cam1_image_files.sort(key=lambda name: int(name.split("_")[0]))

    tip_coordinates = []
    # Loop over images
    for img_num, (image0, image1) in enumerate(
        zip(cam0_image_files, cam1_image_files)
    ):

        print(f"\nProcessing image pair {img_num+1}/{len(cam0_image_files)}")

        try:
            # Read images
            img0 = cv2.imread(f"{image_dir}/cam_0/{image0}")
            img1 = cv2.imread(f"{image_dir}/cam_1/{image1}")

            # Segmentation
            segmentation_masks = []
            print("Segmenting images...")
            start_time = time.time()
            for image in [img0, img1]:
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image_reshaped = image_hsv.reshape((-1, 3))

                foreground_prob = pixel_classifier.predict_proba(
                    image_reshaped
                )[:, 1]
                max_idx = np.argmax(foreground_prob)
                max_y, max_x = np.unravel_index(
                    max_idx, (image.shape[0], image.shape[1])
                )

                input_point = np.array([[max_x, max_y]])
                input_label = np.array([1])

                print("Applying SAM model...")
                sam_predictor.set_image(image)

                masks, scores, logits = sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )

                segmentation_masks.append(masks[0])

            print("Performing voxel carving...")
            voxel_map.fill(0)  # reset voxel map
            for index, (i, j, k) in enumerate(np.ndindex(N_X, N_Y, N_Z)):
                cam0_voxel_coords = voxel_lookup_table[0, index, :]
                cam1_voxel_coords = voxel_lookup_table[1, index, :]

                if (
                    segmentation_masks[0][
                        int(cam0_voxel_coords[1]), int(cam0_voxel_coords[0])
                    ]
                    > 0
                    and segmentation_masks[1][
                        int(cam1_voxel_coords[1]), int(cam1_voxel_coords[0])
                    ]
                    > 0
                ):
                    voxel_map[i, j, k] = 1

            occupied_voxels = np.argwhere(voxel_map == 1)
            occupied_points = occupied_voxels * VOXEL_SIZE - origin

            # Skeletonization
            z_values = occupied_points[:, 2]
            unique_z = np.unique(z_values)
            min_z = np.min(unique_z)
            min_z_slice = occupied_points[np.isclose(z_values, min_z)]
            mean_xy = min_z_slice[:, :2].mean(axis=0)
            base_point = np.array([mean_xy[0], mean_xy[1], min_z])

            skeleton_map = skeletonize(voxel_map)
            skeleton = np.argwhere(skeleton_map == 1) * VOXEL_SIZE - origin
            skeleton = np.vstack((skeleton, base_point))

            sorted_skeleton = skeleton[np.argsort(skeleton[:, 2])]

            # Fit a spline to the sorted skeleton points
            tck, u = splprep(sorted_skeleton.T)
            u_fine = np.linspace(
                0, 1, 100
            )  # Generate fine parameter values for smooth interpolation
            spline = np.array(splev(u_fine, tck))

            # base_idx = np.argmin(np.abs(centerline_points[:, 2]))
            # tip_idx = np.argmax(centerline_points[:, 2])
            # base_idx = centerline_points[0, 2]
            # tip_idx = centerline_points[-1, 2]

            base_coord = spline[:, 0]
            tip_coord = spline[:, -1]
            print(f"Base coordinates: {base_coord}")
            print(f"Tip coordinates: {tip_coord}")

            tip_x = tip_coord[0] - base_coord[0]
            tip_y = tip_coord[1] - base_coord[1]
            tip_z = tip_coord[2] - base_coord[2]
            tip_coordinates.append([tip_x, tip_y, tip_z])
            print(f"Tip coordinates in base frame: {tip_coordinates[-1]}")
            end_time = time.time()
            print(
                f"Processing time for image pair {img_num+1}: {end_time - start_time:.2f} seconds"
            )
            # visualize_results(
            #     img0,
            #     img1,
            # )  # Visualize results for the current image pair

            break

        except Exception as e:
            print(f"⚠️ Error processing image pair {img_num+1}: {e}")
            tip_coordinates.append([-999, -999, -999])
            continue

    tip_pose_df = pd.DataFrame(
        tip_coordinates, columns=["tip_x", "tip_y", "tip_z"]
    )
    save_dir = "C:\\Users\\jlim\\OneDrive - Cor Medical Ventures\\Documents\\Channel Robotics\\Catheter Calibration Data\\LC_v3_05_20_25_T1"

    tip_pose_df.to_csv(
        os.path.join(save_dir, "tip_coordinates.csv"), index=True
    )
