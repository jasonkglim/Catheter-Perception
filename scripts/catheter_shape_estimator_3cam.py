# Python class for running pose estimation on a pair of images
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
import torch
from camera_calib_data import *
from skimage.morphology import skeletonize, ball
from scipy.interpolate import splprep, splev
import pdb
import traceback
from scipy.spatial import cKDTree
from scipy import ndimage


def predict_full_image(clf, img_bgr):
    feats = to_features_from_bgr(img_bgr)  # HxWx9
    H, W, _ = feats.shape
    X = feats.reshape(-1, feats.shape[-1])
    y_proba = clf.predict_proba(X)[:, 1].reshape(H, W)  # foreground prob
    y_pred = (y_proba >= 0.5).astype(np.uint8)
    return y_pred, y_proba


def to_features_from_bgr(img_bgr):
    """
    Build per-pixel color features from BGR image:
    [B, G, R, H, S, V, L, a, b]
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    B, G, R = cv2.split(img_bgr)
    H, S, V = cv2.split(img_hsv)
    L, A, Bb = cv2.split(img_lab)
    feats = np.stack([B, G, R, H, S, V, L, A, Bb], axis=-1)  # HxWx9
    return feats


class CatheterShapeEstimator:
    def __init__(self, force_cpu=True, voxel_size=0.0005, voxel_range=0.04):
        if force_cpu:
            self.device = "cpu"  # Force CPU for compatibility
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = "vit_b"
        checkpoint_path = "C:\\Users\\jlim\\Documents\\GitHub\\segment-anything\\models\\sam_vit_b_01ec64.pth"
        # checkpoint_path = "/home/arclab/repos/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
        self.sam = sam_model_registry[self.model_type](
            checkpoint=checkpoint_path
        )
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

        # Load camera calibration data
        self.num_cams = 3
        self.K = [K0, K1, K2]  # Camera intrinsic matrices
        self.d = [d0, d1, d2]  # Distortion coefficients
        self.R = [
            R_0_w,
            R_1_w,
            R_2_w,
        ]  # Rotation matrices from world to camera frame
        self.T = [
            T_0_w,
            T_1_w,
            T_2_w,
        ]  # Translation vectors from world to camera frame
        # Bounding boxes for cropping cam0 and cam1 images
        self.crop_box = [
            (414, 232, 805, 569),
            (361, 221, 851, 533),
            (405, 322, 840, 576),
        ]

        # Initialize voxel map and lookup table
        self.voxel_lookup_table = None
        self.voxel_map_setup(voxel_size=voxel_size, voxel_range=voxel_range)

        # Load pixel color classification model
        self.pixel_classifiers = []
        classifier_base_path = "C:\\Users\\jlim\\Documents\\GitHub\\Catheter-Perception\\pixel_classification"
        # classifier_base_path = "/home/arclab/catkin_ws/src/Catheter-Perception/pixel_classification"
        # for i in range(3):
        #     classifier_path = os.path.join(
        #         classifier_base_path,
        #         f"RF_3class_GreenPurpleWhite_cam{i}_model.pkl",
        #     )
        #     with open(classifier_path, "rb") as f:
        #         self.pixel_classifiers.append(pickle.load(f))
        classifier_model = "RF_subtraction_classifier"
        with open(
            os.path.join(classifier_base_path, classifier_model + ".pkl"), "rb"
        ) as f:
            self.pixel_classifier = pickle.load(f)

        # Load ref images for subtraction segmentation method
        ref_img_dir = "C:\\Users\\jlim\\Documents\\GitHub\\Catheter-Perception\\camera_calibration\\08-13-25\\test_calib_images"
        # ref_img_dir = "/home/arclab/catkin_ws/src/Catheter-Perception/camera_calibration/08-14-25/test_calib_images"
        self.ref_images = []
        for i in range(self.num_cams):
            ref_img_path = os.path.join(ref_img_dir, f"cam{i}_0.png")
            ref_img = cv2.imread(ref_img_path)
            self.ref_images.append(ref_img)

    def voxel_map_setup(self, voxel_size, voxel_range):
        self.voxel_size = voxel_size
        self.voxel_range = voxel_range  # total physical size of voxel space
        n_x = int(voxel_range / voxel_size)
        n_y = int(voxel_range / voxel_size)
        n_z = int(voxel_range / voxel_size)
        self.voxel_map_size = (n_x, n_y, n_z)
        # location of physical world frame origin in voxel space
        self.voxel_origin = (
            np.array([(n_x - 1) / 2, (n_y - 1) / 2, (n_z - 1) * 0.9])
            * voxel_size
        )
        # origin = np.array([0, 0, 0])
        # Physical world frame coordinates of voxels
        self.voxel_coordinates = (
            np.mgrid[0:n_x, 0:n_y, 0:n_z].reshape(3, -1).T * voxel_size
            - self.voxel_origin
        )
        self.voxel_lookup_table = np.zeros(
            (self.num_cams, n_x * n_y * n_z, 2), dtype=np.float32
        )
        for cam_num in range(self.num_cams):
            rvec, _ = cv2.Rodrigues(
                self.R[cam_num]
            )  # Rotation vector from rotation matrix
            image_coordinates, _ = cv2.projectPoints(
                self.voxel_coordinates,
                rvec,
                self.T[cam_num],
                self.K[cam_num],
                self.d[cam_num],
            )
            # print(f"Min image coordinates: {np.min(image_coordinates, axis=0)}")
            # print(f"Max image coordinates: {np.max(image_coordinates, axis=0)}")
            self.voxel_lookup_table[cam_num, :, :] = image_coordinates.reshape(
                -1, 2
            )

    def color_classify(self, image, cam_num, visualize=False):
        """
        Classify each pixel in the image based on color.
        Currently uses three color classes: red, green, other.
        Args:
            image (numpy array): Input image to classify.

        Returns:
            class_mask (numpy array): Classified image mask indicating color class.
            prob_mask (numpy array): Probability mask for each class.
        """
        # Convert to HSV and reshape for classification
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_image.reshape((-1, 3))
        # Classify pixels using the pre-trained model
        prediction = self.pixel_classifiers[cam_num].predict(hsv_pixels)
        class_mask = prediction.reshape(image.shape[:2])
        prob_mask = self.pixel_classifiers[cam_num].predict_proba(hsv_pixels)
        prob_mask = prob_mask.reshape((image.shape[0], image.shape[1], -1))

        # # Visualize the classification results
        # if visualize:
        #     plt.figure()
        #     plt.imshow(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
        #     plt.imshow(class_mask, cmap="jet", alpha=0.5)
        #     plt.title("Class Mask")
        #     plt.colorbar(label="Class")
        #     plt.axis("off")
        #     plt.tight_layout()
        #     plt.show()

        return class_mask, prob_mask

    def open_close_mask(self, mask, kernel_size=12):
        """
        Apply morphological opening and closing to the mask.
        Args:
            mask (numpy array): Input binary mask.
            kernel_size (int): Size of the structuring element.

        Returns:
            numpy array: Processed mask after opening and closing.
        """
        # Ensure mask is binary
        mask = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
        return closed_mask

    def sam_segment(self, image, point_coords, point_labels, visualize=False):
        """
        Segment the image using SAM with given point coordinates and labels.
        Args:
            image (numpy array): Input image to segment.
            point_coords (numpy array): Coordinates of points to use as prompts.
            point_labels (numpy array): Labels for the points.

        Returns:
            sam_masks (list): List of masks generated by SAM.
        """
        self.sam_predictor.set_image(image)
        sam_masks, _, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

        # if visualize:
        #     for mask in sam_masks:
        #         # Visualize sam mask by overlaying it on original image. display input points as well
        #         img_disp = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        #         plt.imshow(img_disp)
        #         plt.imshow(mask, alpha=0.5, cmap="jet")
        #         plt.scatter(
        #             point_coords[:, 0],
        #             point_coords[:, 1],
        #             c="lime",
        #             cmap="jet",
        #             s=50,
        #             marker="x",
        #             label="Input Point",
        #         )
        #         plt.axis("off")
        #         plt.show()

        return sam_masks

    def manual_prompt_sam_segment(self, image, visualize=False):
        """
        Segment image by clicking on the foreground and background pixels.
        """
        # Store the clicked points
        self.clicked_points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click: add point as foreground
                self.clicked_points.append((x, y, 1))
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click: add point as background
                self.clicked_points.append((x, y, 0))

        # Show the image and set the mouse callback
        cv2.imshow("Image", image)
        cv2.setMouseCallback("Image", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Separate the points into coordinates and labels
        if not self.clicked_points:
            return []  # No points clicked

        point_coords = np.array([pt[:2] for pt in self.clicked_points])
        point_labels = np.array([pt[2] for pt in self.clicked_points])
        return self.sam_segment(
            image, point_coords, point_labels, visualize=visualize
        )

    def get_max_prob_pixel(self, prob_mask):
        """
        Get the pixel with the maximum probability for each class.
        Args:
            prob_mask (numpy array): Probability mask for each class.

        Returns:
            tuple: Coordinates of the pixel with the maximum probability.
        """
        prob_array = prob_mask.reshape(-1, prob_mask.shape[-1])
        max_prob = np.max(prob_array, axis=0)
        max_idx = np.argmax(prob_array, axis=0)
        unraveled_idx = np.unravel_index(max_idx, prob_mask.shape[:2])
        max_prob_idx = np.array(unraveled_idx).T
        return max_prob_idx, max_prob

    def get_centroid(self, mask):
        """
        Compute the centroid of a binary mask.
        Args:
            mask (numpy array): Binary mask to compute the centroid of.
        Returns:
            tuple: Coordinates of the centroid (x, y).
        """
        points = np.argwhere(mask)
        if points.size == 0:
            return None  # or raise an error if appropriate
        centroid = points.mean(axis=0)  # (row, col) format
        return (centroid[1], centroid[0])  # convert to (x, y)

    def order_points_by_distance(self, unordered_points):
        """
        Order points starting from highest z value, then by closest distance to previous point.

        Args:
            unordered_points (numpy.ndarray): Array of 3D points with shape (N, 3)

        Returns:
            numpy.ndarray: Ordered skeleton points
        """
        if len(unordered_points) == 0:
            return unordered_points

        unordered_points = np.array(unordered_points)
        ordered_points = []
        remaining_points = unordered_points.copy()

        # Find the point with the highest z value as starting point
        max_z_idx = np.argmax(remaining_points[:, 2])
        current_point = remaining_points[max_z_idx]
        ordered_points.append(current_point)
        remaining_points = np.delete(remaining_points, max_z_idx, axis=0)

        # Order remaining points by closest distance to previous point
        while len(remaining_points) > 0:
            # Calculate distances from current point to all remaining points
            distances = np.linalg.norm(
                remaining_points - current_point, axis=1
            )

            # Find the closest point
            closest_idx = np.argmin(distances)
            current_point = remaining_points[closest_idx]
            ordered_points.append(current_point)

            # Remove the selected point from remaining points
            remaining_points = np.delete(remaining_points, closest_idx, axis=0)

        return np.array(ordered_points)

    def convert_pixel_coords(self, pixel_coords, cam_num):
        """
        Converts pixel coordinates in cropped images to original image coordinates.
        Args:
            pixel_coords (tuple): Pixel coordinates in the cropped image.
        Returns:
            tuple: Converted pixel coordinates in the original image.
        """
        orig_x = pixel_coords[0] + self.crop_box[cam_num][1]
        orig_y = pixel_coords[1] + self.crop_box[cam_num][0]
        return (orig_x, orig_y)

    def get_centerline_skeleton(self, voxel_map):
        """
        Fit a spline to the occupied points in voxel space.
        Args:
            occupied_points (numpy array): Points occupied by the catheter in voxel space.
        Returns:
            numpy array: Spline fitted to the occupied points.
            numpy array: Sorted skeleton points.
        """
        # # # # # First perform opening and closing on voxel map
        voxel_map = ndimage.binary_opening(voxel_map, structure=ball(1))
        voxel_map = ndimage.binary_closing(voxel_map, structure=ball(1))

        # Compute base point from min z slice centroid
        occupied_voxels = np.argwhere(voxel_map == 1)
        occupied_points = occupied_voxels * self.voxel_size - self.voxel_origin
        self.occupied_points = occupied_points

        # # Remove the first and last few z slices to avoid noise at base and tip
        # occupied_points = occupied_points[
        #     (occupied_points[:, 2] > -0.002) & (occupied_points[:, 2] < 0.002)
        # ]

        # z_values = occupied_points[:, 2]
        # unique_z = np.unique(z_values)
        # min_z = np.min(unique_z)
        # min_z_slice = occupied_points[np.isclose(z_values, min_z)]
        # mean_xy = min_z_slice[:, :2].mean(axis=0)
        # base_point = np.array([mean_xy[0], mean_xy[1], min_z])

        # Skeletonize voxel map
        skeleton_map = skeletonize(voxel_map)
        skeleton = (
            np.argwhere(skeleton_map == 1) * self.voxel_size
            - self.voxel_origin
        )  # 3D coordinates of skeleton points
        # add base point to skeleton
        # skeleton = np.vstack((skeleton, base_point))
        # Sort skeleton points
        sorted_skeleton = self.order_points_by_distance(skeleton)
        # print("Sorted Skeleton")
        # print(sorted_skeleton)

        # Fit a spline to the sorted skeleton points
        print("shape of sorted skeleton: ", sorted_skeleton.shape)
        tck, u = splprep(sorted_skeleton.T)
        u_fine = np.linspace(
            0, 1, 100
        )  # Generate fine parameter values for smooth interpolation
        spline = np.array(splev(u_fine, tck))
        # spline = sorted_skeleton.T  # Use sorted skeleton directly for now

        return spline, sorted_skeleton

    def estimate_tip_pose(
        self,
        images=[],
        prompt_type="max_prob",
        visualize=False,
        save_path=None,
    ):
        """'
        Estimate the pose of the cathetertip in the given images.
        Args:
            images (list of tuples): List of image pairs to process.
        Returns:
            list: Estimated tip positions in the format [(x, y, z), ...].
            list: Corresponding angles (theta, phi) for each tip position.
        """
        # # Intentionally raise error to test error handling
        # raise Exception("Test error handling")
        tip_positions = []
        base_positions = []
        angles = []
        for image_pair in images:

            # # First verify voxel space setup is correct by visualizing voxel coordinates on images
            # for cam_num, image in enumerate(image_pair):
            #     image_disp = image.copy()
            #     # Project voxel coordinates to image space
            #     voxel_coords_2d = []
            #     for voxel_coords in self.voxel_coordinates:
            #         rvec, _ = cv2.Rodrigues(self.R[cam_num])
            #         projected_coords, _ = cv2.projectPoints(
            #             np.array([voxel_coords]),
            #             rvec,
            #             self.T[cam_num],
            #             self.K[cam_num],
            #             self.d[cam_num],
            #         )
            #         pt = projected_coords[0].ravel()
            #         voxel_coords_2d.append(pt)

            #     voxel_coords_2d = np.array(voxel_coords_2d)

            #     # Project world frame axes to image
            #     origin_3d = np.array([[0, 0, 0]], dtype=np.float32)
            #     x_axis_3d = np.array(
            #         [[0.01, 0, 0]], dtype=np.float32
            #     )  # 1cm along X
            #     y_axis_3d = np.array(
            #         [[0, 0.01, 0]], dtype=np.float32
            #     )  # 1cm along Y
            #     z_axis_3d = np.array(
            #         [[0, 0, 0.01]], dtype=np.float32
            #     )  # 1cm along Z

            #     origin_img, _ = cv2.projectPoints(
            #         origin_3d,
            #         rvec,
            #         self.T[cam_num],
            #         self.K[cam_num],
            #         self.d[cam_num],
            #     )
            #     x_img, _ = cv2.projectPoints(
            #         x_axis_3d,
            #         rvec,
            #         self.T[cam_num],
            #         self.K[cam_num],
            #         self.d[cam_num],
            #     )
            #     y_img, _ = cv2.projectPoints(
            #         y_axis_3d,
            #         rvec,
            #         self.T[cam_num],
            #         self.K[cam_num],
            #         self.d[cam_num],
            #     )
            #     z_img, _ = cv2.projectPoints(
            #         z_axis_3d,
            #         rvec,
            #         self.T[cam_num],
            #         self.K[cam_num],
            #         self.d[cam_num],
            #     )

            #     origin_pt = origin_img[0, 0]
            #     x_pt = x_img[0, 0]
            #     y_pt = y_img[0, 0]
            #     z_pt = z_img[0, 0]

            #     plt.figure(figsize=(10, 8))
            #     plt.imshow(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB))

            #     # Plot voxel coordinates with transparency
            #     plt.scatter(
            #         voxel_coords_2d[:, 0],
            #         voxel_coords_2d[:, 1],
            #         c="blue",
            #         s=1,
            #         alpha=0.1,
            #         label="Voxel Coordinates",
            #     )

            #     # Draw world frame axes
            #     plt.annotate(
            #         "",
            #         xy=x_pt,
            #         xytext=origin_pt,
            #         arrowprops=dict(arrowstyle="->", color="red", lw=2),
            #         label="X-axis",
            #     )
            #     plt.annotate(
            #         "",
            #         xy=y_pt,
            #         xytext=origin_pt,
            #         arrowprops=dict(arrowstyle="->", color="green", lw=2),
            #         label="Y-axis",
            #     )
            #     plt.annotate(
            #         "",
            #         xy=z_pt,
            #         xytext=origin_pt,
            #         arrowprops=dict(arrowstyle="->", color="blue", lw=2),
            #         label="Z-axis",
            #     )

            #     # Add axis labels
            #     plt.text(
            #         x_pt[0],
            #         x_pt[1],
            #         "X",
            #         fontsize=12,
            #         color="red",
            #         fontweight="bold",
            #     )
            #     plt.text(
            #         y_pt[0],
            #         y_pt[1],
            #         "Y",
            #         fontsize=12,
            #         color="green",
            #         fontweight="bold",
            #     )
            #     plt.text(
            #         z_pt[0],
            #         z_pt[1],
            #         "Z",
            #         fontsize=12,
            #         color="blue",
            #         fontweight="bold",
            #     )

            #     plt.title(f"Voxel Coordinates on Camera {cam_num}")
            #     plt.xlabel("X (pixels)")
            #     plt.ylabel("Y (pixels)")
            #     plt.legend()
            #     plt.axis("off")
            #     plt.show()

            # Segment both images
            seg_masks = []
            for cam_num, image in enumerate(image_pair):

                # Crop image
                box = self.crop_box[cam_num]
                image_cropped = image[box[1] : box[3], box[0] : box[2]]

                # # Run color classifier
                # class_mask, prob_mask = self.color_classify(
                #     image_cropped, cam_num, visualize=visualize
                # )
                # max_prob_indices, _ = self.get_max_prob_pixel(prob_mask)

                # Subtraction Segmentation technique
                t0 = time.perf_counter()

                def overlay_mask_on_image(img_bgr, mask, alpha=0.5):
                    """Overlay a binary mask onto an image (BGR)."""
                    overlay = img_bgr.copy()
                    color = (0, 255, 0)  # green overlay
                    overlay[mask > 0] = (
                        overlay[mask > 0] * (1 - alpha)
                        + np.array(color) * alpha
                    ).astype(np.uint8)
                    return overlay

                ref_img = self.ref_images[cam_num]
                ref_img_cropped = ref_img[box[1] : box[3], box[0] : box[2]]
                subtracted_img = cv2.absdiff(image_cropped, ref_img_cropped)
                class_mask, prob_mask = predict_full_image(
                    self.pixel_classifier, subtracted_img
                )
                mask_morph = self.open_close_mask(class_mask, kernel_size=10)
                x, y, w, h = cv2.boundingRect(mask_morph)
                mask_morph_overlay = overlay_mask_on_image(
                    image_cropped, mask_morph, alpha=0.42
                )
                # cv2.rectangle(
                #     mask_morph_overlay, (x, y), (x + w, y + h), (255, 0, 0), 2
                # )
                # cv2.imshow("Mask Morph Overlay", mask_morph_overlay)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                t1 = time.perf_counter()
                bounding_box = np.array([x, y, x + w, y + h])
                self.sam_predictor.set_image(image_cropped)
                sam_masks, scores, logits = self.sam_predictor.predict(
                    box=bounding_box,
                    multimask_output=False,
                )
                t2 = time.perf_counter()
                print("Time for color classification:", t1 - t0)
                print("Time for SAM prediction:", t2 - t1)

                # point_coords = None
                # point_labels = None
                # if prompt_type == "max_prob":
                #     # Run SAM predictor using max probability pixel as prompt
                #     labels = [1, 2]
                #     point_coords = np.flip(
                #         max_prob_indices[labels], axis=1
                #     )  # Use red and 'other' pixels as foreground prompts
                #     # (need to flip image coordinates for SAM)
                #     point_labels = np.ones(
                #         point_coords.shape[0]
                #     )  # Label for the point TODO:
                # elif prompt_type == "centroid":
                #     # Use centroids as prompts
                #     cath_mask = class_mask == 1  # cath class mask
                #     tip_mask = class_mask == 2  # tip class mask
                #     tip_mask = self.open_close_mask(
                #         tip_mask
                #     )  # Morphological operations to clean mask
                #     cath_mask = self.open_close_mask(
                #         cath_mask
                #     )  # Morphological operations to clean mask
                #     # # Visualize masks after morphology
                #     # plt.figure(figsize=(12, 6))
                #     # plt.subplot(1, 2, 1)
                #     # plt.imshow(tip_mask, cmap="gray")
                #     # plt.title("Tip Mask post morphology")
                #     # plt.axis("off")
                #     # plt.subplot(1, 2, 2)
                #     # plt.imshow(cath_mask, cmap="gray")
                #     # plt.title("Catheter Mask post morphology")
                #     # plt.axis("off")
                #     # plt.show()

                #     tip_centroid = self.get_centroid(tip_mask)
                #     cath_centroid = self.get_centroid(cath_mask)
                #     if cath_centroid is None and tip_centroid is None:
                #         raise ValueError("Both centroids are None")
                #     if cath_centroid is None:
                #         point_coords = np.array([tip_centroid])
                #     elif tip_centroid is None:
                #         point_coords = np.array([cath_centroid])
                #     else:
                #         point_coords = np.array([tip_centroid, cath_centroid])
                #     point_labels = np.ones(point_coords.shape[0])
                # elif prompt_type == "max_prob_centroid":
                #     # Use centroid of main cath region and max prob pixel of tip region
                #     cath_mask = class_mask == 1  # cath class mask
                #     cath_mask = self.open_close_mask(cath_mask, kernel_size=5)
                #     cath_centroid = self.get_centroid(cath_mask)
                #     max_prob_tip = np.flip(max_prob_indices[2])
                #     point_coords = np.array([cath_centroid, max_prob_tip])
                #     point_labels = np.ones(point_coords.shape[0])
                # elif prompt_type == "manual":
                #     # Store the clicked points
                #     self.clicked_points = []

                #     def click_event(event, x, y, flags, param):
                #         if event == cv2.EVENT_LBUTTONDOWN:
                #             # Left click: add point as foreground
                #             self.clicked_points.append((x, y, 1))
                #         elif event == cv2.EVENT_RBUTTONDOWN:
                #             # Right click: add point as background
                #             self.clicked_points.append((x, y, 0))

                #     # Show the image and set the mouse callback
                #     cv2.imshow("Image", image_cropped.copy())
                #     cv2.setMouseCallback("Image", click_event)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                #     # Separate the points into coordinates and labels
                #     if not self.clicked_points:
                #         return []  # No points clicked

                #     point_coords = np.array(
                #         [pt[:2] for pt in self.clicked_points]
                #     )
                #     point_labels = np.array(
                #         [pt[2] for pt in self.clicked_points]
                #     )

                # # SAM segmentation
                # self.sam_predictor.set_image(image_cropped)
                # sam_masks, _, _ = self.sam_predictor.predict(
                #     point_coords=point_coords,
                #     point_labels=point_labels,
                #     multimask_output=False,
                # )

                # Visualize SAM masks if needed
                if visualize:
                    plt.figure()
                    plt.imshow(
                        cv2.cvtColor(image_cropped.copy(), cv2.COLOR_BGR2RGB)
                    )
                    plt.imshow(sam_masks[0].copy(), alpha=0.5, cmap="jet")
                    # # Show input points on the image
                    # for pt in point_coords.astype(int):
                    #     # # correct pt for cropped image
                    #     # pt[0] += box[0]
                    #     # pt[1] += box[1]
                    #     plt.scatter(
                    #         pt[0],
                    #         pt[1],
                    #         c="lime",
                    #         s=50,
                    #         marker="x",
                    #         label="Input Point",
                    #     )
                    # Show box
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (bounding_box[0], bounding_box[1]),
                            bounding_box[2] - bounding_box[0],
                            bounding_box[3] - bounding_box[1],
                            edgecolor="red",
                            facecolor="none",
                            linewidth=2,
                            label="Bounding Box",
                        )
                    )
                    plt.title(f"Segmented Image {cam_num}")
                    plt.axis("off")
                    if save_path is not None:
                        plt.savefig(save_path + f"_segmented_cam{cam_num}.png")
                    # plt.show()

                # Pad SAM mask to original image size
                sam_mask = np.zeros(
                    (image.shape[0], image.shape[1]), dtype=np.uint8
                )
                sam_mask[box[1] : box[3], box[0] : box[2]] = sam_masks[
                    0
                ].astype(np.uint8)

                seg_masks.append(sam_mask)

            # Perform voxel carving
            voxel_map = np.zeros(self.voxel_map_size, dtype=np.uint8)
            for index, (i, j, k) in enumerate(np.ndindex(self.voxel_map_size)):
                cam0_voxel_coords = self.voxel_lookup_table[0, index, :]
                cam1_voxel_coords = self.voxel_lookup_table[1, index, :]
                cam2_voxel_coords = self.voxel_lookup_table[2, index, :]

                if (
                    seg_masks[0][
                        int(cam0_voxel_coords[1]),
                        int(cam0_voxel_coords[0]),
                    ]
                    > 0
                    and seg_masks[1][
                        int(cam1_voxel_coords[1]),
                        int(cam1_voxel_coords[0]),
                    ]
                    > 0
                    and seg_masks[2][
                        int(cam2_voxel_coords[1]),
                        int(cam2_voxel_coords[0]),
                    ]
                    > 0
                ):
                    voxel_map[i, j, k] = 1
            occupied_voxels = np.argwhere(voxel_map == 1)
            occupied_points = (
                occupied_voxels * self.voxel_size - self.voxel_origin
            )
            self.occupied_points = occupied_points
            center_spline, skeleton = self.get_centerline_skeleton(
                voxel_map.copy()
            )
            # center_spline = self.compute_centerline(voxel_map)
            # sort by z
            center_spline = center_spline.T
            # center_spline = center_spline[np.argsort(center_spline[:, 2])[::-1]]
            # print("Center spline shape: ", center_spline.shape)
            t3 = time.perf_counter()
            print("Time for voxel carving:", t3 - t2)
            # Compute tip position relative to base
            spline_tip = center_spline[-1, :]
            spline_base = center_spline[0, :]
            tip_x = spline_tip[0] - spline_base[0]
            tip_y = spline_tip[1] - spline_base[1]
            tip_z = spline_tip[2] - spline_base[2]
            tip_pos = (tip_x, tip_y, -1 * tip_z)
            tip_positions.append(spline_tip)
            base_positions.append(spline_base)
            angles.append(self.tip_pos_to_angles([tip_pos])[0])
            # Visualize results
            if visualize:
                self.visualize_results(
                    image_pair,
                    voxel_map,
                    center_spline,
                    skeleton,
                    seg_masks,
                    save_path,
                )

        return tip_positions, base_positions, angles

    def tip_pos_to_angles(self, tip_positions):
        """
        Convert tip positions to theta, phi angle parameters.
        Args:
            tip_positions (list of tuples): List of tip positions (x, y, z).
        Returns:
            tuple: Theta (bending angle) and phi (bending plane angle) in radians.
        """
        angles = []
        for pos in tip_positions:
            x, y, z = pos
            phi = np.arctan2(y, x)
            theta = np.pi - 2 * np.arctan2(z, np.sqrt(x**2 + y**2))
            angles.append((theta, phi))
        return angles

    def get_centerline_sample(self, voxel_map, num_samples=200, radius=0.002):
        """
        Sample points from a tube-like 3D point cloud and find the local centroid
        of neighbors to approximate the centerline.
        """
        occupied_voxels = np.argwhere(voxel_map == 1)
        occupied_points = occupied_voxels * self.voxel_size - self.voxel_origin
        tree = cKDTree(occupied_points)
        sampled_idx = np.random.choice(
            len(occupied_points), num_samples, replace=False
        )
        centerline_points = []

        for idx in range(len(occupied_points)):
            neighbors_idx = tree.query_ball_point(occupied_points[idx], radius)
            if len(neighbors_idx) > 3:
                local_points = occupied_points[neighbors_idx]
                center = local_points.mean(axis=0)
                centerline_points.append(center)
        centerline_points = np.array(centerline_points).reshape(-1, 3)

        return centerline_points

    def visualize_results(
        self,
        imgs,
        voxel_map,
        center_spline,
        skeleton,
        seg_masks,
        save_path=None,
    ):
        """
        Visualize the results of the pose estimation.
        Args:
            img0 (numpy array): Image from camera 0.
            img1 (numpy array): Image from camera 1.
            voxel_map (numpy array): Voxel map of the catheter.
            base_coord (tuple): Base coordinates of the catheter.
            tip_coord (tuple): Tip coordinates of the catheter.
        """
        voxel_coordinates = self.voxel_coordinates
        occupied_points = (
            np.argwhere(voxel_map == 1) * self.voxel_size - self.voxel_origin
        )

        # Convert base and tip coordinates to numpy arrays
        base_coord = np.array(center_spline[0, :])
        tip_coord = np.array(center_spline[-1, :])
        centerline_points = np.array(center_spline)

        # Visualize voxel map in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plt.title("Voxel Map Before Morphology")
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
        marker_size = (72 * self.voxel_size / (x_max - x_min)) ** 2 * 10
        ax.scatter(
            occupied_points[:, 0],
            occupied_points[:, 1],
            occupied_points[:, 2],
            c="r",
            marker="s",
            s=marker_size,
            edgecolor="k",
            # alpha=0.5
        )
        # ax.plot(
        #     skeleton[:, 0],
        #     skeleton[:, 1],
        #     skeleton[:, 2],
        #     "g.-",
        #     # s=marker_size,
        #     label="Skeleton"
        # )
        if save_path is not None:
            plt.savefig(save_path + "_voxel_map_3d.png")

        # # Perform opening and closing on voxel map
        # voxel_map = ndimage.morphology.binary_opening(voxel_map, structure=ball(1))
        # voxel_map = ndimage.morphology.binary_closing(voxel_map, structure=ball(1))
        # occupied_points_morph = (
        #     np.argwhere(voxel_map == 1) * self.voxel_size - self.voxel_origin
        # )

        # # Visualize the voxel map after morphology
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # plt.title("Voxel Map After Morphology")
        # # Set axes limits to match voxel grid limits
        # x_min, y_min, z_min = voxel_coordinates.min(axis=0)
        # x_max, y_max, z_max = voxel_coordinates.max(axis=0)
        # ax.set_xlim([x_min, x_max])
        # ax.set_ylim([y_min, y_max])
        # ax.set_zlim([z_min, z_max])
        # # Set labels and aspect ratio
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
        # marker_size = (72 * self.voxel_size / (x_max - x_min)) ** 2 * 10
        # ax.scatter(
        #     occupied_points_morph[:, 0],
        #     occupied_points_morph[:, 1],
        #     occupied_points_morph[:, 2],
        #     c="r",
        #     marker="s",
        #     s=marker_size,
        #     edgecolor="k",
        #     # alpha=0.5
        # )
        # # ax.plot(
        # #     skeleton[:, 0],
        # #     skeleton[:, 1],
        # #     skeleton[:, 2],
        # #     "g.-",
        # #     # s=marker_size,
        # #     label="Skeleton"
        # # )
        # if save_path is not None:
        #     plt.savefig(save_path + "_voxel_map_morph_3d.png")

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # # Set axes limits to match voxel grid limits
        # x_min, y_min, z_min = voxel_coordinates.min(axis=0)
        # x_max, y_max, z_max = voxel_coordinates.max(axis=0)
        # ax.set_xlim([x_min, x_max])
        # ax.set_ylim([y_min, y_max])
        # ax.set_zlim([z_min, z_max])
        # # Set labels and aspect ratio
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
        # marker_size = (72 * self.voxel_size / (x_max - x_min)) ** 2 * 10
        # # ax.scatter(
        # #     occupied_points[:, 0],
        # #     occupied_points[:, 1],
        # #     occupied_points[:, 2],
        # #     c="b",
        # #     marker="s",
        # #     s=marker_size,
        # #     edgecolor="k",
        # #     alpha=0.5
        # # )
        # ax.plot(
        #     skeleton[:, 0],
        #     skeleton[:, 1],
        #     skeleton[:, 2],
        #     "g.-",
        #     # s=marker_size,
        #     label="Skeleton",
        # )
        # if save_path is not None:
        #     plt.savefig(save_path + "_skeleton_map_3d.png")

        # # Visualize Centerline
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection="3d")

        # # Plot centerline
        # ax.plot(
        #     centerline_points[:, 0],
        #     centerline_points[:, 1],
        #     centerline_points[:, 2],
        #     "b.-",
        #     label="Centerline",
        # )

        # # Plot base and tip
        # ax.scatter(*base_coord, color="g", s=80, label="Base")
        # ax.scatter(*tip_coord, color="r", s=80, label="Tip")

        # # Set axes limits
        # ax.set_xlim([x_min, x_max])
        # ax.set_ylim([y_min, y_max])
        # ax.set_zlim([z_min, z_max])

        # ax.set_xlabel("X (m)")
        # ax.set_ylabel("Y (m)")
        # ax.set_zlabel("Z (m)")
        # ax.set_title("Catheter Centerline and Tip Pose")
        # ax.legend()
        # plt.tight_layout()
        # plt.show()

        # # Top-down view: projection onto the XY plane
        # fig2 = plt.figure(figsize=(6, 6))
        # ax2 = fig2.add_subplot(111)

        # # Plot all occupied points as background
        # ax2.scatter(
        #     occupied_points[:, 0],
        #     occupied_points[:, 1],
        #     c="lightgray",
        #     s=5,
        #     label="Occupied Voxels",
        # )

        # # Plot centerline
        # ax2.plot(
        #     centerline_points[:, 0],
        #     centerline_points[:, 1],
        #     "b.-",
        #     label="Centerline",
        # )

        # # Plot base and tip
        # ax2.scatter(
        #     base_coord[0], base_coord[1], color="g", s=80, label="Base"
        # )
        # ax2.scatter(tip_coord[0], tip_coord[1], color="r", s=80, label="Tip")

        # ax2.set_xlabel("X (m)")
        # ax2.set_ylabel("Y (m)")
        # ax2.set_title("Top-Down View (XY Projection)")
        # ax2.set_aspect("equal")
        # ax2.legend()
        # plt.tight_layout()
        # plt.show()

        # Visualize centerline projection onto images
        def project_points(
            points_3d, rvec, tvec, camera_matrix, dist_coeffs=None
        ):
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
        occupied_points_3d = occupied_points.reshape(-1, 3).astype(np.float32)

        # Project to cam0
        rvec0, _ = cv2.Rodrigues(self.R[0])
        centerline_img0 = project_points(
            centerline_3d, rvec0, self.T[0], self.K[0], self.d[0]
        )
        base_img0 = project_points(
            base_3d, rvec0, self.T[0], self.K[0], self.d[0]
        )
        tip_img0 = project_points(
            tip_3d, rvec0, self.T[0], self.K[0], self.d[0]
        )

        # Project to cam1
        rvec1, _ = cv2.Rodrigues(self.R[1])
        centerline_img1 = project_points(
            centerline_3d, rvec1, self.T[1], self.K[1], self.d[1]
        )
        base_img1 = project_points(
            base_3d, rvec1, self.T[1], self.K[1], self.d[1]
        )
        tip_img1 = project_points(
            tip_3d, rvec1, self.T[1], self.K[1], self.d[1]
        )

        # Project to cam2
        rvec2, _ = cv2.Rodrigues(self.R[2])
        centerline_img2 = project_points(
            centerline_3d, rvec2, self.T[2], self.K[2], self.d[2]
        )
        base_img2 = project_points(
            base_3d, rvec2, self.T[2], self.K[2], self.d[2]
        )
        tip_img2 = project_points(
            tip_3d, rvec2, self.T[2], self.K[2], self.d[2]
        )

        # Visualize on cam0 image
        img0_vis = imgs[0].copy()
        for pt in centerline_img0.astype(int):
            cv2.circle(img0_vis, tuple(pt), 2, (255, 0, 0), -1)
        cv2.circle(img0_vis, tuple(base_img0.astype(int)), 2, (0, 255, 0), -1)
        cv2.circle(img0_vis, tuple(tip_img0.astype(int)), 2, (0, 0, 255), -1)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img0_vis, cv2.COLOR_BGR2RGB))
        plt.title("Projection on Camera 0")
        plt.axis("off")
        if save_path is not None:
            plt.savefig(save_path + "_projection_cam0.png")

        # Visualize on cam1 image
        img1_vis = imgs[1].copy()
        for pt in centerline_img1.astype(int):
            cv2.circle(img1_vis, tuple(pt), 2, (255, 0, 0), -1)
        cv2.circle(img1_vis, tuple(base_img1.astype(int)), 2, (0, 255, 0), -1)
        cv2.circle(img1_vis, tuple(tip_img1.astype(int)), 2, (0, 0, 255), -1)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
        plt.title("Projection on Camera 1")
        plt.axis("off")
        if save_path is not None:
            plt.savefig(save_path + "_projection_cam1.png")

        # Visualize on cam2 image
        img2_vis = imgs[2].copy()
        for pt in centerline_img2.astype(int):
            cv2.circle(img2_vis, tuple(pt), 2, (255, 0, 0), -1)
        cv2.circle(img2_vis, tuple(base_img2.astype(int)), 2, (0, 255, 0), -1)
        cv2.circle(img2_vis, tuple(tip_img2.astype(int)), 2, (0, 0, 255), -1)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
        plt.title("Projection on Camera 2")
        plt.axis("off")
        if save_path is not None:
            plt.savefig(save_path + "_projection_cam2.png")

        # # Visualize occupied voxel points projected onto each image
        # for cam_num in range(self.num_cams):
        #     plt.figure()
        #     img_vis = imgs[cam_num].copy()
        #     rvec, _ = cv2.Rodrigues(self.R[cam_num])
        #     projected_points = project_points(
        #         occupied_points_3d,
        #         rvec,
        #         self.T[cam_num],
        #         self.K[cam_num],
        #         self.d[cam_num],
        #     )
        #     plt.scatter(
        #         projected_points[:, 0],
        #         projected_points[:, 1],
        #         c="red",
        #         s=1,
        #         alpha=0.5,
        #     )
        #     plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        #     plt.title(f"Occupied Voxel Points on Camera {cam_num}")
        #     plt.axis("off")
        #     if save_path is not None:
        #         plt.savefig(save_path + f"_occupied_voxels_cam{cam_num}.png")

        plt.close("all")  # Close all figures to free memory
        # plt.show()

        return


if __name__ == "__main__":
    # Example usage
    estimator = CatheterShapeEstimator(
        force_cpu=True, voxel_range=0.05, voxel_size=0.0005
    )

    # Load example images (replace with actual image loading)
    # base_dir = "/home/arclab/catkin_ws/src/Catheter-Control/resources/CalibrationData/LC_v1_rework_08_14_25_T3"
    base_dir = "C:\\Users\\jlim\\OneDrive - Cor Medical Ventures\\Documents\\Channel Robotics\\Catheter Calibration Data\\LC_v1_rework_08_14_25_T3"
    img_dir = os.path.join(base_dir, "image_snapshots")

    cam0_image_files = [
        f
        for f in os.listdir(f"{img_dir}/cam_0")
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    cam1_image_files = [
        f
        for f in os.listdir(f"{img_dir}/cam_1")
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    cam2_image_files = [
        f
        for f in os.listdir(f"{img_dir}/cam_2")
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    # make sure image files are sorted by index
    cam0_image_files.sort(key=lambda name: int(name.split("_")[0]))
    cam1_image_files.sort(key=lambda name: int(name.split("_")[0]))
    cam2_image_files.sort(key=lambda name: int(name.split("_")[0]))

    save_dir = os.path.join(base_dir, "pose_estimation_results")
    os.makedirs(save_dir, exist_ok=True)

    # all_shape_data = []
    all_base_positions = []
    all_tip_positions = []
    all_angles = []
    errors = []
    for num, (path0, path1, path2) in enumerate(
        zip(cam0_image_files, cam1_image_files, cam2_image_files)
    ):
        print(f"Processing Image pair {num+1} / {len(cam0_image_files)}...")
        print(f"Image pair: {path0}, {path1}, {path2}")

        img0 = cv2.imread(os.path.join(img_dir, "cam_0", path0))
        img1 = cv2.imread(os.path.join(img_dir, "cam_1", path1))
        img2 = cv2.imread(os.path.join(img_dir, "cam_2", path2))
        save_path = os.path.join(save_dir, f"{num}")
        # Estimate pose
        try:
            start_time = time.time()
            tip_positions, base_positions, tip_angles = (
                estimator.estimate_tip_pose(
                    images=[(img0, img1, img2)],
                    prompt_type="centroid",
                    visualize=True,
                    save_path=save_path,
                )
            )
            end_time = time.time()
            print(
                f"Time taken for estimation: {end_time - start_time:.2f} seconds"
            )
            print(
                "Estimated tip position (mm): ",
                [f"{x*1e3:.2f}" for x in tip_positions[0]],
            )
            print(
                "Estimated angles (theta, phi): ",
                [f"{np.degrees(a):.2f}" for a in tip_angles[0]],
            )
            # pdb.set_trace()  # Debugging breakpoint
            all_tip_positions.append(tip_positions[0])
            all_base_positions.append(base_positions[0])
            all_angles.append(tip_angles[0])
            # all_shape_data.append(tip_positions[0] + tip_angles[0])
        except Exception as e:
            print(f"Error processing image pair {num+1}: {e}")
            traceback.print_exc()
            all_tip_positions.append([-999, -999, -999])
            all_base_positions.append([-999, -999, -999])
            all_angles.append([-999, -999])
            errors.append(num)
            # Placeholder for error

        if num == 3:
            break

    avg_base_position = np.mean(all_base_positions, axis=0)
    print(f"Average base position (mm): {avg_base_position * 1e3}")
    all_tip_positions = np.array(all_tip_positions)
    all_angles = np.array(all_angles)
    all_tip_positions = all_tip_positions - avg_base_position
    all_shape_data = np.concatenate((all_tip_positions, all_angles), axis=1)
    pd.DataFrame(
        all_shape_data,
        columns=["tip_x", "tip_y", "tip_z", "theta", "phi"],
    ).to_csv(
        os.path.join(base_dir, "catheter_shape_estimates.csv"), index=False
    )

    print("Pose estimation completed for all image pairs.")
    print("Images with errors:", errors)
