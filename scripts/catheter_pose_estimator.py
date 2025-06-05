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

class CatheterPoseEstimator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = 'vit_b'
        self.sam = sam_model_registry[self.model_type](checkpoint='sam_vit_b_01ec64.pth')
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

        # Load camera calibration data
        self.K = [K0, K1]  # Camera intrinsic matrices
        self.d = [d0, d1]  # Distortion coefficients
        self.R = [R_0_w, R_1_w]  # Rotation matrices from world to camera frame
        self.T = [T_0_w, T_1_w]  # Translation vectors from world to camera frame
        # Bounding boxes for cropping cam0 and cam1 images
        self.crop_box = [(210, 109, 420, 216), (187, 172, 362, 264)] 

        # Initialize voxel map and lookup table
        self.voxel_map = None
        self.voxel_lookup_table = None
        self.voxel_map_setup()

        # Load pixel color classification model
        classifier_path = '../pixel_classification/rf_3class_model.pkl'
        with open(classifier_path, 'rb') as f:
            self.pixel_classifier = pickle.load(f)

    def voxel_map_setup(self, voxel_size=0.0005, voxel_range=0.05):
        self.voxel_size = voxel_size
        self.voxel_range = voxel_range  # total physical size of voxel space
        n_x = int(voxel_range / voxel_size)
        n_y = int(voxel_range / voxel_size)
        n_z = int(voxel_range / voxel_size)
        self.voxel_map = np.zeros((n_x, n_y, n_z), dtype=np.uint8)
        # location of physical world frame origin in voxel space
        origin = np.array([(n_x - 1) / 2, (n_y - 1) / 2, 0]) * voxel_size
        # origin = np.array([0, 0, 0])
        # Physical world frame coordinates of voxels
        voxel_coordinates = (
            np.mgrid[0:n_x, 0:n_y, 0:n_z].reshape(3, -1).T * voxel_size 
            - origin
        )
        self.voxel_lookup_table = np.zeros((2, n_x * n_y * n_z, 2),
                                           dtype=np.float32)
        for cam_num in range(2):
            rvec, _ = cv2.Rodrigues(
                self.R[cam_num]
            )  # Rotation vector from rotation matrix
            image_coordinates, _ = cv2.projectPoints(
                voxel_coordinates, rvec, self.T[cam_num],
                self.K[cam_num], self.d[cam_num]
            )
            # print(f"Min image coordinates: {np.min(image_coordinates, axis=0)}")
            # print(f"Max image coordinates: {np.max(image_coordinates, axis=0)}")
            self.voxel_lookup_table[cam_num, :, :] = \
                image_coordinates.reshape(-1, 2)
            
    def color_classify(self, image):
        '''
        Classify each pixel in the image based on color.
        Currently uses three color classes: red, green, other.
        Args:
            image (numpy array): Input image to classify.

        Returns:
            class_mask (numpy array): Classified image mask indicating color class.
            prob_mask (numpy array): Probability mask for each class.
        '''
        # Convert to HSV and reshape for classification
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_image.reshape((-1, 3))
        # Classify pixels using the pre-trained model
        prediction = self.pixel_classifier.predict(hsv_pixels)
        class_mask = prediction.reshape(image.shape[:2])
        prob_mask = self.pixel_classifier.predict_proba(hsv_pixels)
        prob_mask = prob_mask.reshape((image.shape[0], image.shape[1], -1))

    def get_max_prob_pixel(self, prob_mask):
        '''
        Get the pixel with the maximum probability for each class.
        Args:
            prob_mask (numpy array): Probability mask for each class.

        Returns:
            tuple: Coordinates of the pixel with the maximum probability.
        '''
        max_prob = np.max(prob_mask)
        max_prob_pixel = np.unravel_index(np.argmax(prob_mask), prob_mask.shape)
        return max_prob_pixel, max_prob
    
    def convert_pixel_coords(self, pixel_coords, cam_num):
        '''
        Converts pixel coordinates in cropped images to original image coordinates.
        Args:
            pixel_coords (tuple): Pixel coordinates in the cropped image.
        Returns:
            tuple: Converted pixel coordinates in the original image.
        '''
        x, y = pixel_coords
        crop_x, crop_y, crop_w, crop_h = self.crop_box[cam_num]
        orig_x = x + crop_x
        orig_y = y + crop_y
        return (orig_x, orig_y)

    def estimate_pose(self, images=[]):
        ''''
        Estimate the pose of the catheter in the given images.
        Args:
            images (list of tuples): List of image pairs (cam0 and cam1) to process.

        '''
        seg_masks = []
        for image_pair in images:
            for cam_num, image in enumerate(image_pair):
                
            