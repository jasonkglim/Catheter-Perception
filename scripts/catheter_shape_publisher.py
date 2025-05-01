#!/home/arclab/python-envs/sam-env/bin/python

import rospy
import cv2
from segment-anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import String
import numpy as np

def get_tip_pose(masks):
    '''
    Function to calculate the tip pose from image masks from two cameras
    Args:
        masks: List of masks from the two cameras
    Returns:
        tip pose in catheter base frame
    '''
    return np.array([0, 0, 0])

def get_mask(image, predictor):
    '''
    Function to get the mask from the image using SAM
    Args:
        image: Input image
    Returns:
        mask: Generated mask
    '''
    # Input image to SAM model
    predictor.set_image(image)

    # Generate bounding box for catheter
    # Obtain coarse mask
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)
    foreground_mask = cv2.bitwise_not(green_mask)
    binary_mask = np.where(foreground_mask > 0, 1, 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(binary_mask)
    box = np.array([[x, y, x + w, y + h]])

    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=False
    )

    return masks[0]

def main():
    # Set up ros publisher
    rospy.init_node('catheter_shape_publisher', anonymous=True)
    pub = rospy.Publisher('catheter_tip_pose', Point, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Load the SAM model
    checkpoint_path = "/home/arclab/repos/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)

    # Initialize the predictor
    predictor = SamPredictor(sam)

    # Initialize the mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Initialize the cameras
    cap1 = cv2.VideoCapture(0)  # Camera 1
    cap2 = cv2.VideoCapture(1)  # Camera 2

    while not rospy.is_shutdown():
        # Capture images from two cameras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            rospy.logerr("Failed to capture images from cameras")
            break

        # Process the images with SAM
        mask1 = get_mask(frame1, predictor)
        mask2 = get_mask(frame2, predictor)

        # Get the tip pose from the masks
        tip_pose = get_tip_pose([mask1, mask2])

        # Publish the tip pose
        tip_msg = Point()
        tip_msg.x, tip_msg.y, tip_msg.z = tip_pose
        pub.publish(tip_msg)

        rate.sleep()