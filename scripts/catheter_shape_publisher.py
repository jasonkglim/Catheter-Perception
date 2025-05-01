#!/home/arclab/python-envs/sam-env/bin/python

import rospy
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def get_fresh_frame(cap, num_frames=1):
    '''
    Function to get a fresh frame from the camera.
    Necessary since my driver discards newest frames, not oldest
    Args:
        cap: Camera object
        num_frames: Number of frames to skip
    Returns:
        Fresh frame from the camera
    '''
    for _ in range(num_frames):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        rospy.logerr("Failed to capture image from camera")
        return None
    return frame

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_tip_pose(masks):
    '''
    Function to calculate the tip pose from image masks from two cameras
    Args:
        masks: List of masks from the two cameras
    Returns:
        tip pose in catheter base frame
    '''
    rospy.loginfo("Calculating tip pose...")
    return np.array([0, 0, 0])

def get_mask(image, image_name, predictor):
    '''
    Function to get the mask from the image using SAM
    Args:
        image: Input image
    Returns:
        mask: Generated mask
    '''
    rospy.loginfo(f"Generating mask for frame {image_name}...")

    # Input image to SAM model
    start_time = rospy.Time.now().to_sec()
    predictor.set_image(image)
    rospy.loginfo(f"Time taken to set image: {rospy.Time.now().to_sec() - start_time:.4f} seconds")

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

    # Visualize and save the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(masks[0], alpha=0.5, cmap='Reds')
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', lw=2))
    plt.title("Mask with bounding box")
    plt.axis('off')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "..", "images")
    # os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, f"{image_name}.png"))
    plt.close()
    
    return masks[0]

def main():
    # Initialize the ROS node
    rospy.loginfo("Initializing catheter shape publisher ROS node...")
    rospy.init_node('catheter_shape_publisher', anonymous=True)
    pub = rospy.Publisher('catheter_tip_pose', Point, queue_size=10)
    # rate = rospy.Rate(10)  # 10 Hz

    # Load the SAM model
    checkpoint_path = "/home/arclab/repos/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")
    sam.to(device=device)

    # Initialize the predictor
    predictor = SamPredictor(sam)

    # Initialize the mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Initialize the cameras
    cap1 = cv2.VideoCapture(0)  # Camera 1
    cap1.set(cv2.CAP_PROP_FPS, 30)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # cap2 = cv2.VideoCapture(1)  # Camera 2

    desired_period = rospy.Duration(5)  # Desired period for processing
    while not rospy.is_shutdown():
        start_time = rospy.Time.now()

        # Capture images from two cameras
        rospy.loginfo("Capturing images from cameras...")
        frame1 = get_fresh_frame(cap1)
        frame1_name = f"cam1_{rospy.Time.now().to_sec():.2f}"
        # cv2.imshow("stream", frame1)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break

        # Process the images with SAM
        mask1 = get_mask(frame1, frame1_name, predictor)
        # mask2 = get_mask(frame2, f"cam2_{rospy.Time.now().to_sec():.2f}", predictor)

        # Get the tip pose from the masks
        # tip_pose = get_tip_pose([mask1]) #, mask2])
        tip_pose = np.array([0, 0, 0])

        # Publish the tip pose
        tip_msg = Point()
        tip_msg.x, tip_msg.y, tip_msg.z = tip_pose
        pub.publish(tip_msg)

        elapsed = rospy.Time.now() - start_time
        if elapsed > desired_period:
            rospy.logwarn(f"Loop took {elapsed.to_sec():.2f}s — exceeded 1s budget!")
        else:
            rospy.sleep(desired_period - elapsed)

    # Release the cameras
    cap1.release()
    # cap2.release()
    cv2.destroyAllWindows()
    rospy.loginfo("Catheter shape publisher node terminated.")

if __name__ == "__main__":
    main()