#!/home/arclab/python-envs/sam-env/bin/python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import subprocess

# Ensure proper camera configurations
port_ids = [0, 2]
cam0_device = f"/dev/cam0"
cam1_device = f"/dev/cam1"
cam0_focus_value = 35
cam1_focus_value = 75
config_commands = {cam0_device: [
                    f"v4l2-ctl -d {cam0_device} -c focus_automatic_continuous=0",
                    f"v4l2-ctl -d {cam0_device} -c auto_exposure=3",
                    f"v4l2-ctl -d {cam0_device} -c focus_absolute={cam0_focus_value}",
                    # f"v4l2-ctl -d {device} -c exposure_time_absolute=333",
                    # f"v4l2-ctl -d {device} -c gain=0",
                    # f"v4l2-ctl -d {device} -c white_balance_automatic=0",
                    # f"v4l2-ctl -d {device} -c white_balance_temperature=4675",
                    # f"v4l2-ctl -d {device} -c brightness=128",
                    # f"v4l2-ctl -d {device} -c contrast=128",
                    # f"v4l2-ctl -d {device} -c saturation=128",
                    ],
                cam1_device: [
                    f"v4l2-ctl -d {cam1_device} -c focus_automatic_continuous=0",
                    f"v4l2-ctl -d {cam1_device} -c auto_exposure=3",
                    f"v4l2-ctl -d {cam1_device} -c focus_absolute={cam1_focus_value}",
                    # f"v4l2-ctl -d {device} -c exposure_time_absolute=333",
                    # f"v4l2-ctl -d {device} -c gain=0",
                    # f"v4l2-ctl -d {device} -c white_balance_automatic=0",
                    # f"v4l2-ctl -d {device} -c white_balance_temperature=4675",
                    # f"v4l2-ctl -d {device} -c brightness=128",
                    # f"v4l2-ctl -d {device} -c contrast=128",
                    # f"v4l2-ctl -d {device} -c saturation=128",
                    ]
                }

def configure_camera(devices, config_commands):
    for device in devices:

        print(f"Configuring camera on {device}...")

        for command in config_commands[device]:
            subprocess.run(command, shell=True, check=True)

        print("Camera configuration complete!")

def publish_raw_images():
    rospy.init_node('raw_image_publisher', anonymous=True)

    # Publishers for the two cameras
    pub_cam0 = rospy.Publisher('/camera0/image_raw', Image, queue_size=10)
    pub_cam1 = rospy.Publisher('/camera1/image_raw', Image, queue_size=10)

    # Open video capture for both cameras with default settings first
    cap_cam0 = cv2.VideoCapture(cam0_device, cv2.CAP_V4L2)
    cap_cam1 = cv2.VideoCapture(cam1_device, cv2.CAP_V4L2)

    # Check if cameras are opened successfully
    if not cap_cam0.isOpened():
        rospy.logerr("Failed to open /dev/video0")
        return
    if not cap_cam1.isOpened():
        rospy.logerr("Failed to open /dev/video2")
        return

    bridge = CvBridge()
    rate = rospy.Rate(10)  # 30 Hz
    configure_wait_time = rospy.Duration(2) # wait this many seconds to configure manual settings
    configure_yet = False
    start_time = rospy.Time.now()
    while not rospy.is_shutdown():

        # Configure manual camera settings
        if not configure_yet and (rospy.Time.now() - start_time) > configure_wait_time:
            configure_camera([cam0_device, cam1_device], config_commands)
            configure_yet = True
            rospy.loginfo("Camera configuration complete!")


        ret0, frame0 = cap_cam0.read()
        ret1, frame1 = cap_cam1.read()

        if ret0:
            msg_cam0 = bridge.cv2_to_imgmsg(frame0, encoding="bgr8")
            pub_cam0.publish(msg_cam0)
        else:
            rospy.logwarn("Failed to read frame from /dev/video0")

        if ret1:
            msg_cam1 = bridge.cv2_to_imgmsg(frame1, encoding="bgr8")
            pub_cam1.publish(msg_cam1)
        else:
            rospy.logwarn("Failed to read frame from /dev/video2")

        rate.sleep()

    # Release resources
    cap_cam0.release()
    cap_cam1.release()

if __name__ == '__main__':
    try:
        publish_raw_images()
    except rospy.ROSInterruptException:
        pass