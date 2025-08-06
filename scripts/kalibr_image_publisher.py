#!/home/arclab/python-envs/sam-env/bin/python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import subprocess

# Ensure proper camera configurations
cam0_device = "/dev/cam0"
cam1_device = "/dev/cam1"
cam2_device = "/dev/cam2"
cam0_focus_value = 40
cam1_focus_value = 70
cam2_focus_value = 80
cap_frame_width = 1280
cap_frame_height = 720
crop_box = [
    (593, 521, 528, 296),
    (593, 521, 528, 296),
]  # Unused, but left here for optional cropping
config_commands = {
    cam0_device: [
        f"v4l2-ctl -d {cam0_device} -c focus_automatic_continuous=0",
        f"v4l2-ctl -d {cam0_device} -c auto_exposure=3",
        f"v4l2-ctl -d {cam0_device} -c focus_absolute={cam0_focus_value}",
    ],
    cam1_device: [
        f"v4l2-ctl -d {cam1_device} -c focus_automatic_continuous=0",
        f"v4l2-ctl -d {cam1_device} -c auto_exposure=3",
        f"v4l2-ctl -d {cam1_device} -c focus_absolute={cam1_focus_value}",
    ],
    cam2_device: [
        f"v4l2-ctl -d {cam2_device} -c focus_automatic_continuous=0",
        f"v4l2-ctl -d {cam2_device} -c auto_exposure=3",
        f"v4l2-ctl -d {cam2_device} -c focus_absolute={cam2_focus_value}",
    ],
}


def configure_camera(devices, config_commands):
    for device in devices:
        print(f"Configuring camera on {device}...")
        for command in config_commands[device]:
            subprocess.run(command, shell=True, check=True)
        print("Camera configuration complete!")


def publish_raw_images():
    rospy.init_node("raw_image_publisher", anonymous=True)

    desired_cams = [0, 1, 2]
    devices = [f"/dev/cam{cam}" for cam in desired_cams]
    caps = []
    pubs = []

    for i, device in enumerate(devices):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            rospy.logerr(f"Failed to open {device}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_frame_height)
        caps.append(cap)
        pub = rospy.Publisher(f"/cam{i}/image_raw", Image, queue_size=10)
        pubs.append(pub)

    bridge = CvBridge()
    rate = rospy.Rate(5)
    configure_wait_time = rospy.Duration(5)
    configure_yet = False
    start_time = rospy.Time.now()

    print("Press space bar in a camera window to start publishing and recording...")
    start_publishing = False
    while not rospy.is_shutdown():
        if not configure_yet and (rospy.Time.now() - start_time) > configure_wait_time:
            configure_camera(devices, config_commands)
            configure_yet = True
            rospy.loginfo("Camera configuration complete!")

        timestamp = rospy.Time.now()
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                rospy.logerr("Failed to read frame from camera")
                continue

            frame_display = cv2.resize(frame, (cap_frame_width//2, cap_frame_height//2))
            cv2.imshow(f"cam{i}", frame_display)

            if start_publishing:
                msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = timestamp
                msg.header.frame_id = f"cam{i}"
                pubs[i].publish(msg)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rospy.loginfo("Quitting...")
            break
        elif key == ord(" "):
            start_publishing = True
            rospy.loginfo("Starting image publishing loop...")


        rate.sleep()

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        publish_raw_images()
    except rospy.ROSInterruptException:
        pass
