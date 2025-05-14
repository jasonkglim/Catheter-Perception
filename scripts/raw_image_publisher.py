#!/home/arclab/python-envs/sam-env/bin/python

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge



def publish_raw_images():
    rospy.init_node('raw_image_publisher', anonymous=True)

    # Publishers for the two cameras
    pub_cam0 = rospy.Publisher('/camera0/image_raw', Image, queue_size=10)
    pub_cam1 = rospy.Publisher('/camera1/image_raw', Image, queue_size=10)

    # Open video capture for both cameras
    cap_cam0 = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap_cam1 = cv2.VideoCapture('/dev/video2', cv2.CAP_V4L2)

    # Check if cameras are opened successfully
    if not cap_cam0.isOpened():
        rospy.logerr("Failed to open /dev/video0")
        return
    if not cap_cam1.isOpened():
        rospy.logerr("Failed to open /dev/video2")
        return

    bridge = CvBridge()
    rate = rospy.Rate(10)  # 30 Hz

    while not rospy.is_shutdown():
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