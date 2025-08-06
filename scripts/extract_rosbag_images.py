#!/usr/bin/env python3

import os
import cv2
import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# --------------- User Configuration ---------------
bag_file = "/home/arclab/catkin_ws/src/Catheter-Perception/camera_calibration/test_kalibr.bag"
output_dir = "/home/arclab/catkin_ws/src/Catheter-Perception/camera_calibration/08-02-25"   
topics = ["/cam0/image_raw", "/cam1/image_raw", "/cam2/image_raw"]
image_format = "png"  # or jpg
# --------------------------------------------------

bridge = CvBridge()

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print(f"Reading: {bag_file}")
    bag = rosbag.Bag(bag_file, "r")

    for topic in topics:
        topic_dir = os.path.join(output_dir, topic.strip("/").replace("/", "_"))
        ensure_dir(topic_dir)

    count = {topic: 0 for topic in topics}

    for topic, msg, t in bag.read_messages(topics=topics):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"Failed to convert image from {topic}: {e}")
            continue

        timestamp = msg.header.stamp.to_nsec()
        filename = os.path.join(
            output_dir,
            topic.strip("/").replace("/", "_"),
            f"{timestamp}.{image_format}"
        )

        cv2.imwrite(filename, cv_image)
        count[topic] += 1

    bag.close()

    print("\n✅ Extraction complete:")
    for topic in topics:
        print(f"  {topic}: {count[topic]} images saved")

if __name__ == "__main__":
    main()
