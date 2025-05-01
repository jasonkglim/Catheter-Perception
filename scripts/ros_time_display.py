#!/usr/bin/env python3
import rospy
import time

rospy.init_node("ros_time_display", anonymous=True)
rate = rospy.Rate(10)  # 10 Hz

while not rospy.is_shutdown():
    now = rospy.Time.now()
    print(f"ROS Time: {now.to_sec():.4f}", end='\r')
    rate.sleep()
