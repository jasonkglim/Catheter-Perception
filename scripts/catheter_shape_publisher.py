#!/usr/bin/env python

import rospy
import cv2
from segment-anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from geometry_msgs.msg import Point
from std_msgs.msg import String

def get_tip_pose():
    return Point(0.0, 0.0, 0.0)

