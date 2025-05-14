cmake_minimum_required(VERSION 3.0.2)
project(Catheter-Perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package()

install(PROGRAMS
  scripts/catheter_shape_publisher.py  # adjust this path to your actual script
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)