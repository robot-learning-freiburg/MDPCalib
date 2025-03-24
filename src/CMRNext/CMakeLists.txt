cmake_minimum_required(VERSION 2.8.3)
project(cmrnext)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    std_msgs
    calib_msgs
    rosbag
)

catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS
                      src/cmrnext/cmrnext_ros_node.py
                      DESTINATION
                      ${CATKIN_PACKAGE_BIN_DESTINATION})
