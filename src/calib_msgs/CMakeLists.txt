cmake_minimum_required(VERSION 2.8.3)
project(calib_msgs)

find_package(catkin REQUIRED
             COMPONENTS message_generation # Needed to generate custom message
                                           # types
                        roscpp
                        rospy
                        std_msgs)

add_message_files(FILES
                  UInt16MultiArrayStamped.msg
                  ImagePclCorrespondencesStamped.msg
                  StringStamped.msg)

# Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES std_msgs)

# Needed to generate custom messages
catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

include_directories(${catkin_INCLUDE_DIRS})
