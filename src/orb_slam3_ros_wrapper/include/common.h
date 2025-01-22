#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>

// ORB-SLAM3-specific libraries. Directory is defined in CMakeLists.txt: ${ORB_SLAM3_DIR}
#include "include/ImuTypes.h"
#include "include/System.h"

extern ORB_SLAM3::System::eSensor sensor_type;
extern std::string world_frame_id, cam_frame_id, imu_frame_id;

extern ros::Publisher pose_pub, map_points_pub;

void setup_ros_publishers(ros::NodeHandle&, image_transport::ImageTransport&, ORB_SLAM3::System::eSensor);

void publish_ros_camera_pose(Sophus::SE3f, ros::Time);
void publish_ros_tracked_mappoints(std::vector<ORB_SLAM3::MapPoint*>, ros::Time);
void publish_ros_tf_transform(Sophus::SE3f, string, string, ros::Time);

tf::Transform SE3f_to_tfTransform(Sophus::SE3f);
sensor_msgs::PointCloud2 tracked_mappoints_to_pointcloud(std::vector<ORB_SLAM3::MapPoint*>, ros::Time);
