#pragma once

#include <calib_msgs/UInt16MultiArrayStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <mutex>
#include <string>
#include <vector>

namespace pose_synchronizer {

class PoseSynchronizer {
   public:
    explicit PoseSynchronizer(ros::NodeHandle* nh, const std::string& calibration_options,
                              bool disable_pose_synchronizer);

    void LidarCallback(const nav_msgs::OdometryConstPtr& odom_msg, const sensor_msgs::PointCloud2ConstPtr& pcl_msg);
    void CameraCallback(const geometry_msgs::PoseStampedConstPtr& odom_msg, const sensor_msgs::ImageConstPtr& img_msg);

    void TerminateCallback(const calib_msgs::UInt16MultiArrayStamped& initial_transform_data_msg);

   private:
    ros::Publisher synced_data_filename_pub_;
    ros::Publisher synced_poses_filename_pub_;
    bool disable_pose_synchronizer_;
    std::string cache_folder_path_;

    void SynchronizeAndPublishData();
    geometry_msgs::PoseStamped InterpolatePose(const geometry_msgs::PoseStamped& pose_1,
                                               const geometry_msgs::PoseStamped& pose_2, ros::Time reference_time);
    sensor_msgs::PointCloud2 TransformPointcloud(const sensor_msgs::PointCloud2ConstPtr& pointcloud,
                                                 const geometry_msgs::PoseStamped& pose,
                                                 const geometry_msgs::PoseStamped& new_pose);

    std::vector<geometry_msgs::PoseStamped> lidar_buffer_;
    std::vector<ros::Time> lidar_buffer_timestamps_;
    std::vector<sensor_msgs::PointCloud2ConstPtr> lidar_buffer_pcl_;

    std::vector<geometry_msgs::PoseStamped> camera_buffer_;
    std::vector<ros::Time> camera_buffer_timestamps_;
    std::vector<sensor_msgs::ImageConstPtr> camera_buffer_images_;

    std::mutex mutex_;  // to protect the LiDAR buffer

    const float BUFFER_TIME = 4.0;                // keep data in buffer for x seconds
    const float MODALITY_TIME_DIFFERENCE = 0.025;  // synchronize poses max. x seconds apart
};
}  // namespace pose_synchronizer
