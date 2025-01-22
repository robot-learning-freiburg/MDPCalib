#include "pose_synchronizer/pose_synchronizer.h"

#include <calib_msgs/StringStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace pose_synchronizer {

PoseSynchronizer::PoseSynchronizer(ros::NodeHandle* nh, const std::string& calibration_options,
                                   bool disable_pose_synchronizer) {
    synced_data_filename_pub_ = nh->advertise<calib_msgs::StringStamped>("synced_data/filename", 10);
    synced_poses_filename_pub_ = nh->advertise<calib_msgs::StringStamped>("synced_data/filename_poses", 10);
    disable_pose_synchronizer_ = disable_pose_synchronizer;

    // Parse the global configuration file and create cache folder
    YAML::Node node = YAML::LoadFile(calibration_options);
    cache_folder_path_ = node["io"]["cache_folder"].as<std::string>();
    if (cache_folder_path_.back() != std::string("/").front()) {
        cache_folder_path_ += "/";
    }
    if (std::filesystem::exists(cache_folder_path_) && !std::filesystem::is_empty(cache_folder_path_)) {
        ROS_ERROR_STREAM("Cache directory '" << cache_folder_path_ << "' already exists and is not empty.");
        ros::shutdown();
    }
    std::filesystem::create_directories(cache_folder_path_);
}

void PoseSynchronizer::LidarCallback(const nav_msgs::OdometryConstPtr& odom_msg,
                                     const sensor_msgs::PointCloud2ConstPtr& pcl_msg) {
    // ROS_INFO_STREAM("Received odom and point cloud");

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = odom_msg->header;
    pose_msg.pose = odom_msg->pose.pose;
    ros::Time msg_time = odom_msg->header.stamp;

    mutex_.lock();

    lidar_buffer_.push_back(pose_msg);
    lidar_buffer_timestamps_.push_back(msg_time);
    lidar_buffer_pcl_.push_back(pcl_msg);

    // Erase data older than a time threshold
    bool cleared_buffer = false;
    auto it = lidar_buffer_.begin();
    auto it_time = lidar_buffer_timestamps_.begin();
    auto it_pcl = lidar_buffer_pcl_.begin();
    for (; it != lidar_buffer_.end();) {
        if (msg_time - *it_time > ros::Duration(BUFFER_TIME)) {
            it = lidar_buffer_.erase(it);
            it_time = lidar_buffer_timestamps_.erase(it_time);
            it_pcl = lidar_buffer_pcl_.erase(it_pcl);
            cleared_buffer = true;
        } else {
            it++;
            it_time++;
            it_pcl++;
        }
    }
    if (cleared_buffer && lidar_buffer_.size() == 1) {
        ROS_WARN_STREAM("Cleared entire LiDAR buffer.");
    }
    mutex_.unlock();
}

void PoseSynchronizer::CameraCallback(const geometry_msgs::PoseStampedConstPtr& odom_msg,
                                      const sensor_msgs::ImageConstPtr& img_msg) {
    // ROS_INFO_STREAM("Received odom and image");

    ros::Time msg_time = odom_msg->header.stamp;
    camera_buffer_.push_back(*odom_msg);
    camera_buffer_timestamps_.push_back(msg_time);
    camera_buffer_images_.push_back(img_msg);

    // Erase data older than a time threshold
    bool cleared_buffer = false;
    auto it = camera_buffer_.begin();
    auto it_time = camera_buffer_timestamps_.begin();
    auto it_img = camera_buffer_images_.begin();
    for (; it != camera_buffer_.end();) {
        if (msg_time - *it_time > ros::Duration(BUFFER_TIME)) {
            it = camera_buffer_.erase(it);
            it_time = camera_buffer_timestamps_.erase(it_time);
            it_img = camera_buffer_images_.erase(it_img);
            cleared_buffer = true;
        } else {
            it++;
            it_time++;
            it_img++;
        }
    }
    if (cleared_buffer && camera_buffer_.size() == 1) {
        ROS_WARN_STREAM("Cleared entire camera buffer.");
    }

    SynchronizeAndPublishData();
}

void PoseSynchronizer::TerminateCallback(const calib_msgs::UInt16MultiArrayStamped& initial_transform_data_msg) {
    if (std::end(initial_transform_data_msg.data.data) - std::begin(initial_transform_data_msg.data.data) == 2) {
        // Terminate this node. The job is done.
        ROS_INFO_STREAM("Done processing. Shutting down the pose synchronizer node.");
        ros::shutdown();
    }
}

geometry_msgs::PoseStamped PoseSynchronizer::InterpolatePose(const geometry_msgs::PoseStamped& pose_1,
                                                             const geometry_msgs::PoseStamped& pose_2,
                                                             ros::Time reference_time) {
    geometry_msgs::PoseStamped interpolated_pose;
    interpolated_pose.header.stamp = reference_time;

    const double& time_1 = pose_1.header.stamp.toSec();
    const double& time_2 = pose_2.header.stamp.toSec();
    const double& time_ref = reference_time.toSec();
    double s = (time_ref - time_1) / (time_2 - time_1);

    // First interpolate the translational part
    interpolated_pose.pose.position.x = pose_1.pose.position.x + s * (pose_2.pose.position.x - pose_1.pose.position.x);
    interpolated_pose.pose.position.y = pose_1.pose.position.y + s * (pose_2.pose.position.y - pose_1.pose.position.y);
    interpolated_pose.pose.position.z = pose_1.pose.position.z + s * (pose_2.pose.position.z - pose_1.pose.position.z);

    // Then interpolate the orientation using quaternions
    Eigen::Quaterniond quat_1(pose_1.pose.orientation.w, pose_1.pose.orientation.x, pose_1.pose.orientation.y,
                              pose_1.pose.orientation.z);
    Eigen::Quaterniond quat_2(pose_2.pose.orientation.w, pose_2.pose.orientation.x, pose_2.pose.orientation.y,
                              pose_2.pose.orientation.z);
    Eigen::Quaterniond quat_1_to_ref = quat_1.conjugate() * quat_1.slerp(s, quat_2);
    quat_1_to_ref.normalize();
    Eigen::Quaterniond quat_ref = quat_1_to_ref * quat_1;
    quat_ref.normalize();
    interpolated_pose.pose.orientation.x = quat_ref.x();
    interpolated_pose.pose.orientation.y = quat_ref.y();
    interpolated_pose.pose.orientation.z = quat_ref.z();
    interpolated_pose.pose.orientation.w = quat_ref.w();

    return interpolated_pose;
}

sensor_msgs::PointCloud2 PoseSynchronizer::TransformPointcloud(const sensor_msgs::PointCloud2ConstPtr& pointcloud,
                                                               const geometry_msgs::PoseStamped& pose,
                                                               const geometry_msgs::PoseStamped& new_pose) {
    tf2::Stamped<tf2::Transform> tf_pose;
    tf2::fromMsg(pose, tf_pose);
    tf2::Stamped<tf2::Transform> new_tf_pose;
    tf2::fromMsg(new_pose, new_tf_pose);

    // Obtain the relative transform between both poses
    tf2::Transform transform = new_tf_pose.inverseTimes(tf_pose);
    geometry_msgs::TransformStamped transform_stamped;
    transform_stamped.transform = tf2::toMsg(transform);

    // Transform the point cloud to the new_pose
    sensor_msgs::PointCloud2 new_pointcloud;
    tf2::doTransform(*pointcloud, new_pointcloud, transform_stamped);

    return new_pointcloud;
}

void PoseSynchronizer::SynchronizeAndPublishData() {
    mutex_.lock();

    // Not enough poses received yet
    if (camera_buffer_.size() == 0 || lidar_buffer_.size() < 2) {
        mutex_.unlock();
        return;
    }

    bool processed_camera = false;  // Flag used for early stopping

    // Use camera as reference time
    auto camera_time_it = camera_buffer_timestamps_.begin();
    auto camera_it = camera_buffer_.begin();
    auto camera_img_it = camera_buffer_images_.begin();
    for (; camera_time_it != camera_buffer_timestamps_.end();) {
        std::vector<double> time_difference;
        for (const auto& lidar_time : lidar_buffer_timestamps_) {
            // cppcheck-suppress unmatchedSuppression
            time_difference.push_back(std::abs((*camera_time_it - lidar_time).toSec()));
        }

        auto result = std::min_element(time_difference.begin(), time_difference.end());
        int result_arg = std::distance(time_difference.begin(), result);

        if (*result < MODALITY_TIME_DIFFERENCE) {
            // Find the 2nd LiDAR pose for interpolation, preferring interpolation over extrapolation
            int iter_arg_1 = result_arg;
            int iter_arg_2 = -1;
            if (iter_arg_1 == 0 && time_difference.size() > 1) {
                // If the first LiDAR pose is the closest, use the second one for interpolation
                iter_arg_2 = 1;
            } else if (iter_arg_1 == time_difference.size() - 1 && time_difference.size() > 1) {
                // If the last LiDAR pose is the closest, use the second last one for interpolation
                iter_arg_1 -= 1;
                iter_arg_2 = iter_arg_1 + 1;
            } else if (lidar_buffer_timestamps_.at(iter_arg_1) < *camera_time_it) {
                // If the closest LiDAR pose is before the camera pose, use the next one for interpolation
                iter_arg_2 = iter_arg_1 + 1;
            } else {
                // If the closest LiDAR pose is after the camera pose, use the previous one for interpolation
                iter_arg_1 -= 1;
                iter_arg_2 = iter_arg_1 + 1;
            }

            // Interpolate the LiDAR pose to the timestamp of the camera
            geometry_msgs::PoseStamped lidar_pose_msg =
                InterpolatePose(lidar_buffer_.at(iter_arg_1), lidar_buffer_.at(iter_arg_2), *camera_time_it);

            sensor_msgs::PointCloud2 lidar_pcl_msg;
            if (disable_pose_synchronizer_) {
                lidar_pcl_msg = *lidar_buffer_pcl_.at(result_arg);
            } else {
                // Transform the LiDAR point cloud to the same timestamp
                lidar_pcl_msg =
                    TransformPointcloud(lidar_buffer_pcl_.at(result_arg), lidar_buffer_.at(result_arg), lidar_pose_msg);
            }

            // Publish the camera pose, the image, the interpolated LiDAR pose, and the transformed point cloud
            lidar_pose_msg.header.frame_id = camera_it->header.frame_id;
            lidar_pcl_msg.header = lidar_pose_msg.header;

            // Publish a filename message if there is a subscriber for either topic, keeping the
            // sequence IDs consistent. But only write the rosbag if there is a subscriber for
            // the specific topic, otherwise also send an empty dummy message.
            if (synced_data_filename_pub_.getNumSubscribers() || synced_poses_filename_pub_.getNumSubscribers()) {
                calib_msgs::StringStamped filename_msg;
                if (synced_data_filename_pub_.getNumSubscribers()) {
                    rosbag::Bag bag;
                    std::stringstream bag_file;
                    bag_file << cache_folder_path_ << camera_time_it->sec << "_" << std::setfill('0') << std::setw(9)
                             << camera_time_it->nsec << ".bag";
                    bag.open(bag_file.str(), rosbag::bagmode::Write);
                    bag.write("camera_pose", camera_it->header.stamp, *camera_it);
                    bag.write("camera_img", camera_it->header.stamp, *camera_img_it);
                    bag.write("lidar_pose", lidar_pose_msg.header.stamp, lidar_pose_msg);
                    bag.write("lidar_pcl", lidar_pcl_msg.header.stamp, lidar_pcl_msg);
                    bag.close();
                    filename_msg.data = bag_file.str();
                }
                filename_msg.header.stamp = camera_it->header.stamp;
                synced_data_filename_pub_.publish(filename_msg);

                // Plus, also store the poses in a separate file
                calib_msgs::StringStamped poses_filename_msg;
                if (synced_poses_filename_pub_.getNumSubscribers()) {
                    rosbag::Bag poses_bag;
                    std::stringstream poses_bag_file;
                    poses_bag_file << cache_folder_path_ << camera_time_it->sec << "_" << std::setfill('0')
                                   << std::setw(9) << camera_time_it->nsec << "_poses.bag";
                    poses_bag.open(poses_bag_file.str(), rosbag::bagmode::Write);
                    poses_bag.write("camera_pose", camera_it->header.stamp, *camera_it);
                    poses_bag.write("lidar_pose", lidar_pose_msg.header.stamp, lidar_pose_msg);
                    poses_bag.close();
                    poses_filename_msg.data = poses_bag_file.str();
                }
                poses_filename_msg.header.stamp = camera_it->header.stamp;
                synced_poses_filename_pub_.publish(poses_filename_msg);
            }

            ROS_INFO_STREAM("Publish synced data "
                            << *result << " with cam from "
                            << (camera_buffer_timestamps_.back() - *camera_time_it).toSec() << "s ago and lidar from "
                            << (lidar_buffer_timestamps_.back() - lidar_buffer_timestamps_.at(result_arg)).toSec()
                            << "s ago");

            // After having processed the camera pose, we can safely remove it from the buffer
            camera_time_it = camera_buffer_timestamps_.erase(camera_time_it);
            camera_it = camera_buffer_.erase(camera_it);
            camera_img_it = camera_buffer_images_.erase(camera_img_it);
            processed_camera = true;
        } else {
            if (processed_camera) {
                // Having processed camera before but now not being able to find corresponding timestamps
                // implies that all future camera elements will not be able to find correspondences either.
                break;
            }

            camera_time_it++;
            camera_it++;
            camera_img_it++;
        }
    }
    mutex_.unlock();
}

}  // namespace pose_synchronizer
