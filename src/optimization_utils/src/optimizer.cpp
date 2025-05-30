#include "optimization_utils/optimizer.h"

#include <calib_msgs/ImagePclCorrespondencesStamped.h>
#include <calib_msgs/UInt16MultiArrayStamped.h>
#include <ceres/ceres.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <math.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <yaml-cpp/yaml.h>

#include <boost/foreach.hpp>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <iostream>

#include "optimization_utils/ceres_base.h"
#include "optimization_utils/config.h"
#include "optimization_utils/cost_functors.h"
#include "optimization_utils/forward.h"
#include "optimization_utils/pose_graph_constraints.h"
#include "optimization_utils/visualizer.h"

// ToDo: Update the gt extrinsics

namespace optimization_utils {

Optimizer::Optimizer(ros::NodeHandle* nh, const std::string& calibration_options, const SolverOptions& solver_options,
                     const ceres::Problem::Options& problem_options)
    : CeresBase(solver_options.AsCeresOptions(), problem_options), extrinsics_(std::make_shared<PosePQ>()) {
    // ToDo: The gt extrinsics are hard-coded.They wont be valid for the other platforms. Be careful!!
    // camera_pose_subscriber_ = nullptr;
    // lidar_pose_subscriber_ = nullptr;
    synced_poses_filename_subscriber_ = nullptr;
    initial_transform_pub_ = nh->advertise<geometry_msgs::TransformStamped>("/optimizer/initial_transform", 1);
    refined_transform_pub_ = nh->advertise<geometry_msgs::TransformStamped>("/optimizer/refined_transform", 1);
    initial_transform_meta_pub_ =
        nh->advertise<calib_msgs::UInt16MultiArrayStamped>("/optimizer/initial_transform_meta", 1);

    // Parse the global configuration file
    YAML::Node node = YAML::LoadFile(calibration_options);
    number_used_poses_ = node["optimization"]["number_poses"].as<int>();
    skip_n_poses_ = node["optimization"]["starting_pose"].as<int>();
    skip_counter_ = 0;
    number_used_pairs_ = node["optimization"]["number_image_pcl_pairs"].as<int>();
    cache_folder_path_ = node["io"]["cache_folder"].as<std::string>();
    if (cache_folder_path_.back() != std::string("/").front()) {
        cache_folder_path_ += "/";
    }

    // Create run directories
    io_utils_ = IOUtils(node["io"]["path_base"].as<std::string>(), node["io"]["run_name"].as<std::string>());
    io_utils_.createRunDirectories();
    ROS_INFO_STREAM("Created run directories.");
    io_utils_.copyFiles();
    ROS_INFO_STREAM("Copied config files and launch files.");
}

Optimizer::~Optimizer() {
    delete ceres_problem_ptr_;
    if (camera_model_ != nullptr) {
        delete camera_model_;
    }

    // Delete the cache folder
    std::filesystem::remove_all(cache_folder_path_);
}

void Optimizer::SetSubscriber(ros::Subscriber* const synced_poses_filename_subscriber,
                              ros::Subscriber* const gt_extrinsics_subscriber,
                              ros::Subscriber* const correspondences_subscriber,
                              ros::Subscriber* const camera_info_subscriber,
                              ros::Subscriber* const visu_init_subscriber) {
    synced_poses_filename_subscriber_ = synced_poses_filename_subscriber;
    gt_extrinsics_subscriber_ = gt_extrinsics_subscriber;
    correspondences_subscriber_ = correspondences_subscriber;
    camera_info_subscriber_ = camera_info_subscriber;
    visu_init_subscriber_ = visu_init_subscriber;
}

void Optimizer::CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
    ROS_INFO_STREAM("Received intrinsic calibration of camera");

    camera_model_ = new CameraModelPinhole(camera_info_msg->width, camera_info_msg->height, camera_info_msg->P[0],
                                           camera_info_msg->P[5], camera_info_msg->P[2], camera_info_msg->P[6]);

    // Hotfix for Argoverse2
    if (camera_info_msg->P[0] + camera_info_msg->P[5] + camera_info_msg->P[2] + camera_info_msg->P[6] < 10) {
        camera_model_ = new CameraModelPinhole(camera_info_msg->width, camera_info_msg->height, camera_info_msg->K[0],
                                               camera_info_msg->K[4], camera_info_msg->K[2], camera_info_msg->K[5]);
    }

    int img_width, img_height;
    camera_model_->GetImageSize(img_width, img_height);

    ROS_INFO_STREAM("Camera intrinsics " << camera_model_->GetParams());
    ROS_INFO_STREAM("Image width " << img_width);
    ROS_INFO_STREAM("Image height " << img_height);

    if (camera_info_subscriber_ != nullptr) {
        camera_info_subscriber_->shutdown();
    }
}

void Optimizer::CachePosesCallback(const calib_msgs::StringStampedConstPtr& poses_msg) {
    if (skip_counter_ < skip_n_poses_) {
        // Read the bagfile
        rosbag::Bag bag;
        bag.open(poses_msg->data, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back(std::string("camera_pose"));
        rosbag::View view(bag, rosbag::TopicQuery(topics));
        geometry_msgs::PoseStampedConstPtr camera_pose_msg, lidar_pose_msg;
        BOOST_FOREACH (rosbag::MessageInstance const m, view) {  // NOLINT
            camera_pose_msg = m.instantiate<geometry_msgs::PoseStamped>();
        }
        bag.close();

        // Only start counting once ORB-SLAM3 does not report poses at the origin anymore.
        PosePQ camera_pose(camera_pose_msg->pose);
        if (camera_pose.IsAtOrigin()) {
            ROS_INFO_STREAM("Received camera and lidar poses with seq " << poses_msg->header.seq
                                                                        << " [ignored due to origin]");
            std::filesystem::remove(poses_msg->data);
            return;
        }

        skip_counter_++;
        ROS_INFO_STREAM("Received camera and lidar poses with seq " << poses_msg->header.seq << " [ignored "
                                                                    << skip_counter_ << "/" << skip_n_poses_ << "]");
        std::filesystem::remove(poses_msg->data);
        return;
    }

    this->GetTimer();

    ROS_INFO_STREAM("Received camera and lidar poses with seq " << poses_msg->header.seq << " [cached "
                                                                << synced_poses_filenames_.size() + 1 << "/"
                                                                << number_used_poses_ << "]");

    // camera_poses_.push_back(*camera_pose_msg);
    // lidar_poses_.push_back(*lidar_pose_msg);
    synced_poses_filenames_.push_back(std::make_pair(poses_msg->data, poses_msg->header.seq));

    if (synced_poses_filenames_.size() == 1) {
        // Send a first meta message to the CMRNext node to avoid caching unused data
        calib_msgs::UInt16MultiArrayStamped initial_transform_meta_msg;
        initial_transform_meta_msg.data.layout.dim.push_back(std_msgs::MultiArrayDimension());
        initial_transform_meta_msg.data.layout.dim[0].size = 1;
        initial_transform_meta_msg.data.layout.dim[0].stride = 1;
        initial_transform_meta_msg.data.layout.dim[0].label = "seq";
        initial_transform_meta_msg.data.data.push_back(poses_msg->header.seq);
        initial_transform_meta_msg.header.stamp = poses_msg->header.stamp;
        initial_transform_meta_pub_.publish(initial_transform_meta_msg);
    }

    if (synced_poses_filenames_.size() == number_used_poses_) {
        static auto end_proc_timer = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::ratio<1>> duration_proc = end_proc_timer - this->GetTimer();

        std::stringstream proc_duration_stream;
        proc_duration_stream << "Duration of rosbag processing: " << duration_proc.count() << " seconds."
                             << "\n\n";
        ROS_INFO_STREAM(proc_duration_stream.str());
        io_utils_.writeResults(proc_duration_stream);

        if (synced_poses_filename_subscriber_ != nullptr) {
            synced_poses_filename_subscriber_->shutdown();
        }

        ComputeInitialTransform();
    }
}

void Optimizer::CacheCorrespondencesCallback(
    const calib_msgs::ImagePclCorrespondencesStampedConstPtr& correspondences_msg) {
    static auto start_cmrnext_timer = std::chrono::high_resolution_clock::now();

    ROS_INFO_STREAM("Received correspondences with seq " << correspondences_msg->header.seq);

    // Decode the image pixels
    const int img_h = correspondences_msg->image.layout.dim[0].size;
    const int img_w = correspondences_msg->image.layout.dim[1].size;
    std::vector<uint16_t> img_data = correspondences_msg->image.data;
    Eigen::Map<Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> correspondences_img(
        img_data.data(), img_h, img_w);

    // Decode the corresponding points in the point cloud
    const int pcl_h = correspondences_msg->pcl.layout.dim[0].size;
    const int pcl_w = correspondences_msg->pcl.layout.dim[1].size;
    std::vector<float> pcl_data = correspondences_msg->pcl.data;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> correspondences_pcl(
        pcl_data.data(), pcl_h, pcl_w);

    // Cache in a paired fashion
    correspondences_.push_back(std::make_pair(correspondences_img, correspondences_pcl));

    ROS_INFO_STREAM("Cached correspondence with seq " << correspondences_msg->header.seq);

    if (correspondences_.size() == number_used_pairs_) {
        static auto end_cmrnext_timer = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::ratio<1>> duration_cmrnext = end_cmrnext_timer - start_cmrnext_timer;
        std::stringstream cmrnext_duration_stream;
        cmrnext_duration_stream << "Duration of rosbag processing: " << duration_cmrnext.count() << " seconds."
                                << "\n\n";
        ROS_INFO_STREAM(cmrnext_duration_stream.str());
        io_utils_.writeResults(cmrnext_duration_stream);

        if (correspondences_subscriber_ != nullptr) {
            correspondences_subscriber_->shutdown();
        }

        ComputeRefinedTransform();
    }
}

void Optimizer::StoreInitVisuCallback(const sensor_msgs::ImageConstPtr& img_msg) {
    // Convert ROS sensor_msgs::Image into cv::Mat
    cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img_msg);
    cv::Mat img = imgPtr->image;

    // Create filename
    std::string padding = std::string(2 - std::to_string(IOUtils::ctrInitVisuFile).length(), '0');
    std::string filename = "visu_init_" + padding + std::to_string(IOUtils::ctrInitVisuFile++) + ".png";

    // Save rgb image and depth map
    cv::imwrite(io_utils_.pathVisualizationsDir_ + "/" + filename, img);
}

void Optimizer::SetGtExtrinsicsCallback(const geometry_msgs::PoseStamped& gt_extrinsics_msg) {
    // ToDo: There should be an exception handler handling the case that extrinsics is not set
    // Set gt extrinsics
    this->gt_extrinsics_ = std::make_shared<PosePQ>(gt_extrinsics_msg.pose);

    // Log GT calibration if existing
    std::stringstream gtStream;
    gtStream << "Reference extrinsics as translation vector (x,y,z) and quaternion (x,y,z,w): .\nTranslation: ("
             << gt_extrinsics_->p.x() << ", " << gt_extrinsics_->p.y() << ", " << gt_extrinsics_->p.z()
             << ")\nRotation:    (" << gt_extrinsics_->q.x() << ", " << gt_extrinsics_->q.y() << ", "
             << gt_extrinsics_->q.z() << ", " << gt_extrinsics_->q.w() << ")"
             << "\n\n"
             << "Reference extrinsics as pose matrix: "
             << "\n"
             << gt_extrinsics_->ToPoseMatrix() << "\n\n";

    // Log to terminal and to results file
    ROS_INFO_STREAM(gtStream.str());
    io_utils_.writeResults(gtStream);

    // Shut down subscriber
    gt_extrinsics_subscriber_->shutdown();
}

void Optimizer::ComputeInitialTransform() {
    // ToDo: We could also store these constraints to use them lateron in RefineTransform() as they remain the same
    // Allocate space for the optimization parameters
    std::vector<std::shared_ptr<TranslationConstraint>> translation_constraints;
    std::vector<std::shared_ptr<RotationConstraint>> rotation_constraints;

    // This variable will contain the optimized initial transform
    auto& initial_transform = extrinsics_;

    // These variables will containt loaded pose messages
    geometry_msgs::PoseStamped prev_camera_pose_msg, prev_lidar_pose_msg;
    geometry_msgs::PoseStamped camera_pose_msg, lidar_pose_msg;

    // Read the 1st item (limit scope of variables)
    {
        rosbag::Bag bag;
        bag.open(synced_poses_filenames_.front().first, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back(std::string("camera_pose"));
        topics.push_back(std::string("lidar_pose"));
        rosbag::View view(bag, rosbag::TopicQuery(topics));
        BOOST_FOREACH (rosbag::MessageInstance const m, view) {  // NOLINT
            if (m.getTopic() == std::string("camera_pose")) {
                prev_camera_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            } else {
                prev_lidar_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            }
        }
        bag.close();
    }

    // Then, iterate starting from 2nd item
    static auto start_init_timer = std::chrono::high_resolution_clock::now();
    for (auto filename = std::next(synced_poses_filenames_.begin()); filename != synced_poses_filenames_.end();
         filename++) {
        // Read the bagfile
        rosbag::Bag bag;
        bag.open(filename->first, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back(std::string("camera_pose"));
        topics.push_back(std::string("lidar_pose"));
        rosbag::View view(bag, rosbag::TopicQuery(topics));
        BOOST_FOREACH (rosbag::MessageInstance const m, view) {  // NOLINT
            if (m.getTopic() == std::string("camera_pose")) {
                camera_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            } else {
                lidar_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            }
        }
        bag.close();

        PosePQ camera_pose(camera_pose_msg.pose);
        PosePQ lidar_pose(lidar_pose_msg.pose);
        PosePQ prev_camera_pose(prev_camera_pose_msg.pose);
        PosePQ prev_lidar_pose(prev_lidar_pose_msg.pose);

        // If ORB-SLAM3 lost track (created new map), it resets the poses to the origin. To avoid computing a pose
        // difference between poses from different maps, we skip the poses if the current or previous one points to
        // the origin.
        if (camera_pose.IsAtOrigin() || prev_camera_pose.IsAtOrigin()) {
            ROS_INFO_STREAM("Skip poses with seq " << filename->second);
            continue;
        }

        // Compute odometry between two poses
        auto camera_measurement = std::make_shared<MotionMeasurementAsPoseDiff>(
            prev_camera_pose, camera_pose, Eigen::Matrix<double, 6, 6>::Identity());
        auto lidar_measurement = std::make_shared<MotionMeasurementAsPoseDiff>(prev_lidar_pose, lidar_pose,
                                                                               Eigen::Matrix<double, 6, 6>::Identity());
        auto odometry_map = std::make_shared<std::map<std::string, MotionMeasurementAsPoseDiff>>();
        odometry_map->insert(std::pair<std::string, MotionMeasurementAsPoseDiff>("camera", *camera_measurement));
        odometry_map->insert(std::pair<std::string, MotionMeasurementAsPoseDiff>("lidar", *lidar_measurement));

        // Set up the optimization parameters
        auto scale = std::make_shared<double>(10.0);  // Empirically, we initialize the scale with 10.
        scales_.push_back(scale);
        rotation_constraints.push_back(std::make_shared<RotationConstraint>(odometry_map, initial_transform));
        translation_constraints.push_back(
            std::make_shared<TranslationConstraint>(odometry_map, initial_transform, scale));

        // Set up the optimization problem
        AddRotationConstraint(*rotation_constraints.back(), new ceres::CauchyLoss(0.000001));
        AddTranslationConstraint(*translation_constraints.back(), new ceres::CauchyLoss(0.000001));

        // Set the current poses to the previous ones
        prev_camera_pose_msg = camera_pose_msg;
        prev_lidar_pose_msg = lidar_pose_msg;
    }

    // Publish the meta data, i.e., the sequence numbers to obtain the initial transform
    calib_msgs::UInt16MultiArrayStamped initial_transform_meta_msg;
    initial_transform_meta_msg.data.layout.dim.push_back(std_msgs::MultiArrayDimension());
    initial_transform_meta_msg.data.layout.dim[0].size = 2;
    initial_transform_meta_msg.data.layout.dim[0].stride = 1;
    initial_transform_meta_msg.data.layout.dim[0].label = "seq";
    initial_transform_meta_msg.data.data.push_back(synced_poses_filenames_.front().second);
    initial_transform_meta_msg.data.data.push_back(synced_poses_filenames_.back().second);
    initial_transform_meta_msg.header.stamp = camera_pose_msg.header.stamp;
    initial_transform_meta_pub_.publish(initial_transform_meta_msg);

    // Obtain the initial transform. Since this takes some time, we already sent the meta message.
    optimizeProblem();
    static auto end_init_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::ratio<1>> duration_init = end_init_timer - start_init_timer;
    std::stringstream init_duration_stream;
    init_duration_stream << "Duration of initial optimization: " << duration_init.count() << " seconds."
                         << "\n\n";
    ROS_INFO_STREAM(init_duration_stream.str());
    io_utils_.writeResults(init_duration_stream);

    // The initialization is set up such that we obtain the LiDAR pose in the camera coordinate system. We invert this
    // to obtain the camera pose in the LiDAR frame such that we can directly use it to transform the LiDAR points
    // into the camera frame for estimation of correspondences, optimization and visualization
    // We set the translation to zero as CMRNext is robust enough to work with it

    // cppcheck-suppress[constStatement]
    initial_transform->p << 0.0, 0.0, 0.0;
    //    initial_transform->p << 0, 0, 0;

    // ----- LOGGING OF INITIALIZATION RESULTS
    std::stringstream initStream, evalNormInitStream, eval6dInitStream;
    initStream << "Initial extrinsics (after hand-eye calibration) as translation vector (x.y.z) and quaternion "
                  "(x,y,z,w)\nTranslation: ("
               << initial_transform->p.x() << ", " << initial_transform->p.y() << ", " << initial_transform->p.z()
               << ")\nRotation:    (" << initial_transform->q.x() << ", " << initial_transform->q.y() << ", "
               << initial_transform->q.z() << ", " << initial_transform->q.w() << ")"
               << "\n\n"
               << "Initial transform as pose matrix: "
               << "\n"
               << initial_transform->ToPoseMatrix() << "\n\n";

    // Log to terminal and to results file
    ROS_INFO_STREAM(initStream.str());
    io_utils_.writeResults(initStream);

    // Transform inverted pose matrix back to quaternion and position and set up final initial calibration
    Eigen::Matrix4d pose_matrix_inv = initial_transform->ToPoseMatrix().inverse();
    initial_transform->q = Eigen::Quaterniond(pose_matrix_inv.block<3, 3>(0, 0));
    initial_transform->p = Point3d(pose_matrix_inv.block<3, 1>(0, 3));

    // Publish the initial transform
    geometry_msgs::TransformStamped initial_transform_msg;

    initial_transform_msg.transform.translation.x = initial_transform->p.x();
    initial_transform_msg.transform.translation.y = initial_transform->p.y();
    initial_transform_msg.transform.translation.z = initial_transform->p.z();
    initial_transform_msg.transform.rotation.x = initial_transform->q.x();
    initial_transform_msg.transform.rotation.y = initial_transform->q.y();
    initial_transform_msg.transform.rotation.z = initial_transform->q.z();
    initial_transform_msg.transform.rotation.w = initial_transform->q.w();
    initial_transform_msg.header.stamp = camera_pose_msg.header.stamp;
    initial_transform_pub_.publish(initial_transform_msg);
}

void Optimizer::ComputeRefinedTransform() {
    if (camera_model_ == nullptr) {
        ROS_ERROR_STREAM("Has not received camera intrinsics. Terminating.");
        ros::shutdown();
    }

    // Reset before refined optimization
    delete ceres_problem_ptr_;
    ceres_problem_ptr_ = new ceres::Problem();

    auto& refined_transform = extrinsics_;
    // Allocate space for the optimization parameters
    std::vector<std::shared_ptr<TranslationConstraint>> translation_constraints;
    std::vector<std::shared_ptr<RotationConstraint>> rotation_constraints;
    std::vector<std::shared_ptr<ImgPointConstraint>> img_point_constraints;

    ROS_INFO_STREAM("Setting up trajectory constraints for the refinement step.");

    // These variables will containt loaded pose messages
    geometry_msgs::PoseStamped prev_camera_pose_msg, prev_lidar_pose_msg;
    geometry_msgs::PoseStamped camera_pose_msg, lidar_pose_msg;

    // Read the 1st item (limit scope of variables)
    {
        rosbag::Bag bag;
        bag.open(synced_poses_filenames_.front().first, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back(std::string("camera_pose"));
        topics.push_back(std::string("lidar_pose"));
        rosbag::View view(bag, rosbag::TopicQuery(topics));
        BOOST_FOREACH (rosbag::MessageInstance const m, view) {  // NOLINT
            if (m.getTopic() == std::string("camera_pose")) {
                prev_camera_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            } else {
                prev_lidar_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            }
        }
        bag.close();
    }

    // Start from 2nd item and set up trajectory constraints
    uint16_t scales_iter = 0;
    static auto start_refine_timer = std::chrono::high_resolution_clock::now();
    for (auto filename = std::next(synced_poses_filenames_.begin()); filename != synced_poses_filenames_.end();
         filename++) {
        // Read the bagfile
        rosbag::Bag bag;
        bag.open(filename->first, rosbag::bagmode::Read);
        std::vector<std::string> topics;
        topics.push_back(std::string("camera_pose"));
        topics.push_back(std::string("lidar_pose"));
        rosbag::View view(bag, rosbag::TopicQuery(topics));
        BOOST_FOREACH (rosbag::MessageInstance const m, view) {  // NOLINT
            if (m.getTopic() == std::string("camera_pose")) {
                camera_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            } else {
                lidar_pose_msg = *m.instantiate<geometry_msgs::PoseStamped>();
            }
        }
        bag.close();

        PosePQ camera_pose(camera_pose_msg.pose);
        PosePQ lidar_pose(lidar_pose_msg.pose);
        PosePQ prev_camera_pose(prev_camera_pose_msg.pose);
        PosePQ prev_lidar_pose(prev_lidar_pose_msg.pose);

        // If ORB-SLAM3 lost track (created new map), it resets the poses to the origin. To avoid computing a pose
        // difference between poses of different maps, we skip the poses if the current or previous one points to
        // the origin.
        if (camera_pose.IsAtOrigin() || prev_camera_pose.IsAtOrigin()) {
            ROS_INFO_STREAM("Skip poses with seq" << filename->second);
            continue;
        }

        // Compute odometry between two poses
        auto camera_measurement = std::make_shared<MotionMeasurementAsPoseDiff>(
            prev_camera_pose, camera_pose, Eigen::Matrix<double, 6, 6>::Identity());
        auto lidar_measurement = std::make_shared<MotionMeasurementAsPoseDiff>(prev_lidar_pose, lidar_pose,
                                                                               Eigen::Matrix<double, 6, 6>::Identity());
        auto odometry_map = std::make_shared<std::map<std::string, MotionMeasurementAsPoseDiff>>();
        odometry_map->insert(std::pair<std::string, MotionMeasurementAsPoseDiff>("camera", *camera_measurement));
        odometry_map->insert(std::pair<std::string, MotionMeasurementAsPoseDiff>("lidar", *lidar_measurement));

        // Set up the optimization parameters
        // ToDo: Reset attributes such as scale and stuff
        rotation_constraints.push_back(std::make_shared<RotationConstraint>(odometry_map, refined_transform));
        translation_constraints.push_back(
            std::make_shared<TranslationConstraint>(odometry_map, refined_transform, scales_[scales_iter++]));

        // Set up the optimization problem
        AddRotationConstraint(*rotation_constraints.back(), new ceres::CauchyLoss(0.000001),
                              true);  // Use inverse cost function
        AddTranslationConstraint(*translation_constraints.back(), new ceres::CauchyLoss(0.000001),
                                 true);  // Use inverse cost function

        // Set the current poses to the previous ones
        prev_camera_pose_msg = camera_pose_msg;
        prev_lidar_pose_msg = lidar_pose_msg;
    }

    ROS_INFO_STREAM("Setting up image point constraints via correspondences for the refinement step.");
    // Set up image point constraints
    for (auto correspondence_item = std::next(correspondences_.begin()); correspondence_item != correspondences_.end();
         correspondence_item++) {
        // Each correspondence_item holds a set of 2D-3D points. Go through each of them and apply the following steps
        auto& correspondences_img = correspondence_item->first;
        auto& correspondences_pcl = correspondence_item->second;

        for (uint16_t i = 0; i < correspondences_img.cols(); i++) {
            // 1.) Create image point measurement
            auto pt2d = correspondences_img.col(i);
            auto pt3d = correspondences_pcl.col(i);

            auto img_measurement =
                std::make_shared<ImgPointMeasurement>(pt2d.cast<double>(), Eigen::Matrix<double, 2, 2>::Identity());

            // 2.) Create image point constraint
            img_point_constraints.push_back(std::make_shared<ImgPointConstraint>(img_measurement, refined_transform,
                                                                                 pt3d.cast<double>(), *camera_model_));

            // 3.) Add image point constraint to the problem
            AddImgPointConstraint(*img_point_constraints.back(), new ceres::CauchyLoss(0.000001));
        }
    }

    // Obtain the refined transform
    optimizeProblem();
    static auto end_refine_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::ratio<1>> duration_refine = end_refine_timer - start_refine_timer;
    std::stringstream refine_duration_stream;
    refine_duration_stream << "Duration of refinement optimization: " << duration_refine.count() << " seconds."
                           << "\n\n";
    ROS_INFO_STREAM(refine_duration_stream.str());
    io_utils_.writeResults(refine_duration_stream);

    const std::chrono::duration<double, std::ratio<1>> duration_overall = end_refine_timer - this->GetTimer();
    std::stringstream overall_duration_stream;
    overall_duration_stream << "Duration of overall processing: " << duration_overall.count() << " seconds."
                            << "\n\n";
    ROS_INFO_STREAM(overall_duration_stream.str());
    io_utils_.writeResults(overall_duration_stream);

    // Set up pose matrix from the optimized position and quaternion and invert the pose matrix
    Eigen::Matrix4d pose_matrix_inv = refined_transform->ToPoseMatrix().inverse();
    refined_transform->q = Eigen::Quaterniond(pose_matrix_inv.block<3, 3>(0, 0));
    refined_transform->p = Point3d(pose_matrix_inv.block<3, 1>(0, 3));

    // ----- LOGGING OF REFINED RESULTS
    std::stringstream refineStream, evalNormRefineStream, eval6dRefineStream;
    refineStream << "Refined extrinsics (after optimization with correspondences) as translation vector (x.y.z) and "
                    "quaternion (x,y,z,w).\nTranslation: ("
                 << refined_transform->p.x() << ", " << refined_transform->p.y() << ", " << refined_transform->p.z()
                 << ")\nRotation:    (" << refined_transform->q.x() << ", " << refined_transform->q.y() << ", "
                 << refined_transform->q.z() << ", " << refined_transform->q.w() << ")" << std::endl
                 << std::endl
                 << "Refined transform as pose matrix: " << std::endl
                 << refined_transform->ToPoseMatrix() << "\n\n";

    // Log to terminal and to results file
    ROS_INFO_STREAM(refineStream.str());
    io_utils_.writeResults(refineStream);

    // ----- LOGGING OF ERRORS
    auto errorNorm = this->EvaluateNorms(*gt_extrinsics_, *refined_transform);
    auto error6d = this->Evaluate6d(*gt_extrinsics_, *refined_transform);
    eval6dRefineStream
        << "The error (after optimization with correspondences) measured as translation vector (x,y,z) and euler "
           "angles (zyx)\nTranslation: ("
        << error6d.first.x() << ", " << error6d.first.y() << ", " << error6d.first.z() << ") meters\nRotation:    ("
        << error6d.second.x() << ", " << error6d.second.y() << ", " << error6d.second.z() << ") radians"
        << "\n\n";
    ROS_INFO_STREAM(eval6dRefineStream.str());
    io_utils_.writeResults(eval6dRefineStream);

    evalNormRefineStream
        << "The error (after optimization with correspondences) measured as translation magnitude (delta_t) and "
           "rotation magnitude (delta_r)\nTranslation magnitude: ("
        << errorNorm.first << ") meters\nRotation magnitude:    (" << errorNorm.second
        << ") radians\n                       (" << errorNorm.second * 180.0 / M_PI << ") degrees"
        << "\n\n";
    ROS_INFO_STREAM(evalNormRefineStream.str());
    io_utils_.writeResults(evalNormRefineStream);

    // Publish the refined transform
    geometry_msgs::TransformStamped refined_transform_msg;
    refined_transform_msg.transform.translation.x = refined_transform->p.x();
    refined_transform_msg.transform.translation.y = refined_transform->p.y();
    refined_transform_msg.transform.translation.z = refined_transform->p.z();
    refined_transform_msg.transform.rotation.x = refined_transform->q.x();
    refined_transform_msg.transform.rotation.y = refined_transform->q.y();
    refined_transform_msg.transform.rotation.z = refined_transform->q.z();
    refined_transform_msg.transform.rotation.w = refined_transform->q.w();
    refined_transform_msg.header.stamp = camera_pose_msg.header.stamp;
    refined_transform_pub_.publish(refined_transform_msg);

    // Terminate this node. The job is done.
    ROS_INFO_STREAM("Done processing. Shutting down the optimizer node.");
    ros::shutdown();
}

void Optimizer::AddRotationConstraint(const RotationConstraint& rotation_constraint, ceres::LossFunction* lf,
                                      bool invert) {
    ceres::CostFunction* initRotCostFunction = nullptr;
    if (!invert) {
        initRotCostFunction =
            CostFunctorInitNonLin::Create(Eigen::Matrix<double, 3, 3>::Identity(),
                                          rotation_constraint.measurement_ptr_->find("camera")->second.quantity_,
                                          rotation_constraint.measurement_ptr_->find("lidar")->second.quantity_);
    } else {
        initRotCostFunction =
            CostFunctorInitNonLinInv::Create(Eigen::Matrix<double, 3, 3>::Identity(),
                                             rotation_constraint.measurement_ptr_->find("camera")->second.quantity_,
                                             rotation_constraint.measurement_ptr_->find("lidar")->second.quantity_);
    }

    ceres_problem_ptr_->AddResidualBlock(initRotCostFunction, lf,
                                         rotation_constraint.ptr_to_extrinsics_->q.coeffs().data());

    ceres_problem_ptr_->SetParameterization(rotation_constraint.ptr_to_extrinsics_->q.coeffs().data(),
                                            new ceres::EigenQuaternionParameterization);
}

void Optimizer::AddTranslationConstraint(const TranslationConstraint& translation_constraint, ceres::LossFunction* lf,
                                         bool invert) {
    ceres::CostFunction* initTransCostFunction = nullptr;
    if (!invert) {
        initTransCostFunction =
            CostFunctorInitLin::Create(Eigen::Matrix<double, 3, 3>::Identity(),
                                       translation_constraint.measurement_ptr_->find("camera")->second.quantity_,
                                       translation_constraint.measurement_ptr_->find("lidar")->second.quantity_);
    } else {
        initTransCostFunction =
            CostFunctorInitLinInv::Create(Eigen::Matrix<double, 3, 3>::Identity(),
                                          translation_constraint.measurement_ptr_->find("camera")->second.quantity_,
                                          translation_constraint.measurement_ptr_->find("lidar")->second.quantity_);
    }

    // Add residual to the ceres problem
    ceres_problem_ptr_->AddResidualBlock(initTransCostFunction, lf, translation_constraint.ptr_to_extrinsics_->p.data(),
                                         translation_constraint.ptr_to_extrinsics_->q.coeffs().data(),
                                         &(*translation_constraint.scale_));

    ceres_problem_ptr_->SetParameterization(translation_constraint.ptr_to_extrinsics_->q.coeffs().data(),
                                            new ceres::EigenQuaternionParameterization);
}

void Optimizer::AddImgPointConstraint(const ImgPointConstraint& img_point_constraint, ceres::LossFunction* lf) {
    // Set up cost function from constraint
    const Point2d& img_point_meas = img_point_constraint.measurement_ptr_->quantity_;
    const Matrix2d& weights = img_point_constraint.measurement_ptr_->information_matrix_;
    const Point3d& associated_map_pt = img_point_constraint.associated_map_pt_;

    std::unique_ptr<ceres::CostFunctionToFunctor<2, 3>> ceres_projector =
        img_point_constraint.camera_model_.GetWorldPointToImagePointFunctor();

    // Set up cost function from constraint
    ceres::CostFunction* img_point_constraint_cost_function =
        CostFunctorPointReprojectionResidual<ceres::CostFunctionToFunctor<2, 3>>::Create(
            img_point_meas, weights, associated_map_pt, ceres_projector);

    // Get a reference to the pose (DO NOT COPY), since the pose shouldn't be duplicated. Otherwise
    // ceres would interprete this as a new parameter block
    PosePQ& posePQ = *img_point_constraint.ptr_to_pose_;

    // Add the cost function to the problem with the corresponding pose (position and quaternion) to be optimized
    ceres_problem_ptr_->AddResidualBlock(img_point_constraint_cost_function, lf, posePQ.p.data(),
                                         posePQ.q.coeffs().data());

    // Do a local parameterization, since a quaternion is a three dimensional manifold, embedded in a four dimensional
    // space By doing this, ceres considers this, hence only 3 parameters are optimized
    ceres_problem_ptr_->SetParameterization(posePQ.q.coeffs().data(), new ceres::EigenQuaternionParameterization);
}

Eigen::MatrixX4d Optimizer::GetPoseDiff(const Eigen::MatrixX4d& pose_gt, const Eigen::MatrixX4d& pose_pred) {
    // Compute the pose difference as a 4x4 matrix
    auto pose_diff = pose_gt.inverse() * pose_pred;
    return pose_diff;
}

Eigen::MatrixX4d Optimizer::GetPoseDiff(const PosePQ& pose_gt, const PosePQ& pose_pred) {
    // Convert PQ representation into pose matrices and compute the pose difference as a 4x4 matrix
    Eigen::Matrix4d pose_gt_mat = pose_gt.ToPoseMatrix();
    Eigen::Matrix4d pose_pred_mat = pose_pred.ToPoseMatrix();
    Eigen::Matrix4d pose_diff = this->GetPoseDiff(pose_gt_mat, pose_pred_mat);

    return pose_diff;
}

PosePQ Optimizer::InvertPose(const geometry_msgs::PoseStamped& pose_msg) {
    PosePQ out_pose;
    auto pose_pq = PosePQ(pose_msg.pose);
    auto pose = pose_pq.ToPoseMatrix();
    auto inverted_pose = pose.inverse();
    out_pose.q = Eigen::Quaterniond(inverted_pose.block<3, 3>(0, 0));
    out_pose.p = Point3d(inverted_pose.block<3, 1>(0, 3));

    return out_pose;
}

double Optimizer::GetQuatDist(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) {
    auto r = q1 * q2.inverse();
    return 2 * atan2(r.vec().norm(), std::abs(r.w()));
}

}  // namespace optimization_utils
