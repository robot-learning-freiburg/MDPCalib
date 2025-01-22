#pragma once

#include <calib_msgs/ImagePclCorrespondencesStamped.h>
#include <calib_msgs/StringStamped.h>
#include <ceres/ceres.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "optimization_utils/camera_model_pinhole.h"
#include "optimization_utils/ceres_base.h"
#include "optimization_utils/config.h"
#include "optimization_utils/forward.h"
#include "optimization_utils/io_utils.h"
#include "optimization_utils/pose_graph_constraints.h"
#include "optimization_utils/visualizer.h"

namespace optimization_utils {

class Optimizer : public CeresBase {
   public:
    Optimizer(ros::NodeHandle* nh, const std::string& calibration_options, const SolverOptions& solver_options,
              const ceres::Problem::Options& problem_options = ceres::Problem::Options());
    ~Optimizer();

    void SetSubscriber(ros::Subscriber* const synced_poses_filename_subscriber,
                       ros::Subscriber* const gt_extrinsics_subscriber,
                       ros::Subscriber* const correspondences_subscriber, ros::Subscriber* const camera_info_subscriber,
                       ros::Subscriber* const visu_init_subscriber);

    void CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info_msg);
    void CachePosesCallback(const calib_msgs::StringStampedConstPtr& poses_msg);
    void CacheCorrespondencesCallback(const calib_msgs::ImagePclCorrespondencesStampedConstPtr& correspondences_msg);
    void StoreInitVisuCallback(const sensor_msgs::ImageConstPtr& img_msg);
    void SetGtExtrinsicsCallback(const geometry_msgs::PoseStamped& gt_extrinsics_msg);

    void ComputeInitialTransform();
    void ComputeRefinedTransform();

   private:
    void AddRotationConstraint(const RotationConstraint& rotation_constraint, ceres::LossFunction* lf = nullptr);
    void AddTranslationConstraint(const TranslationConstraint& translation_constraint,
                                  ceres::LossFunction* lf = nullptr);
    void AddImgPointConstraint(const ImgPointConstraint& img_point_constraint, ceres::LossFunction* lf = nullptr);

    PosePQ InvertPose(const geometry_msgs::PoseStamped& pose_msg);
    Eigen::MatrixX4d GetPoseDiff(const Eigen::MatrixX4d& pose_gt, const Eigen::MatrixX4d& pose_pred);
    Eigen::MatrixX4d GetPoseDiff(const PosePQ& pose_gt, const PosePQ& pose_pred);

    // Global timer to measure start processing time
    static std::chrono::time_point<std::chrono::high_resolution_clock> GetTimer() {
        static auto start_proc_time = std::chrono::high_resolution_clock::now();
        return start_proc_time;
    }

    template <typename T>
    std::pair<Point3d, Point3d> Evaluate6d(const T& pose_gt, const T& pose_pred) {
        Eigen::Matrix4d poseDiff = this->GetPoseDiff(pose_gt, pose_pred);
        Point3d posError = Point3d(poseDiff.block<3, 1>(0, 3));
        Point3d rotError = poseDiff.block<3, 3>(0, 0).eulerAngles(2, 1, 0);
        return std::make_pair(posError, rotError);
    }

    std::pair<double, double> EvaluateNorms(const PosePQ& pose_gt, const PosePQ& pose_pred) {
        Eigen::Matrix4d poseDiff = this->GetPoseDiff(pose_gt, pose_pred);
        double posError = Point3d(poseDiff.block<3, 1>(0, 3)).norm();

        double rotError = this->GetQuatDist(pose_gt.q, pose_pred.q);
        return std::make_pair(posError, rotError);
    }

    double GetQuatDist(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2);

    ros::Subscriber* synced_poses_filename_subscriber_;
    ros::Subscriber* gt_extrinsics_subscriber_;
    ros::Subscriber* correspondences_subscriber_;
    ros::Subscriber* camera_info_subscriber_;
    ros::Subscriber* visu_init_subscriber_;
    ros::Publisher initial_transform_pub_;
    ros::Publisher initial_transform_meta_pub_;
    ros::Publisher refined_transform_pub_;

    // Camera model for reprojection residuals
    CameraModelPinhole* camera_model_;

    // Cache the odometry poses
    std::vector<std::pair<std::string, int>> synced_poses_filenames_;
    std::string cache_folder_path_;

    // Cache the optimized scales
    std::vector<std::shared_ptr<double>> scales_;

    // Extrinsics
    std::shared_ptr<PosePQ> extrinsics_;
    std::shared_ptr<PosePQ> gt_extrinsics_;

    // Further configuration
    int number_used_poses_;  // Number of poses used for optimization
    int skip_n_poses_;       // Number of initial poses to skip before caching
    int skip_counter_;
    int number_used_pairs_;  // Number of image - point cloud pairs used for refined optimization

    // Cache the correspondences
    Correspondence2D3D correspondences_;

    // IOUtils to store results of the optimizer
    IOUtils io_utils_;
};
}  // namespace optimization_utils
