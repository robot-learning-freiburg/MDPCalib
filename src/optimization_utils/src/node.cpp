#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <ros/console.h>
#include <ros/package.h>
#include <ros/ros.h>

#include <iostream>

#include "optimization_utils/config.h"
#include "optimization_utils/optimizer.h"

using optimization_utils::Optimizer;
using optimization_utils::SolverOptions;

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "optimizer");
    ROS_INFO_STREAM("Starting optimizer node.");

    ros::NodeHandle nh;

    const std::string calibration_options_path = ros::package::getPath("calib_cfg") + "/config/config.yaml";
    const std::string solver_options_path = ros::package::getPath("optimization_utils") + "/config/config.yaml";
    SolverOptions solver_options(solver_options_path);
    Optimizer optimizer(&nh, calibration_options_path, solver_options);

    // Subscribe to camera and lidar poses
    ros::Subscriber synced_poses_filename_subscriber =
        nh.subscribe("/synced_data/filename_poses", 10, &Optimizer::CachePosesCallback, &optimizer);
    ros::Subscriber gt_extrinsics_subscriber =
        nh.subscribe("/gt_extrinsics", 10, &Optimizer::SetGtExtrinsicsCallback, &optimizer);
    ros::Subscriber correspondences_subscriber =
        nh.subscribe("/cmrnext/correspondences", 10, &Optimizer::CacheCorrespondencesCallback, &optimizer);
    // Subscribe to camera info
    ros::Subscriber camera_info_subscriber =
        nh.subscribe("/camera_undistorted/camera_info", 10, &Optimizer::CameraInfoCallback, &optimizer);
    // Subscribe to visualized reprojections using the extrinsics from the initialization step
    // ros::Subscriber visu_init_subscriber =
    //     nh.subscribe("cmrnext/visualization_init", 10, &Optimizer::StoreInitVisuCallback, &optimizer);
    // Pass the subscriber objects to the optimizer instance
    optimizer.SetSubscriber(&synced_poses_filename_subscriber, &gt_extrinsics_subscriber, &correspondences_subscriber,
                            &camera_info_subscriber, nullptr);

    ros::spin();

    return 0;
}
