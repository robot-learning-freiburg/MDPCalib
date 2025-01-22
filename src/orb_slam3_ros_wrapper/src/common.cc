/**
 *
 * Common functions and variables across all modes (mono/stereo, with or w/o imu)
 *
 */

#include "common.h"

ORB_SLAM3::System::eSensor sensor_type;
std::string world_frame_id, cam_frame_id, imu_frame_id;

ros::Publisher pose_pub, map_points_pub;

void setup_ros_publishers(ros::NodeHandle& node_handler, image_transport::ImageTransport& image_transport,
                          ORB_SLAM3::System::eSensor sensor_type) {
    pose_pub = node_handler.advertise<geometry_msgs::PoseStamped>("orb_slam3/camera_pose", 1);

    map_points_pub = node_handler.advertise<sensor_msgs::PointCloud2>("orb_slam3/map_points", 1);
}

void publish_ros_camera_pose(Sophus::SE3f Tcw_SE3f, ros::Time msg_time) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.frame_id = world_frame_id;
    pose_msg.header.stamp = msg_time;

    pose_msg.pose.position.x = Tcw_SE3f.translation().x();
    pose_msg.pose.position.y = Tcw_SE3f.translation().y();
    pose_msg.pose.position.z = Tcw_SE3f.translation().z();

    pose_msg.pose.orientation.w = Tcw_SE3f.unit_quaternion().coeffs().w();
    pose_msg.pose.orientation.x = Tcw_SE3f.unit_quaternion().coeffs().x();
    pose_msg.pose.orientation.y = Tcw_SE3f.unit_quaternion().coeffs().y();
    pose_msg.pose.orientation.z = Tcw_SE3f.unit_quaternion().coeffs().z();

    pose_pub.publish(pose_msg);
}

void publish_ros_tf_transform(Sophus::SE3f T_SE3f, string frame_id, string child_frame_id, ros::Time msg_time) {
    tf::Transform tf_transform = SE3f_to_tfTransform(T_SE3f);

    static tf::TransformBroadcaster tf_broadcaster;

    tf_broadcaster.sendTransform(tf::StampedTransform(tf_transform, msg_time, frame_id, child_frame_id));
}

void publish_ros_tracked_mappoints(std::vector<ORB_SLAM3::MapPoint*> map_points, ros::Time msg_time) {
    sensor_msgs::PointCloud2 cloud = tracked_mappoints_to_pointcloud(map_points, msg_time);

    map_points_pub.publish(cloud);
}

//
// Miscellaneous functions
//
sensor_msgs::PointCloud2 tracked_mappoints_to_pointcloud(std::vector<ORB_SLAM3::MapPoint*> map_points,
                                                         ros::Time msg_time) {
    const int num_channels = 3;  // x y z

    if (map_points.size() == 0) {
        std::cout << "Map point vector is empty!" << std::endl;
    }

    sensor_msgs::PointCloud2 cloud;

    cloud.header.stamp = msg_time;
    cloud.header.frame_id = world_frame_id;
    cloud.height = 1;
    cloud.width = map_points.size();
    cloud.is_bigendian = false;
    cloud.is_dense = true;
    cloud.point_step = num_channels * sizeof(float);
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.fields.resize(num_channels);

    std::string channel_id[] = {"x", "y", "z"};

    for (int i = 0; i < num_channels; i++) {
        cloud.fields[i].name = channel_id[i];
        cloud.fields[i].offset = i * sizeof(float);
        cloud.fields[i].count = 1;
        cloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
    }

    cloud.data.resize(cloud.row_step * cloud.height);

    unsigned char* cloud_data_ptr = &(cloud.data[0]);

    for (unsigned int i = 0; i < cloud.width; i++) {
        if (map_points[i]) {
            Eigen::Vector3d P3Dw = map_points[i]->GetWorldPos().cast<double>();

            tf::Vector3 point_translation(P3Dw.x(), P3Dw.y(), P3Dw.z());

            float data_array[num_channels] = {point_translation.x(), point_translation.y(), point_translation.z()};

            memcpy(cloud_data_ptr + (i * cloud.point_step), data_array, num_channels * sizeof(float));
        }
    }
    return cloud;
}

tf::Transform SE3f_to_tfTransform(Sophus::SE3f T_SE3f) {
    Eigen::Matrix3f R_mat = T_SE3f.rotationMatrix();
    Eigen::Vector3f t_vec = T_SE3f.translation();

    tf::Matrix3x3 R_tf(R_mat(0, 0), R_mat(0, 1), R_mat(0, 2), R_mat(1, 0), R_mat(1, 1), R_mat(1, 2), R_mat(2, 0),
                       R_mat(2, 1), R_mat(2, 2));

    tf::Vector3 t_tf(t_vec(0), t_vec(1), t_vec(2));

    return tf::Transform(R_tf, t_tf);
}
