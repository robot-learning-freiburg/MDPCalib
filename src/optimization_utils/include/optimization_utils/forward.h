#pragma once

#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <vector>

namespace optimization_utils {

// Renaming of variables
using Point2d = Eigen::Vector2d;
using Point3d = Eigen::Vector3d;
using Points2d = std::vector<Point2d>;
using Points3d = std::vector<Point3d>;
using Pose2d = Eigen::Affine2d;
using Pose3d = Eigen::Affine3d;
using Quaterniond = Eigen::Quaterniond;
using Matrix1d = Eigen::Matrix<double, 1, 1>;
using Matrix2d = Eigen::Matrix<double, 2, 2>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Correspondence2D3D = std::vector<std::pair<Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>;

// Define two additional templates, which can be used later
template <typename T>
using Pose3I = Eigen::Transform<T, 3, Eigen::Isometry>;
template <typename T, uint64_t N>
using PoseNI = Eigen::Transform<T, N, Eigen::Isometry>;

struct PosePQ {
    PosePQ() {
        this->p.setZero();
        this->q.setIdentity();
    }
    PosePQ(const Point3d& pt, const Quaterniond& quat) : p(pt), q(quat) {}
    explicit PosePQ(const geometry_msgs::Pose& pose)
        : p(pose.position.x, pose.position.y, pose.position.z),
          q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z) {}

    // Copy constructor
    PosePQ(const PosePQ& posePQ) : p(posePQ.p), q(posePQ.q) {}

    // Sanity checker
    // cppcheck-suppress unusedFunction
    bool IsAtOrigin() {
        if (p[0] == 0 && p[1] == 0 && p[2] == 0 && q.x() == 0 && q.y() == 0 && q.z() == 0 && q.w() == 1) {
            return true;
        }
        return false;
    }

    Eigen::MatrixX4d ToPoseMatrix() const {
        Eigen::Matrix4d pose_matrix = Eigen::Matrix4d::Identity();
        pose_matrix << this->q.normalized().toRotationMatrix();
        pose_matrix.col(3) << this->p;
        return pose_matrix;
    }

    Point3d p;
    Quaterniond q;
};

}  // namespace optimization_utils
