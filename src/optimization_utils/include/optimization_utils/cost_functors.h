#pragma once

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

#include <memory>
#include <vector>

#include "optimization_utils/forward.h"
#include "optimization_utils/measurement_types.h"

namespace optimization_utils {

template <typename C>
struct CostFunctorWeightedReprojection1d {
    CostFunctorWeightedReprojection1d(const double& w, std::unique_ptr<C>& ceres_projector)
        : w_(w), ceres_projector_(ceres_projector.release()) {}
    const double w_;
    const std::unique_ptr<C> ceres_projector_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename C>
struct CostFunctorWeightedReprojection2d {
    CostFunctorWeightedReprojection2d(const Matrix2d& w, std::unique_ptr<C>& ceres_projector)
        : w_(w), ceres_projector_(ceres_projector.release()) {}
    const Matrix2d w_;
    std::unique_ptr<C> ceres_projector_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CostFunctorWeighted6d {
    explicit CostFunctorWeighted6d(const Matrix6d& w) : w_(w) {}
    const Matrix6d w_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CostFunctorWeighted3d {
    explicit CostFunctorWeighted3d(const Matrix3d& w) : w_(w) {}
    const Matrix3d w_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CostFunctorInitLin : public CostFunctorWeighted3d {
    CostFunctorInitLin(const Matrix3d& weights, const PosePQ& cam_b_T_cam_a,
                       const PosePQ& lidar_b_T_lidar_a)
        : CostFunctorWeighted3d(weights),
          cam_b_T_cam_a_(cam_b_T_cam_a),
          lidar_b_T_lidar_a_(lidar_b_T_lidar_a) {}
    template <typename T>
    bool operator()(const T* const pos_calib_ptr, const T* const quat_calib_ptr, const T* const scale_ptr, T* res) const {
        // Map the incoming pointers to Eigen objects
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_calib(pos_calib_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> quat_calib(quat_calib_ptr);

        // Split the computation of the error term into subparts
        Eigen::Matrix<T, 3, 1> eq_left_cam = cam_b_T_cam_a_.q.template cast<T>() * pos_calib +
                                           scale_ptr[0] * cam_b_T_cam_a_.p.template cast<T>();
        Eigen::Matrix<T, 3, 1> eq_right_lidar = quat_calib * lidar_b_T_lidar_a_.p.template cast<T>() + pos_calib;
        Eigen::Matrix<T, 3, 1> delta_pos = eq_left_cam - eq_right_lidar;

        // Compute the residuals
        // [ position ] [ delta_pos ]
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(res);
        residuals.template block<3, 1>(0, 0) = T(2.0) * delta_pos;

        // Scale the residuals by constraint weights
        residuals.applyOnTheLeft((this->w_).template cast<T>());
        return true;
    }

    // cppcheck-suppress unusedFunction
    static ceres::CostFunction* Create(const Matrix3d& weights, const PosePQ& cam_b_T_cam_a,
                                       const PosePQ& lidar_b_T_lidar_a) {
        return new ceres::AutoDiffCostFunction<CostFunctorInitLin, 3, 3, 4, 1>(
            new CostFunctorInitLin(weights, cam_b_T_cam_a, lidar_b_T_lidar_a));
    }

    PosePQ cam_b_T_cam_a_;
    PosePQ lidar_b_T_lidar_a_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct CostFunctorInitNonLin : public CostFunctorWeighted3d {
    CostFunctorInitNonLin(const Matrix3d& weights, const PosePQ& cam_b_T_cam_a,
                          const PosePQ& lidar_b_T_lidar_a)
        : CostFunctorWeighted3d(weights),
          cam_b_T_cam_a_(cam_b_T_cam_a),
          lidar_b_T_lidar_a_(lidar_b_T_lidar_a) {}
    template <typename T>
    bool operator()(const T* const quat_calib_ptr, T* res) const {
        // Map the incoming pointers to Eigen objects
        Eigen::Map<const Eigen::Quaternion<T>> quat_calib(quat_calib_ptr);

        Eigen::Quaternion<T> quat_calib_cam = cam_b_T_cam_a_.q.template cast<T>() * quat_calib;
        Eigen::Quaternion<T> quat_calib_lidar = quat_calib * lidar_b_T_lidar_a_.q.template cast<T>();

        Eigen::Quaternion<T> delta_q = quat_calib_cam * quat_calib_lidar.conjugate().normalized();

        // Compute the residuals
        // [orientation (3x1) ] =  [ 2 * delta_q(0:2)]
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(res);
        residuals.template block<3, 1>(0, 0) = T(2.0) * delta_q.vec();

        // Scale the residuals by constraint weights
        residuals.applyOnTheLeft((this->w_).template cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Matrix3d& weights, const PosePQ& cam_b_T_cam_a,
                                       const PosePQ& lidar_b_T_lidar_a) {
        return new ceres::AutoDiffCostFunction<CostFunctorInitNonLin, 3, 4>(
            new CostFunctorInitNonLin(weights, cam_b_T_cam_a, lidar_b_T_lidar_a));
    }

    PosePQ cam_b_T_cam_a_;
    PosePQ lidar_b_T_lidar_a_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename C>
struct CostFunctorPointReprojectionResidual : public CostFunctorWeightedReprojection2d<C> {
    CostFunctorPointReprojectionResidual(const Point2d& p_img_meas, const Matrix2d& weight, const Point3d& pt_3d,
                                         std::unique_ptr<C>& ceres_projector)
        : CostFunctorWeightedReprojection2d<C>(weight, ceres_projector), p_img_meas_(p_img_meas), pt_3d_(pt_3d) {}

    template <typename T>
    bool operator()(const T* const pos_cam_ptr, const T* const quat_cam_ptr, T* res) const {
        // Map the data arrays to Eigen Objects
        Eigen::Map<const Eigen::Quaternion<T>> quat_cam(quat_cam_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_cam(pos_cam_ptr);

        // Transform the 3d point into the camera frame
        Eigen::Matrix<T, 3, 1> pt_3d_in_cam_frame = quat_cam.conjugate() * (pt_3d_.template cast<T>() - pos_cam);

        // Project the 3d point in the camera frame into the image
        Eigen::Matrix<T, 2, 1> p_img_predicted(-1, -1);
        (*(this->ceres_projector_))(pt_3d_in_cam_frame.data(), p_img_predicted.data());

        // Determine the weighted residual (measured 2d point - predicted 2d point)
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(res);
        residuals.template block<2, 1>(0, 0) = p_img_predicted - p_img_meas_.template cast<T>();
        residuals.applyOnTheLeft((this->w_).template cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Point2d& p_img_meas, const Matrix2d& weight, const Point3d& pt_3d,
                                       std::unique_ptr<C>& ceres_projector) {
        return new ceres::AutoDiffCostFunction<CostFunctorPointReprojectionResidual, 2, 3, 4>(
            new CostFunctorPointReprojectionResidual(p_img_meas, weight, pt_3d, ceres_projector));
    }

    Point2d p_img_meas_;
    Point3d pt_3d_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace optimization_utils
