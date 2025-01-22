#include "optimization_utils/camera_model_pinhole.h"

namespace optimization_utils {

// : Args list, initalize parameters
CameraModelPinhole::CameraModelPinhole(const int& img_width, const int& img_height, const double& fx, const double& fy,
                                       const double& cx, const double& cy)
    : img_width_(img_width), img_height_(img_height), intrinsics_(Eigen::Matrix<double, 4, 1>(fx, fy, cx, cy)) {}

Eigen::Vector2d CameraModelPinhole::GetImagePoint(const Eigen::Vector3d& p3d) const {
    Eigen::Vector2d image_point;
    const auto& fx = intrinsics_[0];
    const auto& fy = intrinsics_[1];
    const auto& cx = intrinsics_[2];
    const auto& cy = intrinsics_[3];

    image_point[0] = p3d[0] / p3d[2] * fx + cx;
    image_point[1] = p3d[1] / p3d[2] * fy + cy;

    return image_point;
}

Eigen::Vector3d CameraModelPinhole::GetViewingRay(const Eigen::Vector2d& image_point) const {
    Eigen::Vector3d ray;
    const auto& fx = intrinsics_[0];
    const auto& fy = intrinsics_[1];
    const auto& cx = intrinsics_[2];
    const auto& cy = intrinsics_[3];

    ray[0] = (image_point[0] - cx) / fx;
    ray[1] = (image_point[1] - cy) / fy;
    ray[2] = 1.0;

    double norm = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);

    ray[0] /= norm;
    ray[1] /= norm;
    ray[2] /= norm;

    return ray;
}

void CameraModelPinhole::GetImageSize(int& img_width, int& img_height) const {
    img_width = img_width_;
    img_height = img_height_;
}

bool CameraModelPinhole::IsVisible(const Eigen::Vector2d& pt2d) const {
    if (0 <= pt2d(0) && pt2d(0) <= img_width_ && 0 <= pt2d(1) && pt2d(1) <= img_height_) return true;

    return false;
}

ModelType CameraModelPinhole::GetType() const { return TYPE; }

const Eigen::Matrix<double, 4, 1>& CameraModelPinhole::GetParams() const { return intrinsics_; }

CameraModelPinhole::CostFunctorForward::CostFunctorForward(const CameraModelPinhole& camera_model)
    : camera_model_(camera_model) {}

template <typename T>
bool CameraModelPinhole::CostFunctorForward::operator()(const T* point_3d_raw, T* image_point_raw) const {
    Eigen::Map<Eigen::Matrix<T, 2, 1>> image_point(image_point_raw);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_3d(point_3d_raw);

    PinholeGetImagePoint GetImagePoint;
    return GetImagePoint(T(), camera_model_.intrinsics_, point_3d, image_point);
}

std::unique_ptr<ceres::CostFunction> CameraModelPinhole::GetWorldPointToImagePointFunction() const {
    return std::make_unique<ceres::AutoDiffCostFunction<CostFunctorForward, 2, 3>>(new CostFunctorForward(*this));
}
}  // namespace optimization_utils
