#include "optimization_utils/camera_model.h"

namespace optimization_utils {

std::unique_ptr<ceres::CostFunction> CameraModel::GetImagePointToViewingRayFunction() const {
    throw std::runtime_error("Not implemented");
}

std::unique_ptr<ceres::CostFunction> CameraModel::GetWorldPointToImagePointFunction() const {
    throw std::runtime_error("Not implemented");
}

std::unique_ptr<ceres::CostFunction> CameraModel::GetWorldPointToImagePointFunction(
    const Eigen::Ref<const Eigen::Vector2d>& /*initialImagePoint*/) const {
    return GetWorldPointToImagePointFunction();
}

std::unique_ptr<ceres::CostFunctionToFunctor<2, 3>> CameraModel::GetWorldPointToImagePointFunctor() const {
    return std::make_unique<ceres::CostFunctionToFunctor<2, 3>>(GetWorldPointToImagePointFunction().release());
}

std::unique_ptr<ceres::CostFunctionToFunctor<2, 3>> CameraModel::GetWorldPointToImagePointFunctor(
    const Eigen::Ref<const Eigen::Vector2d>& initial_image_point) const {
    return std::make_unique<ceres::CostFunctionToFunctor<2, 3>>(
        GetWorldPointToImagePointFunction(initial_image_point).release());
}

}  // namespace optimization_utils
