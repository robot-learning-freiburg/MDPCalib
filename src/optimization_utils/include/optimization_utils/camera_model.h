#pragma once

#include <ceres/ceres.h>

#include <eigen3/Eigen/Core>
#include <memory>

#include "optimization_utils/ceres_base.h"

namespace optimization_utils {

enum class ModelType { PINHOLE };

class CameraModel {
   public:
    typedef std::unique_ptr<CameraModel> Uptr;
    typedef std::shared_ptr<CameraModel> Sptr;

    CameraModel() {}
    virtual ~CameraModel() {}

    virtual Eigen::Vector2d GetImagePoint(const Eigen::Vector3d& p3d) const = 0;

    virtual Eigen::Vector3d GetViewingRay(const Eigen::Vector2d& image_point) const = 0;

    // Interface for Ceres
    virtual std::unique_ptr<ceres::CostFunction> GetImagePointToViewingRayFunction() const;

    virtual std::unique_ptr<ceres::CostFunction> GetWorldPointToImagePointFunction() const;

    virtual std::unique_ptr<ceres::CostFunction> GetWorldPointToImagePointFunction(
    const Eigen::Ref<const Eigen::Vector2d>&  initial_image_point) const;

    virtual std::unique_ptr<ceres::CostFunctionToFunctor<2, 3>> GetWorldPointToImagePointFunctor() const;

    virtual std::unique_ptr<ceres::CostFunctionToFunctor<2, 3>> GetWorldPointToImagePointFunctor(
        const Eigen::Ref<const Eigen::Vector2d>&  initial_image_point) const;

    virtual bool IsVisible(const Eigen::Vector2d& pt2d) const = 0;

    virtual void GetImageSize(int& img_width, int& img_height) const = 0;

    virtual ModelType GetType() const = 0;
};

}  // namespace optimization_utils
