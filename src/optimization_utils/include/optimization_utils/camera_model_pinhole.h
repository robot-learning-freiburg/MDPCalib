#pragma once

#include <memory>

#include "optimization_utils/camera_model.h"

namespace optimization_utils {

class CameraModelPinhole : public CameraModel {
   public:
    static const ModelType TYPE = ModelType::PINHOLE;

    typedef std::unique_ptr<CameraModelPinhole> Uptr;
    typedef std::shared_ptr<CameraModelPinhole> Sptr;

    CameraModelPinhole(const int& img_width, const int& img_height, const double& fx, const double& fy, const double& cx,
                       const double& cy);

    virtual ~CameraModelPinhole() {}

    Eigen::Vector2d GetImagePoint(const Eigen::Vector3d& p3d) const override;

    Eigen::Vector3d GetViewingRay(const Eigen::Vector2d& image_point) const override;

    std::unique_ptr<ceres::CostFunction> GetWorldPointToImagePointFunction() const override;

    void GetImageSize(int& img_width, int& img_height) const override;

    bool IsVisible(const Eigen::Vector2d& pt2d) const override;

    ModelType GetType() const override;

    const Eigen::Matrix<double, 4, 1>& GetParams() const;

    struct PinholeGetImagePoint {
        PinholeGetImagePoint() {}

        template <typename T, typename T1, typename T2, typename T3>
        bool operator()(T, T1&& intrinsics, T2&& p3d, T3&& image_point) const {
            if (p3d[2] <= static_cast<T>(0)) return false;

            const auto& fx = intrinsics[0];
            const auto& fy = intrinsics[1];
            const auto& cx = intrinsics[2];
            const auto& cy = intrinsics[3];

            image_point[0] = p3d[0] / p3d[2] * fx + cx;
            image_point[1] = p3d[1] / p3d[2] * fy + cy;
            return true;
        }
    };

   private:
    class CostFunctorForward {
       public:
        explicit CostFunctorForward(const CameraModelPinhole& camera_model);

        template <typename T>
        bool operator()(const T* point_3d_raw, T* image_point_raw) const;

       private:
        const CameraModelPinhole& camera_model_;
    };

    int img_width_;
    int img_height_;
    Eigen::Matrix<double, 4, 1> intrinsics_;  // 0: fx, 1: fy, 2: cx, 3:fy
};
}  // namespace optimization_utils
