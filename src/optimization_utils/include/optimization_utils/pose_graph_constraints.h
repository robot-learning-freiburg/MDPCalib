#pragma once

#include <stdint.h>

#include <map>
#include <memory>
#include <string>

#include "optimization_utils/camera_model_pinhole.h"
#include "optimization_utils/forward.h"
#include "optimization_utils/measurement_types.h"

// In this class, we gather all information which are necessary to build the optimization problem.

namespace optimization_utils {

// This struct was generated, to have a common non-templated base class. This way all object of the derived classes can
// be placed inside of a common (e.g. STL) container
struct Constraint {
    Constraint() {}
    virtual ~Constraint() = 0;
};

inline Constraint::~Constraint() {}

template <typename T>
struct ConstraintData : public Constraint {
    std::shared_ptr<const T> measurement_ptr_;
    explicit ConstraintData(std::shared_ptr<const T> meas_obj_ptr) : measurement_ptr_(meas_obj_ptr) {}

    virtual ~ConstraintData() = 0;
};

template <typename T>
inline ConstraintData<T>::~ConstraintData() {}

struct PoseConstraintAsPoseDiff : public ConstraintData<MotionMeasurementAsPoseDiff> {
    PoseConstraintAsPoseDiff(std::shared_ptr<MotionMeasurementAsPoseDiff> meas_obj_ptr,
                             const std::shared_ptr<PosePQ> pose_A, const std::shared_ptr<PosePQ> pose_B)
        : ConstraintData(meas_obj_ptr), ptr_to_pose_A_(pose_A), ptr_to_pose_B_(pose_B) {}

    std::shared_ptr<PosePQ> ptr_to_pose_A_ = nullptr;
    std::shared_ptr<PosePQ> ptr_to_pose_B_ = nullptr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct RotationConstraint : public ConstraintData<std::map<std::string, MotionMeasurementAsPoseDiff>> {
    RotationConstraint(std::shared_ptr<std::map<std::string, MotionMeasurementAsPoseDiff>> meas_obj_ptr,
                       const std::shared_ptr<PosePQ> extrinsics)
        : ConstraintData(meas_obj_ptr), ptr_to_extrinsics_(extrinsics) {}
    std::shared_ptr<PosePQ> ptr_to_extrinsics_ = nullptr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct TranslationConstraint : public ConstraintData<std::map<std::string, MotionMeasurementAsPoseDiff>> {
    TranslationConstraint(std::shared_ptr<std::map<std::string, MotionMeasurementAsPoseDiff>> meas_obj_ptr,
                          const std::shared_ptr<PosePQ> extrinsics, const std::shared_ptr<double> scale)
        : ConstraintData(meas_obj_ptr), ptr_to_extrinsics_(extrinsics), scale_(scale) {}
    std::shared_ptr<PosePQ> ptr_to_extrinsics_ = nullptr;
    std::shared_ptr<double> scale_ = nullptr;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ReprojectionConstraint {
    const CameraModel& camera_model_;

    explicit ReprojectionConstraint(const CameraModel& camera_model) : camera_model_(camera_model) {}
};

struct ImgPointConstraint : public ConstraintData<ImgPointMeasurement>, public ReprojectionConstraint {
    ImgPointConstraint(std::shared_ptr<ImgPointMeasurement> meas_obj_ptr, const std::shared_ptr<PosePQ> pose,
                       const Point3d& associated_map_pt, const CameraModel& camera_model)
        : ConstraintData(meas_obj_ptr),
          ReprojectionConstraint(camera_model),
          ptr_to_pose_(pose),
          associated_map_pt_(associated_map_pt) {}

    std::shared_ptr<PosePQ> ptr_to_pose_ = nullptr;
    Point3d associated_map_pt_;
};

}  // namespace optimization_utils
