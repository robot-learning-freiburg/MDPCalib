#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "optimization_utils/forward.h"

namespace optimization_utils {

enum class MeasType { kPoseDiff, kImagePoint};

struct Measurement {
    virtual MeasType GetType() = 0;
    virtual ~Measurement() {}
};

template <typename T1, typename T2>
struct MeasurementData : public Measurement {
    const T1 quantity_;
    const T2 information_matrix_;

    MeasType GetType() override = 0;

    MeasurementData(const T1& quantity, const T2& information_matrix)
        : quantity_(quantity), information_matrix_(information_matrix) {}
    virtual ~MeasurementData() {}
};

struct ImgPointMeasurement : public MeasurementData<Point2d, Matrix2d> {
    ImgPointMeasurement(const Point2d& quantity, const Matrix2d& information_matrix)
        : MeasurementData<Point2d, Matrix2d>(quantity, information_matrix) {}

    ~ImgPointMeasurement() override {}
    MeasType GetType() override { return MeasType::kImagePoint; }
};

struct MotionMeasurementAsPoseDiff : public MeasurementData<PosePQ, Matrix6d> {
    MotionMeasurementAsPoseDiff(const PosePQ& quantity, const Matrix6d& information_matrix)
        : MeasurementData<PosePQ, Matrix6d>(quantity, information_matrix) {}

    MotionMeasurementAsPoseDiff(const PosePQ& pose_A, const PosePQ& pose_B, const Matrix6d& information_matrix)
        :  // Calculate the pose diff from both poses
          MeasurementData<PosePQ, Matrix6d>(
              PosePQ(pose_A.q.conjugate() * (pose_B.p - pose_A.p), pose_A.q.conjugate() * pose_B.q), information_matrix) {}

    ~MotionMeasurementAsPoseDiff() override {}

    MeasType GetType() override { return MeasType::kPoseDiff; }
};


}  // namespace optimization_utils
