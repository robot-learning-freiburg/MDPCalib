#pragma once

// Own classes
#include "optimization_utils/forward.h"
#include "optimization_utils/camera_model_pinhole.h"

// Opencv includes
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/persistence.hpp>

namespace optimization_utils {
namespace visualizer {

// ToDo: Change namings, i.e. scaleFactor -> scale_factor
void showImage(cv::Mat& inImg, const std::string& winName = "No custom name");

void showImage(cv::Mat& inImg, const double& scaleFactor, const std::string& winName = "No custom name");

void visualizeProjectedPoints(const cv::Mat& img,
                              const Points2d& projectedPoints,
                              const double& scaleFactor = 1,
                              const std::string& windowName = "No custom name",
                              const cv::Scalar& color = CV_RGB(255, 0, 0),
                              const int& radiusCircle = 1,
                              const int& lineThickness = 1);

Points2d reprojectLidarPoints(const Correspondence2D3D&, std::shared_ptr<const PosePQ> visuPose,
                              const CameraModelPinhole& camModel);


void drawPointsIntoImage(const cv::Mat& img, const Points2d& projectedPoints,
                         const cv::Scalar& color = CV_RGB(255, 0, 0), const int& radiusCircle = 1,
                         const int& lineThickness = 1);
}}

