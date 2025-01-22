#include "optimization_utils/visualizer.h"

namespace optimization_utils{
namespace visualizer{
void visualizeProjectedPoints(const cv::Mat& img,
                              const Points2d& projectedPoints,
                              const double& scaleFactor,
                              const std::string& windowName,
                              const cv::Scalar& color,
                              const int& radiusCircle,
                              const int& lineThickness){
    drawPointsIntoImage(img, projectedPoints, color, radiusCircle, lineThickness);
    cv::Mat outImg;
    cv::resize(img, outImg, cv::Size(), scaleFactor, scaleFactor);

    showImage(outImg, windowName);
}

void drawPointsIntoImage(const cv::Mat& img,
                         const Points2d& projectedPoints,
                         const cv::Scalar& color,
                         const int& radiusCircle,
                         const int& lineThickness){
    for(auto p : projectedPoints){
        cv::Point cvP(static_cast<int16_t>(p[0]), static_cast<int>(p[1]));
        cv::circle(img, cvP, radiusCircle, color, cv::FILLED);
    }
}


void showImage(cv::Mat& inImg, const std::string& winName){
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::imshow(winName, inImg);
    cv::waitKey(0);
    cv::destroyWindow(winName);
}

void showImage(cv::Mat& inImg, const double& scaleFactor, const std::string& winName){
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::Mat outImg;
    cv::resize(inImg, outImg, cv::Size(), scaleFactor, scaleFactor);
    cv::imshow(winName, outImg);
    cv::waitKey(0);
    cv::destroyWindow(winName);
}

Points2d reprojectLidarPoints(const Correspondence2D3D& correspondences,
                              std::shared_ptr<const PosePQ> visuPose,
                              const CameraModelPinhole& camModel){

    Points2d reprojectedLidarPoints;

    for (auto correspondence_item = std::next(correspondences.begin()); correspondence_item != correspondences.end();
         correspondence_item++) {
        // Each correspondence_item holds a set of 2D-3D points. Go through each of them and apply the following steps
        auto& correspondences_pcl = correspondence_item->second;

        for (uint16_t i = 0; i < correspondences_pcl.cols(); i++) {
            // 1.) Create image point measurement
            // auto pt2d = correspondences_img.col(i).cast<double>();
            auto pt3d = correspondences_pcl.col(i).cast<double>();
            auto pt3dCamFrame = visuPose->q.conjugate() * (pt3d - visuPose->p);
            auto ptImgPoint = camModel.GetImagePoint(pt3dCamFrame);
            if(camModel.IsVisible(ptImgPoint)){
                reprojectedLidarPoints.push_back(ptImgPoint);
            }
        }
    }
    return reprojectedLidarPoints;
}

}}

