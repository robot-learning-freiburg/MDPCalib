#pragma once

// ----- Own classes
#include "optimization_utils/forward.h"

// ----- STL includes
#include <vector>
#include <filesystem>
#include <iostream>
#include <cstdlib>

// ----- Boost includes
#include <boost/filesystem.hpp>

// ----- OpenCV and cv_bridge
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

namespace fs = std::filesystem;

namespace optimization_utils{


struct IOUtils{
    static uint16_t ctrLaunchFile;
    static uint16_t ctrConfigFile;
    static uint16_t ctrInitVisuFile;
    static uint16_t ctrRefineVisuFile;

    IOUtils();
    IOUtils(const std::string& pathBase, const std::string& runName);
    ~IOUtils(){}

    // Path attributes
    std::string pathRunDir_;
    std::string pathConfigDir_;
    std::string pathLaunchDir_;
    std::string pathResultsDir_;
    std::string pathVisualizationsDir_;

    // ToDo: Replace string by filesystem
    bool createDir(const std::string& path);

    void createRunDirectories();

    void copyFiles();

    std::vector<fs::path> get_all(fs::path const & root, std::string const & ext);

    void copyFile(const fs::path& pathFileSrc, const fs::path& pathDirDist);

    void writeResults(std::stringstream& stream);

    void writeImage(const sensor_msgs::ImageConstPtr& img_msg);

    PosePQ evaluate(const PosePQ& poseGt, const PosePQ& posePred);

};


}