#include "optimization_utils/io_utils.h"

namespace optimization_utils {

uint16_t IOUtils::ctrLaunchFile = 0;
uint16_t IOUtils::ctrConfigFile = 0;
uint16_t IOUtils::ctrInitVisuFile = 0;
uint16_t IOUtils::ctrRefineVisuFile = 0;

IOUtils::IOUtils() {}

IOUtils::IOUtils(const std::string& pathBase, const std::string& runName)
    : pathRunDir_(pathBase + "/experiments" + "/" + runName),
      pathConfigDir_(pathRunDir_ + "/configs"),
      pathLaunchDir_(pathRunDir_ + "/launch"),
      pathResultsDir_(pathRunDir_ + "/results"),
      pathVisualizationsDir_(pathRunDir_ + "/visualizations") {}

// ToDo: Put this into the Optimizer Constructor
void IOUtils::createRunDirectories() {
    // Create run directory
    createDir(pathRunDir_);

    // Create subdirectories
    createDir(pathConfigDir_);
    createDir(pathLaunchDir_);
    createDir(pathResultsDir_);
    createDir(pathVisualizationsDir_);
}

void IOUtils::copyFiles() {
    // Root dir should be pointing to the top src dir, as parents: src->optimization_utils->src
    // ToDo: This is a really bad way to to go as we assume pathRunDir to be set to the path of the workspace
    fs::path rootDir = fs::path(pathRunDir_).parent_path().parent_path();

    // Copy and paste config files
    // std::vector<fs::path> pathsConfigFiles = get_all(rootDir, ".yaml");
    // for(auto& pathCfg : pathsConfigFiles){
    //     this->copyFile(pathCfg, pathConfigDir_);
    //     std::string padding = std::string(2 - std::to_string(IOUtils::ctrConfigFile).length(), '0');
    //     fs::path filename = pathCfg.filename();
    //     fs::path ext = pathCfg.extension();
    //     std::filesystem::rename(pathConfigDir_ / filename,
    //                             (pathConfigDir_ / filename).replace_filename(filename.stem().string() +
    //                             std::string("_" + padding + std::to_string(IOUtils::ctrConfigFile++)) +
    //                             ext.string()));
    // }

    // // Copy and paste launch files
    // std::vector<fs::path> pathsLaunchFiles = get_all(rootDir, ".launch");
    // for(auto& pathLaunch : pathsLaunchFiles){
    //     this->copyFile(pathLaunch, pathLaunchDir_);
    //     std::string padding = std::string(2 - std::to_string(IOUtils::ctrLaunchFile).length(), '0');
    //     fs::path filename = pathLaunch.filename();
    //     fs::path ext = pathLaunch.extension();
    //     std::filesystem::rename(pathLaunchDir_ / filename,
    //                             (pathLaunchDir_ / filename).replace_filename(filename.stem().string() +
    //                             std::string("_" + padding + std::to_string(IOUtils::ctrLaunchFile++)) +
    //                             ext.string()));
    // }
}

bool IOUtils::createDir(const std::string& pathBase) {
    fs::path pathDir(pathBase);

    if (fs::create_directories(pathDir)) {
        std::cout << "Directory " << pathDir << " created." << std::endl;
        return true;
    } else if (fs::exists(pathDir)) {
        std::cerr << "Directory " << pathDir << " already exists.";
        std::exit(EXIT_FAILURE);
    } else {
        std::cerr << "The directory " << pathDir
                  << " does not exist and could not be created. Please check the permissions for the specified path.";
        std::exit(EXIT_FAILURE);
    }
    // This function will actually never return false.
    return false;
}

std::vector<fs::path> IOUtils::get_all(fs::path const& root, std::string const& ext) {
    std::vector<fs::path> paths;

    if (fs::exists(root) && fs::is_directory(root)) {
        for (auto const& entry : fs::recursive_directory_iterator(root)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ext) paths.emplace_back(entry.path());
        }
    }

    return paths;
}

void IOUtils::copyFile(const fs::path& pathFileSrc, const fs::path& pathDirDist) {
    auto target = pathDirDist / pathFileSrc.filename();
    try {
        fs::copy_file(pathFileSrc, target, fs::copy_options::overwrite_existing);
    } catch (std::exception& e) {
        std::cout << e.what();
    }
}

void IOUtils::writeResults(std::stringstream& stream) {
    // ToDo: Why assing fs to std::string?? Fix this
    std::string pathResultsFile = fs::path(this->pathResultsDir_ + "/results.txt");

    std::ofstream resultsFile;
    resultsFile.open(pathResultsFile, std::fstream::in | std::fstream::out | std::fstream::app);

    if (resultsFile.is_open()) {
        resultsFile << std::fixed << std::setprecision(10) << stream.rdbuf();
    }
    resultsFile.close();
}

void IOUtils::writeImage(const sensor_msgs::ImageConstPtr& img_msg) {
    // Convert ROS sensor_msgs::Image into cv::Mat
    cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img_msg);
    std::cout << imgPtr->header << std::endl;
    cv::Mat img = imgPtr->image;

    // Generate filename
    std::string padding = std::string(2 - std::to_string(IOUtils::ctrInitVisuFile).length(), '0');
    std::string filename = "visu_init_" + padding + std::to_string(IOUtils::ctrInitVisuFile++) + ".png";

    // Save image with reprojections
    cv::imwrite(this->pathVisualizationsDir_ + "/" + filename, img);
}

}  // namespace optimization_utils
