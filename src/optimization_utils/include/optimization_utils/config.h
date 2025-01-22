#pragma once

#include <ceres/ceres.h>
#include <ceres/types.h>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>

namespace optimization_utils {
struct Params {
    Params() = delete;
    explicit Params(const std::string& pathConfigFile) {
        YAML::Node node = YAML::LoadFile(pathConfigFile);
        FromConfig(node);
    }

    bool FromConfig(const YAML::Node& n) {
        // -----
        // Extrinsic calibration (ref to camera)
        // Translation
        t_x_ = n["t_x"].as<double>();
        t_y_ = n["t_y"].as<double>();
        t_z_ = n["t_z"].as<double>();
        // Rotation (Quaternion)
        q_w_ = n["q_w"].as<double>();
        q_x_ = n["q_x"].as<double>();
        q_y_ = n["q_y"].as<double>();
        q_z_ = n["q_z"].as<double>();

        // -----
        // Camera model parameters
        camera_model_type_ = n["camera_model_type"].as<std::string>();
        image_width_ = n["image_width"].as<uint16_t>();
        image_height_ = n["image_height"].as<uint16_t>();
        // intrinsic parameters
        f_x_ = n["f_x"].as<double>();
        f_y_ = n["f_y"].as<double>();
        c_x_ = n["c_x"].as<double>();
        c_y_ = n["c_y"].as<double>();

        return true;
    }

    // Extrinsics:
    double t_x_;
    double t_y_;
    double t_z_;
    double q_w_;
    double q_x_;
    double q_y_;
    double q_z_;

    // Intrinsics:
    std::string camera_model_type_;
    uint16_t image_width_;
    uint16_t image_height_;
    double f_x_;
    double f_y_;
    double c_x_;
    double c_y_;
};

class SolverOptions {
   public:
    SolverOptions() = delete;
    explicit SolverOptions(const std::string& path_config_file) {
        YAML::Node node = YAML::LoadFile(path_config_file);
        FromYaml(node);
    }

    bool FromYaml(const YAML::Node& n) {
        // Ceres solver options
        update_state_every_iteration_ = n["update_state_every_iteration"].as<bool>();
        minimizer_progress_to_stdout_ = n["minimizer_progress_to_stdout"].as<bool>();
        max_num_iterations_ = n["max_num_iterations"].as<uint16_t>();
        function_tolerance_ = n["function_tol"].as<double>();
        parameter_tolerance_ = n["parameter_tol"].as<double>();
        gradient_tolerance_ = n["gradient_tol"].as<double>();
        linear_solver_type_ = this->StringToLinearSolverType(n["linear_solver_type"].as<std::string>());
        return true;
    }

    // cppcheck-suppress unusedFunction
    ceres::Solver::Options AsCeresOptions() const {
        ceres::Solver::Options options;
        options.update_state_every_iteration = update_state_every_iteration_;
        options.max_num_iterations = max_num_iterations_;
        options.linear_solver_type = linear_solver_type_;  // ceres::SPARSE_NORMAL_CHOLESKY;

        options.parameter_tolerance = parameter_tolerance_;  // 1.e-20;
        options.function_tolerance = function_tolerance_;    // 1.e-20;
        options.gradient_tolerance = gradient_tolerance_;    // 1.e-20;
        options.minimizer_progress_to_stdout = minimizer_progress_to_stdout_;

        return options;
    }

   private:
    ceres::LinearSolverType StringToLinearSolverType(const std::string& val) {
        ceres::LinearSolverType ceres_solver;
        ceres::StringToLinearSolverType(val, &ceres_solver);
        return ceres_solver;
    }

    // Ceres solver options
    bool minimizer_progress_to_stdout_;
    uint64_t max_num_iterations_;
    double function_tolerance_;
    double parameter_tolerance_;
    double gradient_tolerance_;
    bool update_state_every_iteration_;
    ceres::LinearSolverType linear_solver_type_;
};

}  // namespace optimization_utils
