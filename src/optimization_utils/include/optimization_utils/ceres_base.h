#pragma once

#include <ceres/ceres.h>
#include <ceres/problem.h>

#include <iostream>
#include <memory>

namespace optimization_utils {
class CeresBase {
   public:
    typedef std::unique_ptr<CeresBase> Uptr;
    typedef std::shared_ptr<CeresBase> Sptr;

    CeresBase() = delete;

    CeresBase(const ceres::Solver::Options& solver_options, const ceres::Problem::Options& problem_options)
        : ceres_problem_ptr_(new ceres::Problem()), solver_options_(solver_options), problem_options_(problem_options) {}

    // Copy Constructor
    CeresBase(const CeresBase& other)
        // cppcheck-suppress copyCtorPointerCopying
        : ceres_problem_ptr_(other.ceres_problem_ptr_),
          solver_options_(other.solver_options_),
          problem_options_(other.problem_options_),
          summary_(other.summary_) {}

    CeresBase& operator=(const CeresBase& other) {
        ceres_problem_ptr_ = other.ceres_problem_ptr_;
        solver_options_ = other.solver_options_;
        summary_ = other.summary_;
        return *this;
    }

    virtual ~CeresBase() {}

    // cppcheck-suppress unusedFunction
    void optimizeProblem() { ceres::Solve(solver_options_, ceres_problem_ptr_, &summary_); }

   protected:
    ceres::Problem* ceres_problem_ptr_;
    ceres::Solver::Options solver_options_;
    const ceres::Problem::Options& problem_options_;
    ceres::Solver::Summary summary_;

   private:
};

}  // namespace optimization_utils
