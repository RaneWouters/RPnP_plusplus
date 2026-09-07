#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "types.h"

namespace r2ppnp {
namespace internal {

Eigen::RowVectorXd param_r2_base(
    const Eigen::Vector3d& X0,
    const Eigen::Vector3d& X1,
    const Eigen::Matrix3Xd& X2);

struct TrialResult {
    std::vector<Eigen::Matrix3d> R_candidates;
    std::vector<Eigen::Vector3d> t_candidates;
    std::vector<double> scores;
};

TrialResult rpnp_trial(
    int i0, int i1,
    const Eigen::Matrix3Xd& XXw,
    const Eigen::Matrix3Xd& xxv,
    const Eigen::RowVectorXd& bin_r1,
    const Eigen::RowVectorXd& bin_r2,
    const Config& config);

void build_histogram(
    const std::vector<int>& indices,
    int nr1, int nr2,
    Eigen::MatrixXi& H);

void gaussian_smooth_circular(
    const Eigen::MatrixXd& H,
    Eigen::MatrixXd& H_smooth);

void dilate_3x3(
    const Eigen::MatrixXd& H,
    Eigen::MatrixXd& H_dilated);

std::vector<std::pair<int, int>> find_local_peaks(
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& H_dilated,
    double thH,
    double peak_ratio = 0.7);

} // namespace internal
} // namespace r2ppnp
