#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>

namespace r2ppnp {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kRansacMinDistImage = 7.0;
constexpr double kRansacMinDistWorld = 1e-3;

struct Config {
    // Normalized-image chord-distance inlier threshold. Pixel-coordinate
    // entry points derive this value from th_pixel and the focal lengths.
    double thv = 0.01;
    int nr1 = 30;
    double ransac_p = 0.7;
    int max_trials = 5000;
    double th_cons = 0.01;
    double thH_factor = 0.005;
    int min_thH = 5;
    double thH_ratio = 0.7;
    int num_peaks = 4;
    int gn_max_iter = 4;
    double gn_converge = 1e-4;
    int seed = 3;
    bool finalize = true;

    Config() = default;
};

struct PnPResult {
    bool success = false;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    int num_trials = 0;
    int num_inliers = 0;
    double score = 0.0;
    std::vector<bool> inliers;
    Eigen::VectorXd weights;
    Eigen::VectorXd errors;
    std::string message = "no valid pose found";
};

} // namespace r2ppnp
