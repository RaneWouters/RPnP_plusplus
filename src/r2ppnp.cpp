#include "r2ppnp/r2ppnp.h"

#include "r2ppnp/hough_voting.h"
#include "r2ppnp/optimize_gn.h"
#include "r2ppnp/utils.h"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace r2ppnp {
namespace {

constexpr Eigen::Index kMinimumPointCount = 6;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void validate_config(const Config& config) {
    require(std::isfinite(config.thv) && config.thv > 0.0,
            "config.thv must be finite and greater than zero");
    require(config.nr1 >= 3 && config.nr1 <= 360,
            "config.nr1 must be in the interval [3, 360]");
    require(std::isfinite(config.ransac_p) && config.ransac_p > 0.0 && config.ransac_p < 1.0,
            "config.ransac_p must be in the open interval (0, 1)");
    require(config.max_trials > 0 && config.max_trials <= 10000000,
            "config.max_trials must be in the interval [1, 10000000]");
    require(std::isfinite(config.th_cons) && config.th_cons > 0.0,
            "config.th_cons must be finite and greater than zero");
    require(std::isfinite(config.thH_factor) && config.thH_factor >= 0.0 &&
                config.thH_factor <= 1.0,
            "config.thH_factor must be in the interval [0, 1]");
    require(config.min_thH >= 1, "config.min_thH must be at least 1");
    require(std::isfinite(config.thH_ratio) && config.thH_ratio > 0.0 && config.thH_ratio <= 1.0,
            "config.thH_ratio must be in the interval (0, 1]");
    require(config.num_peaks >= 1 && config.num_peaks <= 100,
            "config.num_peaks must be in the interval [1, 100]");
    require(config.gn_max_iter >= 0 && config.gn_max_iter <= 100,
            "config.gn_max_iter must be in the interval [0, 100]");
    require(std::isfinite(config.gn_converge) && config.gn_converge > 0.0,
            "config.gn_converge must be finite and greater than zero");
}

void validate_correspondences(
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& image_points)
{
    require(world_points.cols() == image_points.cols(),
            "world_points and image_points must contain the same number of columns");
    require(world_points.cols() >= kMinimumPointCount,
            "RPnP++ requires at least 6 correspondences");
    require(world_points.cols() <= std::numeric_limits<int>::max(),
            "the number of correspondences exceeds the supported integer range");
    require(world_points.allFinite(), "world_points must contain only finite values");
    require(image_points.allFinite(), "image_points must contain only finite values");
}

void validate_camera_matrix(const Eigen::Matrix3d& K) {
    require(K.allFinite(), "K must contain only finite values");
    require(K(2, 2) > 1e-12, "K(2,2) must be positive");
    require(K(0, 0) / K(2, 2) > 0.0 && K(1, 1) / K(2, 2) > 0.0,
            "K must have positive homogeneous-normalized fx and fy");
    Eigen::FullPivLU<Eigen::Matrix3d> lu(K);
    require(lu.rank() == 3, "K must be invertible");
}

void initialize_diagnostics(PnPResult& result, Eigen::Index n) {
    result.inliers.assign(static_cast<std::size_t>(n), false);
    result.weights = Eigen::VectorXd::Zero(n);
    result.errors = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
    result.num_inliers = 0;
    result.score = 0.0;
}

void populate_normalized_diagnostics(
    PnPResult& result,
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix3Xd& rays,
    double threshold)
{
    Eigen::VectorXd depth;
    project_d_err(result.R, result.t, 1.0, world_points, rays, depth, result.errors);
    const Eigen::Index n = world_points.cols();
    result.inliers.assign(static_cast<std::size_t>(n), false);
    result.weights = Eigen::VectorXd::Zero(n);
    result.num_inliers = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const bool inlier = std::isfinite(result.errors(i)) && result.errors(i) < threshold;
        result.inliers[static_cast<std::size_t>(i)] = inlier;
        result.weights(i) = inlier ? 1.0 : 0.0;
        result.num_inliers += inlier ? 1 : 0;
    }
    result.score = static_cast<double>(result.num_inliers) / static_cast<double>(n);
}

void populate_pixel_diagnostics(
    PnPResult& result,
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& pixel_points,
    const Eigen::Matrix3d& K,
    double threshold)
{
    const Eigen::Index n = world_points.cols();
    result.inliers.assign(static_cast<std::size_t>(n), false);
    result.weights = Eigen::VectorXd::Zero(n);
    result.errors = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
    result.num_inliers = 0;

    const Eigen::Matrix3Xd camera_points =
        result.R * world_points + result.t.replicate(1, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        // Points behind the calibrated camera are never valid reprojections.
        if (!camera_points.col(i).allFinite() || camera_points(2, i) <= 1e-12) {
            continue;
        }
        const Eigen::Vector3d projected = K * camera_points.col(i);
        if (!projected.allFinite() || std::abs(projected(2)) <= 1e-12) {
            continue;
        }
        const Eigen::Vector2d pixel = projected.head<2>() / projected(2);
        const double error = (pixel_points.col(i) - pixel).norm();
        result.errors(i) = error;
        const bool inlier = std::isfinite(error) && error < threshold;
        result.inliers[static_cast<std::size_t>(i)] = inlier;
        result.weights(i) = inlier ? 1.0 : 0.0;
        result.num_inliers += inlier ? 1 : 0;
    }
    result.score = static_cast<double>(result.num_inliers) / static_cast<double>(n);
}

bool valid_pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    if (!R.allFinite() || !t.allFinite()) {
        return false;
    }
    const double orthogonality_error =
        (R.transpose() * R - Eigen::Matrix3d::Identity()).norm();
    return orthogonality_error < 1e-6 && std::abs(R.determinant() - 1.0) < 1e-6;
}

double normalized_pixel_threshold(const Eigen::Matrix3d& K, double th_pixel) {
    // The geometric mean treats anisotropic focal lengths symmetrically while
    // retaining the MATLAB value exactly when fx == fy. Dividing by K(2,2)
    // makes this conversion invariant to homogeneous scaling of K.
    const double fx = K(0, 0) / K(2, 2);
    const double fy = K(1, 1) / K(2, 2);
    return th_pixel / std::sqrt(fx * fy);
}

} // namespace

PnPResult r2ppnp(
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& norm_points,
    const Config& config)
{
    validate_config(config);
    validate_correspondences(world_points, norm_points);

    PnPResult best;
    const Eigen::Index npt = world_points.cols();
    initialize_diagnostics(best, npt);

    const Eigen::Matrix3Xd xxv = xxn2xxv(norm_points);
    require(xxv.allFinite(), "normalized image rays must be finite");

    const int nr1 = config.nr1;
    const int nr2 = nr1 * 2;
    const double step_r1 = kPi / static_cast<double>(nr1);
    const double step_r2 = 2.0 * kPi / static_cast<double>(nr2);

    Eigen::RowVectorXd bin_r1(nr1);
    for (int i = 0; i < nr1; ++i) {
        bin_r1(i) = step_r1 * 0.5 - kPi * 0.5 + step_r1 * i;
    }

    Eigen::RowVectorXd bin_r2(nr2);
    for (int i = 0; i < nr2; ++i) {
        bin_r2(i) = step_r2 * 0.5 - kPi + step_r2 * i;
    }

    std::mt19937 rng(config.seed >= 0 ? static_cast<unsigned int>(config.seed)
                                      : std::random_device{}());
    std::uniform_int_distribution<int> dist(0, static_cast<int>(npt) - 1);

    const std::int64_t pair_count =
        static_cast<std::int64_t>(npt) * static_cast<std::int64_t>(npt - 1) / 2;
    const int max_trials = static_cast<int>(std::min<std::int64_t>(config.max_trials, pair_count));
    int cnt_trial = 0;
    double target_trials = static_cast<double>(max_trials);

    // Keep the target as a double, as in MATLAB. With an integer counter this
    // naturally executes ceil(target_trials) trials rather than truncating it.
    while (static_cast<double>(cnt_trial) < target_trials) {
        ++cnt_trial;

        // Match the reference MATLAB semantics: one pair is sampled per
        // RANSAC trial. Invalid/degenerate pairs consume the trial and are
        // skipped, so degenerate inputs always terminate.
        const int i0 = dist(rng);
        const int i1 = dist(rng);
        const double image_distance = (xxv.col(i1) - xxv.col(i0)).norm();
        const double world_distance = (world_points.col(i1) - world_points.col(i0)).norm();
        if (i0 == i1 || !std::isfinite(image_distance) || !std::isfinite(world_distance) ||
            image_distance < config.thv * kRansacMinDistImage ||
            world_distance < kRansacMinDistWorld) {
            continue;
        }

        internal::TrialResult trial = internal::rpnp_trial(
            i0, i1, world_points, xxv, bin_r1, bin_r2, config);

        for (std::size_t k = 0; k < trial.scores.size(); ++k) {
            const double score = trial.scores[k] / static_cast<double>(npt);
            if (!std::isfinite(score) || score <= best.score ||
                !valid_pose(trial.R_candidates[k], trial.t_candidates[k])) {
                continue;
            }

            best.R = trial.R_candidates[k];
            best.t = trial.t_candidates[k];
            best.score = score;
            best.success = true;

            double inlier_probability = std::max(score, 5.0 / static_cast<double>(npt));
            const double miss_probability = std::min(
                std::max(1.0 - inlier_probability * inlier_probability, 0.0001), 0.9999);
            const double updated_trials =
                std::log(1.0 - config.ransac_p) / std::log(miss_probability);
            if (std::isfinite(updated_trials)) {
                target_trials = std::min(static_cast<double>(max_trials), updated_trials);
            }
        }
    }

    best.num_trials = cnt_trial;
    if (!best.success) {
        best.message = "no valid pose found";
        return best;
    }

    if (config.finalize) {
        const std::vector<int> weight_orders = {3, 4, -1};
        internal::GNResult refined = internal::dsw_gn(
            best.R, best.t, world_points, xxv, config.thv, weight_orders,
            config.gn_max_iter, config.gn_converge);
        if (refined.success && valid_pose(refined.R, refined.t)) {
            best.R = refined.R;
            best.t = refined.t;
        }
    }

    populate_normalized_diagnostics(best, world_points, xxv, config.thv);
    best.success = valid_pose(best.R, best.t) && best.num_inliers >= 5;
    best.message = best.success ? "success" : "pose refinement produced an invalid result";
    return best;
}

PnPResult r2ppnp_from_pixels(
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& pixel_points,
    const Eigen::Matrix3d& K,
    double th_pixel,
    const Config& config)
{
    validate_config(config);
    validate_correspondences(world_points, pixel_points);
    validate_camera_matrix(K);
    require(std::isfinite(th_pixel) && th_pixel > 0.0,
            "th_pixel must be finite and greater than zero");

    const Eigen::Matrix2Xd normalized_points = xx2xxn(K, pixel_points);
    require(normalized_points.allFinite(), "normalizing pixel_points produced non-finite values");

    Config local_config = config;
    local_config.thv = normalized_pixel_threshold(K, th_pixel);
    PnPResult result = r2ppnp(world_points, normalized_points, local_config);
    if (result.success) {
        populate_pixel_diagnostics(result, world_points, pixel_points, K, th_pixel);
        result.success = result.num_inliers >= 5;
        result.message = result.success ? "success" : "pose has fewer than 5 pixel-space inliers";
    }
    return result;
}

PnPResult refine(
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& pixel_points,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& R_init,
    const Eigen::Vector3d& t_init,
    double th_pixel,
    const Config& config)
{
    validate_config(config);
    validate_correspondences(world_points, pixel_points);
    validate_camera_matrix(K);
    require(std::isfinite(th_pixel) && th_pixel > 0.0,
            "th_pixel must be finite and greater than zero");
    require(R_init.allFinite() && t_init.allFinite(), "R_init and t_init must contain only finite values");
    require((R_init.transpose() * R_init - Eigen::Matrix3d::Identity()).norm() < 1e-3 &&
                std::abs(R_init.determinant() - 1.0) < 1e-3,
            "R_init must be a proper 3D rotation matrix");

    const Eigen::Matrix2Xd normalized_points = xx2xxn(K, pixel_points);
    const Eigen::Matrix3Xd rays = xxn2xxv(normalized_points);
    const double threshold = normalized_pixel_threshold(K, th_pixel);
    const std::vector<int> weight_orders = {3, 4, -1};

    internal::GNResult refined = internal::dsw_gn(
        R_init, t_init, world_points, rays, threshold, weight_orders,
        config.gn_max_iter, config.gn_converge);

    PnPResult result;
    result.R = R_init;
    result.t = t_init;
    initialize_diagnostics(result, world_points.cols());
    if (!refined.success || !valid_pose(refined.R, refined.t)) {
        populate_pixel_diagnostics(result, world_points, pixel_points, K, th_pixel);
        result.message = "refinement failed: fewer than 5 inliers or degenerate optimization";
        return result;
    }

    result.R = refined.R;
    result.t = refined.t;
    populate_pixel_diagnostics(result, world_points, pixel_points, K, th_pixel);
    result.success = result.num_inliers >= 5;
    result.message = result.success ? "success" : "refined pose has fewer than 5 pixel-space inliers";
    return result;
}

} // namespace r2ppnp
