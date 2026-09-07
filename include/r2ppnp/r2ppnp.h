#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include "types.h"

namespace r2ppnp {

// Estimate a world-to-camera pose from normalized image coordinates. Throws
// std::invalid_argument when inputs or configuration values violate the public
// API contract. Algorithmic failure is reported as PnPResult::success=false.
PnPResult r2ppnp(
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& norm_points,
    const Config& config = Config{});

// Pixel-coordinate entry point. Pixel observations must already be
// undistorted. PnPResult::errors are returned in pixels.
PnPResult r2ppnp_from_pixels(
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& pixel_points,
    const Eigen::Matrix3d& K,
    double th_pixel = 10.0,
    const Config& config = Config{});

// Run the paper's final dynamic soft-weighted Gauss-Newton refinement from an
// existing proper rotation and translation.
PnPResult refine(
    const Eigen::Matrix3Xd& world_points,
    const Eigen::Matrix2Xd& pixel_points,
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& R_init,
    const Eigen::Vector3d& t_init,
    double th_pixel = 10.0,
    const Config& config = Config{});

} // namespace r2ppnp
