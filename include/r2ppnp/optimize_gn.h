#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "types.h"

namespace r2ppnp {
namespace internal {

struct GNResult {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::VectorXd w;
    Eigen::VectorXd err;
    bool success = false;
};

GNResult sw_gn(
    const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& t0,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    double thv,
    const Eigen::VectorXd& err_in,
    int weight_order,
    int max_iter,
    double converge_threshold);

GNResult dsw_gn(
    const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& t0,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    double thv,
    const std::vector<int>& weight_orders,
    int max_iter,
    double converge_threshold,
    const Eigen::VectorXd& err_in = {});

GNResult optimize_gn(
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    const Eigen::Vector3d& t0,
    double thv,
    const Eigen::VectorXd& w_in,
    const Eigen::VectorXd& err_in,
    int weight_order,
    int max_iter,
    double converge_threshold);

} // namespace internal
} // namespace r2ppnp
