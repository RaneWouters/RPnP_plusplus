#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "types.h"

namespace r2ppnp {

Eigen::Matrix2Xd xx2xxn(const Eigen::Matrix3d& K, const Eigen::Matrix2Xd& xx);

Eigen::Matrix2Xd xxn2xx(const Eigen::Matrix3d& K, const Eigen::Matrix2Xd& xxn);

Eigen::Matrix3Xd xxn2xxv(const Eigen::Matrix2Xd& xxn);

void project_d_err(
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    double s,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    Eigen::VectorXd& d,
    Eigen::VectorXd& err);

Eigen::VectorXd calc_weight(
    const Eigen::VectorXd& err,
    double th,
    int weight_order);

std::vector<bool> calcInliers(
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix2Xd& x,
    const Eigen::Matrix3d& K,
    double th_pixel);

double calc_dR(const Eigen::Matrix3d& R_gt, const Eigen::Matrix3d& R);
double calc_dt(const Eigen::Vector3d& t_gt, const Eigen::Vector3d& t);

double safe_acos(double x);
Eigen::Vector3d xcross_vec(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
Eigen::Vector3d xnormalize_vec(const Eigen::Vector3d& X);
Eigen::Matrix3Xd xnormalize(const Eigen::Matrix3Xd& X);
double xnorm(const Eigen::Vector3d& v);
Eigen::RowVectorXd xnorm_colwise(const Eigen::Matrix3Xd& X);
double clip(double x, double lo, double hi);
Eigen::RowVectorXd clip_vec(const Eigen::RowVectorXd& x, double lo, double hi);

double normalize_angle(double a);
void normalize_angles_in_place(Eigen::RowVectorXd& angles);

Eigen::Vector3d weighted_mean(const Eigen::Matrix3Xd& X, const Eigen::VectorXd& w);

} // namespace r2ppnp
