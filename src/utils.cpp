#include "r2ppnp/utils.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>

namespace r2ppnp {

Eigen::Matrix2Xd xx2xxn(const Eigen::Matrix3d& K, const Eigen::Matrix2Xd& xx) {
    Eigen::Index n = xx.cols();
    Eigen::Matrix3Xd xxh(3, n);
    xxh.topRows(2) = xx;
    xxh.row(2).setOnes();
    Eigen::Matrix3Xd xn3 = K.fullPivLu().solve(xxh);
    Eigen::Matrix2Xd result(2, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double scale = xn3(2, i);
        if (!std::isfinite(scale) || std::abs(scale) <= 1e-12) {
            result.col(i).setConstant(std::numeric_limits<double>::quiet_NaN());
        } else {
            result.col(i) = xn3.col(i).head<2>() / scale;
        }
    }
    return result;
}

Eigen::Matrix2Xd xxn2xx(const Eigen::Matrix3d& K, const Eigen::Matrix2Xd& xxn) {
    Eigen::Index n = xxn.cols();
    Eigen::Matrix3Xd xxn3(3, n);
    xxn3.topRows(2) = xxn;
    xxn3.row(2).setOnes();
    Eigen::Matrix3Xd xx3 = K * xxn3;
    Eigen::Matrix2Xd result(2, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double scale = xx3(2, i);
        if (!std::isfinite(scale) || std::abs(scale) <= 1e-12) {
            result.col(i).setConstant(std::numeric_limits<double>::quiet_NaN());
        } else {
            result.col(i) = xx3.col(i).head<2>() / scale;
        }
    }
    return result;
}

Eigen::Matrix3Xd xxn2xxv(const Eigen::Matrix2Xd& xxn) {
    Eigen::Index n = xxn.cols();
    Eigen::Matrix3Xd xxv(3, n);
    xxv.topRows(2) = xxn;
    xxv.row(2).setOnes();
    return xnormalize(xxv);
}

void project_d_err(
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    double s,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    Eigen::VectorXd& d,
    Eigen::VectorXd& err)
{
    Eigen::Index n = X.cols();
    Eigen::Matrix3Xd Y = s * R * X + t.replicate(1, n);
    d = xnorm_colwise(Y);

    Eigen::Matrix3Xd vp(3, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double denom = std::max(d(i), 1e-12);
        vp.col(i) = Y.col(i) / denom;
    }

    err.resize(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        err(i) = (v.col(i) - vp.col(i)).norm();
    }
}

Eigen::VectorXd calc_weight(
    const Eigen::VectorXd& err,
    double th,
    int weight_order)
{
    Eigen::VectorXd w(err.size());
    if (weight_order < 0) {
        for (Eigen::Index i = 0; i < err.size(); ++i) {
            w(i) = (err(i) < th) ? 1.0 : 0.0;
        }
    } else {
        for (Eigen::Index i = 0; i < err.size(); ++i) {
            double e = std::max(err(i), th);
            double val = th / e;
            w(i) = std::pow(val, weight_order);
        }
    }
    return w;
}

std::vector<bool> calcInliers(
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix2Xd& x,
    const Eigen::Matrix3d& K,
    double th_pixel)
{
    Eigen::Index n = X.cols();
    Eigen::Matrix3Xd Y = R * X + t.replicate(1, n);
    std::vector<bool> inliers(static_cast<std::size_t>(n), false);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (!Y.col(i).allFinite() || Y(2, i) <= 1e-12) continue;
        const Eigen::Vector3d projected = K * Y.col(i);
        if (!projected.allFinite() || std::abs(projected(2)) <= 1e-12) continue;
        const Eigen::Vector2d pixel = projected.head<2>() / projected(2);
        const double err = (x.col(i) - pixel).norm();
        inliers[static_cast<std::size_t>(i)] = std::isfinite(err) && err < th_pixel;
    }
    return inliers;
}

double calc_dR(const Eigen::Matrix3d& R_gt, const Eigen::Matrix3d& R) {
    Eigen::Matrix3d R1 = R;
    Eigen::RowVectorXd col_norms = xnorm_colwise(R1);
    for (int i = 0; i < 3; ++i) {
        double n = std::max(col_norms(i), 1e-12);
        R1.col(i) /= n;
    }
    Eigen::Vector3d tmp = (R_gt.array() * R1.array()).colwise().sum();
    double max_val = 0.0;
    for (int i = 0; i < 3; ++i) {
        double v = safe_acos(clip(tmp(i), -1.0, 1.0));
        if (v > max_val) max_val = v;
    }
    return max_val * 180.0 / kPi;
}

double calc_dt(const Eigen::Vector3d& t_gt, const Eigen::Vector3d& t) {
    double n_gt = t_gt.norm();
    if (n_gt < 1e-12) return (t_gt - t).norm();
    return (t_gt - t).norm() / n_gt;
}

double safe_acos(double x) {
    return std::acos(std::max(-1.0, std::min(1.0, x)));
}

Eigen::Vector3d xcross_vec(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return a.cross(b);
}

Eigen::Vector3d xnormalize_vec(const Eigen::Vector3d& X) {
    double n = X.norm();
    if (n < 1e-12) return X;
    return X / n;
}

Eigen::Matrix3Xd xnormalize(const Eigen::Matrix3Xd& X) {
    Eigen::Matrix3Xd Y(3, X.cols());
    for (Eigen::Index i = 0; i < X.cols(); ++i) {
        double n = X.col(i).norm();
        if (n < 1e-12)
            Y.col(i) = X.col(i);
        else
            Y.col(i) = X.col(i) / n;
    }
    return Y;
}

double xnorm(const Eigen::Vector3d& v) {
    return v.norm();
}

Eigen::RowVectorXd xnorm_colwise(const Eigen::Matrix3Xd& X) {
    Eigen::RowVectorXd l(X.cols());
    for (Eigen::Index i = 0; i < X.cols(); ++i) {
        l(i) = X.col(i).norm();
    }
    return l;
}

double clip(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

Eigen::RowVectorXd clip_vec(const Eigen::RowVectorXd& x, double lo, double hi) {
    Eigen::RowVectorXd y = x;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        y(i) = clip(y(i), lo, hi);
    }
    return y;
}

double normalize_angle(double a) {
    double r = std::fmod(a + kPi, 2.0 * kPi);
    if (r < 0) r += 2.0 * kPi;
    return r - kPi;
}

void normalize_angles_in_place(Eigen::RowVectorXd& angles) {
    for (Eigen::Index i = 0; i < angles.size(); ++i) {
        angles(i) = normalize_angle(angles(i));
    }
}

Eigen::Vector3d weighted_mean(const Eigen::Matrix3Xd& X, const Eigen::VectorXd& w) {
    double sw = w.sum();
    if (sw < 1e-12) {
        return X.rowwise().mean();
    }
    Eigen::Vector3d Y = Eigen::Vector3d::Zero();
    for (Eigen::Index i = 0; i < X.cols(); ++i) {
        Y += X.col(i) * w(i);
    }
    return Y / sw;
}

} // namespace r2ppnp
