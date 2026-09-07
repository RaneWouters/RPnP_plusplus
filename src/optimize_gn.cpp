#include "r2ppnp/optimize_gn.h"
#include "r2ppnp/utils.h"
#include "debug_env.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <cstdio>

#ifdef DEBUG_RPNP_TRIAL
#define DBG(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

namespace r2ppnp {
namespace internal {

GNResult sw_gn(
    const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& t0,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    double thv,
    const Eigen::VectorXd& err_in,
    int weight_order,
    int max_iter,
    double converge_threshold)
{
    GNResult result;
    result.R = R0;
    result.t = t0;

    Eigen::VectorXd err = err_in;
    if (err.size() == 0) {
        Eigen::VectorXd d;
        project_d_err(R0, t0, 1.0, X, v, d, err);
    }

    Eigen::VectorXd w = calc_weight(err, thv, weight_order);
    Eigen::Index n = v.cols();

    if (X.cols() != v.cols() || err.size() != v.cols() ||
        !X.allFinite() || !v.allFinite() || !err.allFinite()) {
        return result;
    }

    Eigen::Vector3d z = weighted_mean(v, w);
    z = xnormalize_vec(z);
    if (!z.allFinite() || z.norm() < 1e-12) {
        return result;
    }

    Eigen::Vector3d x, y;
    if (z(0) > z(1)) {
        x = xnormalize_vec(xcross_vec(Eigen::Vector3d(0, 1, 0), z));
        y = xnormalize_vec(xcross_vec(z, x));
    } else {
        y = xnormalize_vec(xcross_vec(z, Eigen::Vector3d(1, 0, 0)));
        x = xnormalize_vec(xcross_vec(y, z));
    }
    if (!x.allFinite() || !y.allFinite() || x.norm() < 1e-12 || y.norm() < 1e-12) {
        return result;
    }

    Eigen::Matrix3d R_vcam;
    R_vcam.col(0) = x;
    R_vcam.col(1) = y;
    R_vcam.col(2) = z;

    Eigen::Matrix3Xd v_vcam = R_vcam.transpose() * v;

    Eigen::Vector3d X_bar = weighted_mean(X, w);
    Eigen::Matrix3Xd X_centered = X - X_bar.replicate(1, n);
    Eigen::Matrix3Xd X_vcam = (R_vcam.transpose() * R0) * X_centered;
    Eigen::Vector3d t_vcam = R_vcam.transpose() * (R0 * X_bar + t0);

    GNResult opt_result = optimize_gn(
        X_vcam, v_vcam, t_vcam, thv, w, err, weight_order,
        max_iter, converge_threshold);

    if (!opt_result.success || !opt_result.R.allFinite() || !opt_result.t.allFinite()) {
        return result;
    }

    result.R = R_vcam * opt_result.R * R_vcam.transpose() * R0;
    result.t = R_vcam * opt_result.t - result.R * X_bar;
    result.w = opt_result.w;
    result.err = opt_result.err;
    result.success = opt_result.success;
    return result;
}

GNResult dsw_gn(
    const Eigen::Matrix3d& R0,
    const Eigen::Vector3d& t0,
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    double thv,
    const std::vector<int>& weight_orders,
    int max_iter,
    double converge_threshold,
    const Eigen::VectorXd& err_in)
{
    static bool gn_debug = env_flag_enabled("GN_DEBUG");
    GNResult result;
    result.R = R0;
    result.t = t0;

    if (X.cols() != v.cols() || X.cols() < 5 || weight_orders.empty() ||
        !X.allFinite() || !v.allFinite()) {
        return result;
    }

    Eigen::VectorXd err = err_in;
    if (err.size() == 0) {
        Eigen::VectorXd d;
        project_d_err(R0, t0, 1.0, X, v, d, err);
    }

    Eigen::Index n = X.cols();
    if (err.size() != n || !err.allFinite()) {
        return result;
    }
    int n_inlier = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (err(i) < thv) ++n_inlier;
    }

    if (n_inlier < 5) {
        result.success = false;
        return result;
    }

    std::vector<bool> idx_mask(n, false);
    for (Eigen::Index i = 0; i < n; ++i) {
        idx_mask[i] = (err(i) < thv * 7);
    }

    int m = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (idx_mask[i]) ++m;
    }

    Eigen::Matrix3Xd X1(3, m);
    Eigen::Matrix3Xd v1(3, m);
    Eigen::VectorXd err1(m);
    int col = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (idx_mask[i]) {
            X1.col(col) = X.col(i);
            v1.col(col) = v.col(i);
            err1(col) = err(i);
            ++col;
        }
    }

    Eigen::Matrix3d R_cur = R0;
    Eigen::Vector3d t_cur = t0;
    Eigen::VectorXd w1;
    Eigen::VectorXd err1_out;

    if (gn_debug) {
        int cw0 = 0;
        for (Eigen::Index i = 0; i < m; ++i) {
            if (r2ppnp::calc_weight(err1, thv, weight_orders[0])(i) > 0.99999) ++cw0;
        }
        std::fprintf(stderr, "DSW_GN n=%ld n_inlier=%d m=%d thv=%.6f weights_sz=%zu max_iter=%d\n",
            (long)n, n_inlier, m, thv, weight_orders.size(), max_iter);
        std::fprintf(stderr, "DSW_GN R0_col0=[%.15e,%.15e,%.15e]\n", R0(0,0), R0(1,0), R0(2,0));
        std::fprintf(stderr, "DSW_GN t0=[%.15e,%.15e,%.15e]\n", t0(0), t0(1), t0(2));
        std::fprintf(stderr, "DSW_GN cw0=%d best_score0=%.6f\n", cw0,
            r2ppnp::calc_weight(err1, thv, weight_orders[0]).sum());
    }

    for (size_t wi = 0; wi < weight_orders.size(); ++wi) {
        GNResult swr = sw_gn(
            R_cur, t_cur, X1, v1, thv, err1, weight_orders[wi],
            max_iter, converge_threshold);
        if (!swr.success || swr.w.size() != m || swr.err.size() != m ||
            !swr.R.allFinite() || !swr.t.allFinite() ||
            !swr.w.allFinite() || !swr.err.allFinite()) {
            result.success = false;
            return result;
        }
        R_cur = swr.R;
        t_cur = swr.t;
        w1 = swr.w;
        err1_out = swr.err;
        err1 = err1_out;
    }

    if (gn_debug) {
        int final_inlier = 0;
        for (Eigen::Index i = 0; i < w1.size(); ++i)
            if (w1(i) > 0.99999) ++final_inlier;
        std::fprintf(stderr, "DSW_GN final_cw=%d final_score=%.6f\n", final_inlier, w1.sum());
    }

    result.w.resize(n);
    result.err.resize(n);
    result.w.setZero();
    result.err = err;
    col = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (idx_mask[i]) {
            result.w(i) = w1(col);
            result.err(i) = err1_out(col);
            ++col;
        }
    }

    result.R = R_cur;
    result.t = t_cur;
    result.success = true;
    return result;
}

GNResult optimize_gn(
    const Eigen::Matrix3Xd& X,
    const Eigen::Matrix3Xd& v,
    const Eigen::Vector3d& t0,
    double thv,
    const Eigen::VectorXd& w_in,
    const Eigen::VectorXd& err_in,
    int weight_order,
    int max_iter,
    double converge_threshold)
{
    GNResult result;
    Eigen::Index n = v.cols();
    result.R = Eigen::Matrix3d::Identity();
    result.t = t0;
    result.w = w_in;
    result.err = err_in;
    if (X.cols() != n || n < 5 || w_in.size() != n || err_in.size() != n ||
        !X.allFinite() || !v.allFinite() || !t0.allFinite() ||
        !w_in.allFinite() || !err_in.allFinite()) {
        return result;
    }

    int cw = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (w_in(i) > 0.99999) ++cw;
    }
    if (cw < 5) {
        return result;
    }

    Eigen::Matrix2Xd xx(2, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double vz = v(2, i);
        if (std::abs(vz) < 1e-12) {
            vz = std::copysign(1e-12, vz == 0.0 ? 1.0 : vz);
        }
        xx(0, i) = v(0, i) / vz;
        xx(1, i) = v(1, i) / vz;
    }

    Eigen::VectorXd uu = xx.row(0);
    Eigen::VectorXd vv = xx.row(1);

    Eigen::MatrixXd Ap(20, n);
    Ap.setZero();
    Ap.middleRows(0, 3) = X;
    Ap.middleRows(6, 3) = -uu.transpose().replicate(3, 1).array() * X.array();
    Ap.middleRows(13, 3) = X;
    Ap.middleRows(16, 3) = -vv.transpose().replicate(3, 1).array() * X.array();

    Eigen::VectorXd tv1 = Eigen::VectorXd::Constant(n, t0(0));
    Eigen::VectorXd tv2 = Eigen::VectorXd::Constant(n, t0(1));
    Eigen::VectorXd tv3 = Eigen::VectorXd::Constant(n, t0(2));
    Ap.row(9) = tv1 - tv3.cwiseProduct(uu);
    Ap.row(19) = tv2 - tv3.cwiseProduct(vv);

    Eigen::MatrixXd Bp(6, n);
    Bp.setZero();
    Bp.row(0) = -Eigen::VectorXd::Ones(n);
    Bp.row(2) = uu;
    Bp.row(4) = -Eigen::VectorXd::Ones(n);
    Bp.row(5) = vv;

    Eigen::Vector3d s = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::VectorXd r(10);
    r << 1, 0, 0, 0, 1, 0, 0, 0, 1, 1;

    Eigen::Matrix3d best_R = R;
    Eigen::Vector3d best_t = t0;
    Eigen::VectorXd best_w = w_in;
    Eigen::VectorXd best_err = err_in;
    double best_score = w_in.sum();
    Eigen::VectorXd w = w_in;

    bool gn_debug = env_flag_enabled("GN_DEBUG");
    if (gn_debug) {
        std::fprintf(stderr, "GN_ENTER n=%ld cw0=%d best_score0=%.6f thv=%.6f w_order=%d max_iter=%d\n",
            (long)n, cw, best_score, thv, weight_order, max_iter);
        std::fprintf(stderr, "GN_ENTER t0=[%.15e,%.15e,%.15e]\n", t0(0), t0(1), t0(2));
        std::fprintf(stderr, "GN_ENTER s0=[%.15e,%.15e,%.15e]\n", s(0), s(1), s(2));
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        DBG("DEBUG_GNSTEP iter=%d best_score=%.6f\n", iter, best_score);

        Eigen::MatrixXd A(2 * n, 10);
        A.setZero();
        for (Eigen::Index i = 0; i < n; ++i) {
            for (int j = 0; j < 10; ++j) {
                A(i, j) = Ap(j, i) * w(i);
                A(i + n, j) = Ap(10 + j, i) * w(i);
            }
        }

        Eigen::MatrixXd B(2 * n, 3);
        for (Eigen::Index i = 0; i < n; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i, j) = Bp(j, i) * w(i);
                B(i + n, j) = Bp(3 + j, i) * w(i);
            }
        }

        Eigen::Matrix3d BTB = B.transpose() * B + Eigen::Matrix3d::Identity() * 1e-3;
        Eigen::LDLT<Eigen::Matrix3d> b_solver(BTB);
        if (b_solver.info() != Eigen::Success) break;
        Eigen::MatrixXd C = b_solver.solve(B.transpose() * A);
        if (b_solver.info() != Eigen::Success || !C.allFinite()) break;

        Eigen::MatrixXd M = A - B * C;
        Eigen::MatrixXd MTM = M.transpose() * M;

        Eigen::MatrixXd J(10, 3);
        J << 2*s(0), -2*s(1), -2*s(2),
             2*s(1), 2*s(0), -2,
             2*s(2), 2, 2*s(0),
             2*s(1), 2*s(0), 2,
             -2*s(0), 2*s(1), -2*s(2),
             -2, 2*s(2), 2*s(1),
             2*s(2), -2, 2*s(0),
             2, 2*s(2), 2*s(1),
             -2*s(0), -2*s(1), 2*s(2),
             0, 0, 0;

        const Eigen::Matrix3d normal_matrix =
            J.transpose() * MTM * J + Eigen::Matrix3d::Identity() * 1e-3;
        Eigen::LDLT<Eigen::Matrix3d> step_solver(normal_matrix);
        if (step_solver.info() != Eigen::Success) break;
        Eigen::Vector3d ds = -step_solver.solve((r.transpose() * MTM * J).transpose());
        if (step_solver.info() != Eigen::Success || !ds.allFinite()) break;
        double nds = ds.norm();

        if (gn_debug) {
            std::fprintf(stderr, "GN_ITER %d: nds=%.6e ds=[%.15e,%.15e,%.15e]\n", iter, nds, ds(0), ds(1), ds(2));
            std::fprintf(stderr, "GN_ITER %d: C=[%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e]\n",
                iter, C(0,0),C(0,1),C(0,2),C(0,3),C(0,4),C(0,5),C(0,6),C(0,7),C(0,8),C(0,9));
            std::fprintf(stderr, "GN_ITER %d:   [%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e]\n",
                iter, C(1,0),C(1,1),C(1,2),C(1,3),C(1,4),C(1,5),C(1,6),C(1,7),C(1,8),C(1,9));
            std::fprintf(stderr, "GN_ITER %d:   [%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e]\n",
                iter, C(2,0),C(2,1),C(2,2),C(2,3),C(2,4),C(2,5),C(2,6),C(2,7),C(2,8),C(2,9));
        }

        DBG("DEBUG_GNSTEP ds=[%.6e,%.6e,%.6e] nds=%.6e\n", ds(0), ds(1), ds(2), nds);

        if (nds < converge_threshold) {
            DBG("DEBUG_GNSTEP converged nds=%.6e\n", nds);
            break;
        }

        Eigen::Vector3d s0 = s;
        Eigen::Vector3d ds0 = ds;

        bool found_better = false;
        // Match the three line-search scales used by the reference MATLAB implementation.
        const double step_scales[] = {0.1, 0.05, 0.025};
        const int n_step_scales = sizeof(step_scales) / sizeof(step_scales[0]);
        for (int i_step = 0; i_step < n_step_scales; ++i_step) {
            double lstep = step_scales[i_step];
            Eigen::Vector3d ds_step = ds;
            if (nds > lstep) {
                ds_step = ds / nds * lstep;
            }
            Eigen::Vector3d s_try = s0 + ds_step;

            r(0) = s_try(0)*s_try(0) - s_try(1)*s_try(1) - s_try(2)*s_try(2) + 1;
            r(1) = 2*s_try(0)*s_try(1) - 2*s_try(2);
            r(2) = 2*s_try(0)*s_try(2) + 2*s_try(1);
            r(3) = 2*s_try(0)*s_try(1) + 2*s_try(2);
            r(4) = -s_try(0)*s_try(0) + s_try(1)*s_try(1) - s_try(2)*s_try(2) + 1;
            r(5) = 2*s_try(1)*s_try(2) - 2*s_try(0);
            r(6) = 2*s_try(0)*s_try(2) - 2*s_try(1);
            r(7) = 2*s_try(1)*s_try(2) + 2*s_try(0);
            r(8) = -s_try(0)*s_try(0) - s_try(1)*s_try(1) + s_try(2)*s_try(2) + 1;
            r(9) = 1;

            double scale = 1.0 + s_try(0)*s_try(0) + s_try(1)*s_try(1) + s_try(2)*s_try(2);

            Eigen::Matrix3d R_mat;
            R_mat << r(0), r(1), r(2),
                     r(3), r(4), r(5),
                     r(6), r(7), r(8);
            R_mat /= scale;

            Eigen::VectorXd Crt = C * r;
            Eigen::Vector3d t_vec = (t0 + Crt.head<3>()) / scale;

            Eigen::VectorXd d_proj, err_proj;
            project_d_err(R_mat, t_vec, 1.0, X, v, d_proj, err_proj);
            Eigen::VectorXd w_new = calc_weight(err_proj, thv, weight_order);
            double score = w_new.sum();

            if (gn_debug) {
                std::fprintf(stderr, "GN_LS iter=%d i=%d lstep=%.6f ds_try=[%.15e,%.15e,%.15e] score=%.6f (best=%.6f)\n",
                    iter, i_step, lstep, ds_step(0), ds_step(1), ds_step(2), score, best_score);
            }

            if (score >= best_score) {
                best_score = score;
                best_R = R_mat;
                best_t = t_vec;
                best_w = w_new;
                best_err = err_proj;
                s = s_try;
                found_better = true;
                if (gn_debug) std::fprintf(stderr, "GN_LS ACCEPTED i=%d\n", i_step);
                break;
            } else {
                s = s0;
                ds = ds0;
            }
        }
        if (!found_better) {
            break;
        }
        R = best_R;
        w = best_w;
    }

    result.R = best_R;
    result.t = best_t;
    result.w = best_w;
    result.err = best_err;
    result.success = true;
    return result;
}

} // namespace internal
} // namespace r2ppnp
