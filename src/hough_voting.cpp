#include "r2ppnp/hough_voting.h"
#include "r2ppnp/utils.h"
#include "r2ppnp/optimize_gn.h"
#include "debug_env.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstdio>

#ifdef DEBUG_RPNP_TRIAL
#define DBG(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

namespace r2ppnp {
namespace internal {

Eigen::RowVectorXd param_r2_base(
    const Eigen::Vector3d& X0,
    const Eigen::Vector3d& X1,
    const Eigen::Matrix3Xd& X2)
{
    Eigen::Index m = X2.cols();
    Eigen::Vector3d vX1 = X1 - X0;
    Eigen::Matrix3Xd vX2 = X2 - X1.replicate(1, m);

    Eigen::Matrix3Xd nX2(3, m);
    for (Eigen::Index i = 0; i < m; ++i) {
        nX2.col(i) = xnormalize_vec(vX1.cross(vX2.col(i)));
    }

    Eigen::RowVectorXd r2_base(m);
    for (Eigen::Index i = 0; i < m; ++i) {
        double s_i = nX2.col(0).dot(vX2.col(i));
        double sign_i = (s_i >= 0) ? 1.0 : -1.0;
        double dot_val = clip(nX2.col(0).dot(nX2.col(i)), -1.0, 1.0);
        r2_base(i) = sign_i * safe_acos(dot_val);
    }
    return r2_base;
}

void gaussian_smooth_circular_1d(
    const Eigen::VectorXd& H_1d,
    Eigen::VectorXd& H_smooth_1d,
    int nr1,
    int nr2)
{
    int N = nr1 * nr2;
    H_smooth_1d.resize(N);

    const double sigma = 0.5;
    const double sigma2 = 2.0 * sigma * sigma;

    Eigen::Vector3d G;
    double g_sum = 0;
    for (int d = 1; d >= -1; --d) {
        double val = std::exp(-(d * d) / sigma2);
        G(d + 1) = val;
        g_sum += val;
    }
    G /= g_sum;

    for (int i = 0; i < N; ++i) {
        double val = 0;
        for (int d = -1; d <= 1; ++d) {
            int ii = ((i + d) % N + N) % N;
            val += G(d + 1) * H_1d(ii);
        }
        H_smooth_1d(i) = val;
    }
    if (env_flag_enabled("HOUGH_DEBUG") && N > 618) {
        int lin = 617;
        std::fprintf(stderr, "SMOOTH1D kern=[%.4f,%.4f,%.4f] lin=%d raw_before=%f raw_after=%f\n",
            G(0), G(1), G(2), lin, H_1d(lin), H_smooth_1d(lin));
        std::fprintf(stderr, "SMOOTH1D left(%d)=%f center(%d)=%f right(%d)=%f\n",
            616, H_1d(616), 617, H_1d(617), 618, H_1d(618));
    }
}

void rebuild_histogram_1d(
    const std::vector<int>& jk_l,
    int nr1, int nr2,
    Eigen::MatrixXd& H_out)
{
    int total_bins = nr1 * nr2;
    Eigen::VectorXd H_1d = Eigen::VectorXd::Zero(total_bins);
    for (int jk : jk_l) {
        if (jk >= 0 && jk < total_bins) {
            H_1d(jk) += 1.0;
        }
    }

    Eigen::VectorXd H_smooth_1d;
    gaussian_smooth_circular_1d(H_1d, H_smooth_1d, nr1, nr2);

    bool hough_debug = env_flag_enabled("HOUGH_DEBUG");
    if (hough_debug) {
        int lin = 10 * nr2 + 17;
        if (lin >= 0 && lin < total_bins) {
            std::fprintf(stderr, "HOUGH 1d_smooth lin=%d raw=%.4f smooth_1d=%.4f\n",
                lin, H_1d(lin), H_smooth_1d(lin));
        }
    }

    H_out.resize(nr1, nr2);
    for (int i = 0; i < nr1; ++i) {
        for (int j = 0; j < nr2; ++j) {
            H_out(i, j) = H_smooth_1d(i * nr2 + j);
        }
    }
}

void dilate_3x3(
    const Eigen::MatrixXd& H,
    Eigen::MatrixXd& H_dilated)
{
    int nr1 = static_cast<int>(H.rows());
    int nr2 = static_cast<int>(H.cols());
    H_dilated.resize(nr1, nr2);

    for (int i = 0; i < nr1; ++i) {
        for (int j = 0; j < nr2; ++j) {
            double mx = -std::numeric_limits<double>::max();
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ii = std::max(0, std::min(nr1 - 1, i + di));
                    int jj = std::max(0, std::min(nr2 - 1, j + dj));
                    mx = std::max(mx, H(ii, jj));
                }
            }
            H_dilated(i, j) = mx;
        }
    }
}

std::vector<std::pair<int, int>> find_local_peaks(
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& H_dilated,
    double thH,
    double peak_ratio)
{
    int nr1 = static_cast<int>(H.rows());
    int nr2 = static_cast<int>(H.cols());
    double maxH = H.maxCoeff();
    double threshold = std::max(thH, maxH * peak_ratio);

    std::vector<std::pair<int, int>> peaks;
    for (int i = 0; i < nr1; ++i) {
        for (int j = 0; j < nr2; ++j) {
            if (std::abs(H(i, j) - H_dilated(i, j)) < 1e-10 && H(i, j) >= threshold) {
                peaks.push_back({i, j});
            }
        }
    }
    return peaks;
}

TrialResult rpnp_trial(
    int i0, int i1,
    const Eigen::Matrix3Xd& XXw,
    const Eigen::Matrix3Xd& xxv,
    const Eigen::RowVectorXd& bin_r1,
    const Eigen::RowVectorXd& bin_r2,
    const Config& config)
{
    TrialResult result;
    Eigen::Index npt = XXw.cols();
    int nr1 = config.nr1;
    int nr2 = nr1 * 2;

    DBG("DEBUG_RPNP_TRIAL_ENTER i0=%d i1=%d npt=%ld\n", i0, i1, (long)npt);

    if (xxv.col(i0).head<2>().norm() > xxv.col(i1).head<2>().norm()) {
        std::swap(i0, i1);
    }

    std::vector<bool> i2s_mask(static_cast<size_t>(npt), true);
    i2s_mask[static_cast<size_t>(i0)] = false;
    i2s_mask[static_cast<size_t>(i1)] = false;

    Eigen::Vector3d v0 = xxv.col(i0);
    Eigen::Vector3d v1 = xxv.col(i1);

    Eigen::Vector3d X0 = XXw.col(i0);
    Eigen::Vector3d X1 = XXw.col(i1);

    Eigen::Index m_raw = npt - 2;
    Eigen::Matrix3Xd v2(3, m_raw);
    Eigen::Matrix3Xd X2(3, m_raw);
    Eigen::Index col = 0;
    for (Eigen::Index i = 0; i < npt; ++i) {
        if (i2s_mask[static_cast<size_t>(i)]) {
            v2.col(col) = xxv.col(i);
            X2.col(col) = XXw.col(i);
            ++col;
        }
    }

    double D1 = (X1 - X0).norm();
    Eigen::RowVectorXd D2(m_raw), D3(m_raw);
    for (Eigen::Index i = 0; i < m_raw; ++i) {
        D2(i) = (X2.col(i) - X0).norm();
        D3(i) = (X2.col(i) - X1).norm();
    }

    std::vector<Eigen::Index> valid_idx;
    for (Eigen::Index i = 0; i < m_raw; ++i) {
        const Eigen::Vector3d edge1 = X1 - X0;
        const Eigen::Vector3d edge2 = X2.col(i) - X0;
        const double area = edge1.cross(edge2).norm();
        const double area_scale = std::max(edge1.norm() * edge2.norm(), 1.0);
        if (D2(i) > 1e-5 && D3(i) > 1e-5 &&
            std::isfinite(area) && area > 1e-10 * area_scale) {
            valid_idx.push_back(i);
        }
    }
    int m = static_cast<int>(valid_idx.size());
    if (m < 4) {
        return result;
    }

    Eigen::Matrix3Xd v2f(3, m), X2f(3, m);
    Eigen::RowVectorXd D2f(m), D3f(m);
    for (int i = 0; i < m; ++i) {
        v2f.col(i) = v2.col(valid_idx[static_cast<size_t>(i)]);
        X2f.col(i) = X2.col(valid_idx[static_cast<size_t>(i)]);
        D2f(i) = D2(valid_idx[static_cast<size_t>(i)]);
        D3f(i) = D3(valid_idx[static_cast<size_t>(i)]);
    }

    Eigen::RowVectorXd r2_base_data = param_r2_base(X0, X1, X2f);

    double cg1 = clip(v0.dot(v1), -1.0, 1.0);
    Eigen::RowVectorXd cg2(m), cg3(m);
    for (int i = 0; i < m; ++i) {
        cg2(i) = clip(v0.dot(v2f.col(i)), -1.0, 1.0);
        cg3(i) = clip(v1.dot(v2f.col(i)), -1.0, 1.0);
    }

    double sg1 = std::sqrt(std::max(0.0, 1.0 - cg1 * cg1));
    Eigen::RowVectorXd sg2(m);
    for (int i = 0; i < m; ++i) {
        sg2(i) = std::sqrt(std::max(0.0, 1.0 - cg2(i) * cg2(i)));
    }

    double l1 = cg1;
    Eigen::RowVectorXd l2 = cg2;
    double C1 = sg1;
    Eigen::RowVectorXd C2 = sg2;

    Eigen::RowVectorXd k = D2f / D1;
    Eigen::RowVectorXd A1 = k.array() * k.array();
    Eigen::RowVectorXd A2 = A1.array() * (C1 * C1) - C2.array() * C2.array();
    Eigen::RowVectorXd A3 = l2.array() * cg3.array() - l1;
    Eigen::RowVectorXd A4 = l1 * cg3.array() - l2.array();
    Eigen::RowVectorXd A5 = cg3;
    Eigen::RowVectorXd A6 = (D3f.array().square() - D1 * D1 - D2f.array().square()) / (2 * D1 * D1);
    Eigen::RowVectorXd A7 = 1.0 - l1 * l1 - l2.array().square()
                            + l1 * l2.array() * cg3.array()
                            + C1 * C1 * A6.array();

    Eigen::RowVectorXd B4 = A6.array().square() - A1.array() * A5.array().square();
    Eigen::RowVectorXd B3 = 2.0 * (A3.array() * A6.array() - A1.array() * A4.array() * A5.array());
    Eigen::RowVectorXd B2 = A3.array().square() + 2.0 * A6.array() * A7.array()
                            - A1.array() * A4.array().square()
                            - A2.array() * A5.array().square();
    Eigen::RowVectorXd B1 = 2.0 * (A3.array() * A7.array() - A2.array() * A4.array() * A5.array());
    Eigen::RowVectorXd B0 = A7.array().square() - A2.array() * A4.array().square();

    Eigen::RowVectorXd t1s = bin_r1.array().tan() * C1;

    DBG("DEBUG_POLY l1=%.15e C1=%.15e m=%d\n", l1, C1, m);
    DBG("DEBUG_POLY A1[0]=%.15e A2[0]=%.15e A3[0]=%.15e A4[0]=%.15e A5[0]=%.15e A6[0]=%.15e A7[0]=%.15e\n",
        A1(0), A2(0), A3(0), A4(0), A5(0), A6(0), A7(0));
    DBG("DEBUG_POLY B0[0]=%.15e B1[0]=%.15e B2[0]=%.15e B3[0]=%.15e B4[0]=%.15e\n",
        B0(0), B1(0), B2(0), B3(0), B4(0));
#ifdef DEBUG_RPNP_TRIAL
    if (t1s.size() >= 30) {
        DBG("DEBUG_POLY t1s[0]=%.15e t1s[15]=%.15e t1s[29]=%.15e\n",
            t1s(0), t1s(15), t1s(29));
    } else {
        DBG("DEBUG_POLY t1s_size=%ld first=%.15e last=%.15e\n",
            static_cast<long>(t1s.size()), t1s(0), t1s(t1s.size() - 1));
    }
#endif

    Eigen::MatrixXd Y(m, nr1);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < nr1; ++j) {
            double t = t1s(j);
            double Y0_val = ((((B4(i) * t) + B3(i)) * t + B2(i)) * t + B1(i)) * t + B0(i);
            double tmp = A4(i) + A5(i) * t;
            Y(i, j) = std::abs(Y0_val) / std::max(tmp * tmp, 1e-12);
        }
    }

    std::vector<int> i_l, j_l;
    std::vector<double> t1_l_vals;
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < nr1; ++c) {
            if (Y(r, c) < config.th_cons) {
                i_l.push_back(r);
                j_l.push_back(c + 1);
                t1_l_vals.push_back(t1s(c));
            }
        }
    }

    if (i_l.empty()) return result;

    size_t L = i_l.size();
    std::vector<double> t2_l_vals(L, 0.0);
    std::vector<bool> valid_t2(L, false);
    for (size_t idx = 0; idx < L; ++idx) {
        int i_idx = i_l[idx];
        double t1 = t1_l_vals[idx];
        double tmp1 = A6(i_idx) * t1 * t1 + A3(i_idx) * t1 + A7(i_idx);
        double tmp2 = A4(i_idx) + A5(i_idx) * t1;
        if (std::abs(tmp2) > 1e-9) {
            t2_l_vals[idx] = -tmp1 / tmp2;
            valid_t2[idx] = std::isfinite(t2_l_vals[idx]);
        }
    }

    std::vector<int> i_l_f, j_l_f;
    std::vector<double> t1_l_f, t2_l_f;
    for (size_t idx = 0; idx < L; ++idx) {
        if (!valid_t2[idx]) continue;
        double d1 = l1 + t1_l_vals[idx];
        double d2 = l2(i_l[idx]) + t2_l_vals[idx];
        if (d2 > 0.05 && d1 > 0.05) {
            i_l_f.push_back(i_l[idx]);
            j_l_f.push_back(j_l[idx]);
            t1_l_f.push_back(t1_l_vals[idx]);
            t2_l_f.push_back(t2_l_vals[idx]);
        }
    }

    if (i_l_f.size() < 3) return result;

    double step_r2_val = 2.0 * kPi / nr2;

    DBG("DEBUG_CAND total_candidates=%zu filtered=%zu\n", i_l.size(), i_l_f.size());
    DBG("DEBUG_CAND i_l[0]=%d j_l[0]=%d t1_l[0]=%.15e t2_l[0]=%.15e\n",
        i_l_f[0], j_l_f[0], t1_l_f[0], t2_l_f[0]);

    Eigen::Vector3d nV_base = xnormalize_vec(xcross_vec(v0, v1));
    if (!nV_base.allFinite() || nV_base.norm() < 1e-12) return result;
    DBG("DEBUG_R2 first3 r2_base=[%.15e,%.15e,%.15e] dir_r=[%d,%d,%d]\n",
        r2_base_data(0), r2_base_data(1), r2_base_data(2),
        static_cast<int>(nV_base.dot(v2f.col(0)) > 0 ? 1 : (nV_base.dot(v2f.col(0)) < 0 ? -1 : -1)),
        static_cast<int>(nV_base.dot(v2f.col(1)) > 0 ? 1 : (nV_base.dot(v2f.col(1)) < 0 ? -1 : -1)),
        static_cast<int>(nV_base.dot(v2f.col(2)) > 0 ? 1 : (nV_base.dot(v2f.col(2)) < 0 ? -1 : -1)));

    size_t Lf = i_l_f.size();
    std::vector<int> jk_l;

    for (size_t idx = 0; idx < Lf; ++idx) {
        int i_idx = i_l_f[idx];
        double d1_val = l1 + t1_l_f[idx];
        double d2_val = l2(i_idx) + t2_l_f[idx];

        double dir_r_val = nV_base.dot(v2f.col(i_idx));
        if (std::abs(dir_r_val) < 1e-12) dir_r_val = -1;
        if (dir_r_val < 0) dir_r_val = -1;
        else dir_r_val = 1;

        Eigen::Vector3d vd1 = v1 * d1_val - v0;
        Eigen::Vector3d vd2 = v2f.col(i_idx) * d2_val - v0;
        Eigen::Vector3d nX_val = xnormalize_vec(vd1.cross(vd2));
        if (!nX_val.allFinite() || nX_val.norm() < 1e-12) continue;

        double dot_nv_nx = clip(nV_base.dot(nX_val), -1.0, 1.0);
        double r2_val = dir_r_val * safe_acos(dot_nv_nx) - r2_base_data(i_idx);

        r2_val = normalize_angle(r2_val);

        int k_val = static_cast<int>(clip(
            std::floor((r2_val + kPi) / step_r2_val),
            0.0, static_cast<double>(nr2 - 1)));
        int jk = (j_l_f[idx] - 1) * nr2 + k_val;
        jk_l.push_back(jk);
    }

    if (jk_l.size() >= 4) {
        DBG("DEBUG_R2 first_jk jk[0]=%d jk[1]=%d jk[2]=%d jk[3]=%d\n",
            jk_l[0], jk_l[1], jk_l[2], jk_l[3]);
    }

    int total_bins = nr1 * nr2;
    bool hough_debug = env_flag_enabled("HOUGH_DEBUG");
    Eigen::MatrixXd H_smooth;
    rebuild_histogram_1d(jk_l, nr1, nr2, H_smooth);

    Eigen::MatrixXd H_dilated;
    dilate_3x3(H_smooth, H_dilated);

    if (hough_debug) {
        Eigen::VectorXd H_raw = Eigen::VectorXd::Zero(total_bins);
        for (int jk : jk_l) if (jk >= 0 && jk < total_bins) H_raw(jk) += 1.0;
        double raw_max = H_raw.maxCoeff();
        int raw_peak_idx = -1;
        H_raw.maxCoeff(&raw_peak_idx);
        int raw_row = raw_peak_idx / nr2;
        int raw_col = raw_peak_idx % nr2;
        std::fprintf(stderr, "HOUGH raw_peak=(%d,%d) raw_val=%.4f smoothed_val=%.4f nvotes=%zu\n",
            raw_row, raw_col, raw_max, H_smooth.maxCoeff(), jk_l.size());
        std::fprintf(stderr, "HOUGH first_jk_count=%zu\n", jk_l.size());
        // Print smoothed values at key positions
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                int rr = 10 + dr;
                int cc = 17 + dc;
                int lin = rr * nr2 + cc;
                if (rr >= 0 && rr < nr1 && cc >= 0 && cc < nr2) {
                    double rv = H_raw(lin);
                    double sv = H_smooth(rr, cc);
                    std::fprintf(stderr, "HOUGH raw(%d,%d)=%.4f smooth(%d,%d)=%.4f\n", rr, cc, rv, rr, cc, sv);
                }
            }
        }
    }

    const double thH_val = std::max(
        static_cast<double>(config.min_thH), static_cast<double>(npt) * config.thH_factor);
    std::vector<std::pair<int, int>> peaks =
        find_local_peaks(H_smooth, H_dilated, thH_val, config.thH_ratio);

    if (hough_debug && !peaks.empty()) {
        double maxH = H_smooth.maxCoeff();
        std::fprintf(stderr, "HOUGH i0=%d i1=%d nvotes=%zu maxH=%.4f npeaks=%zu thH=%.3f\n",
            i0, i1, jk_l.size(), maxH, peaks.size(), thH_val);
        for (size_t pi = 0; pi < std::min(peaks.size(), size_t(5)); ++pi) {
            int r = peaks[pi].first, c = peaks[pi].second;
            std::fprintf(stderr, "HOUGH peak[%zu]=(%d,%d) val=%.4f\n", pi, r, c, H_smooth(r, c));
            if (pi == 0) {
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        int rr = ((r + dr) % nr1 + nr1) % nr1;
                        int cc = ((c + dc) % nr2 + nr2) % nr2;
                        std::fprintf(stderr, "  nb(%d,%d)=%.4f ", rr, cc, H_smooth(rr, cc));
                    }
                    std::fprintf(stderr, "\n");
                }
            }
        }
    }

    if (peaks.empty()) return result;

    std::vector<double> score_m;
    for (const auto& p : peaks) {
        score_m.push_back(H_smooth(p.first, p.second));
    }

    std::vector<size_t> sorted_indices(peaks.size());
    for (size_t i = 0; i < sorted_indices.size(); ++i) sorted_indices[i] = i;
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t a, size_t b) {
        return score_m[a] > score_m[b];
    });

    Eigen::Vector3d y_v = xnormalize_vec(X1 - X0);
    Eigen::Vector3d z_v = xnormalize_vec(X2f.col(0) - X0);
    Eigen::Vector3d x_v = xnormalize_vec(y_v.cross(z_v));
    z_v = xnormalize_vec(x_v.cross(y_v));
    if (!x_v.allFinite() || !y_v.allFinite() || !z_v.allFinite() ||
        x_v.norm() < 1e-12 || y_v.norm() < 1e-12 || z_v.norm() < 1e-12) {
        return result;
    }
    Eigen::Matrix3d Rm0;
    Rm0.col(0) = x_v;
    Rm0.col(1) = y_v;
    Rm0.col(2) = z_v;

    int num_peaks_to_try = std::min(config.num_peaks, static_cast<int>(peaks.size()));

    for (int pi = 0; pi < num_peaks_to_try; ++pi) {
        size_t peak_idx = sorted_indices[pi];
        int j_m = peaks[peak_idx].first;
        int k_m = peaks[peak_idx].second;

        // Use integer bin center (matching MATLAB), not sub-pixel centroid
        double d1_val = l1 + t1s(j_m);
        double r2_val = bin_r2(k_m);

        Eigen::Vector3d xc = xnormalize_vec(xcross_vec(v0, v1));
        Eigen::Vector3d yc = v1 * d1_val - v0;
        double ny = yc.norm();
        double s0_val = ny / D1;
        if (!std::isfinite(ny) || !std::isfinite(s0_val) ||
            ny < 1e-12 || s0_val < 1e-12) {
            continue;
        }
        yc = yc / ny;
        Eigen::Vector3d zc = xnormalize_vec(xc.cross(yc));
        Eigen::Matrix3d Rc0;
        Rc0.col(0) = xc;
        Rc0.col(1) = yc;
        Rc0.col(2) = zc;
        Eigen::Vector3d tc0 = v0;

        double c2_val = std::cos(r2_val);
        double s2_val = std::sin(r2_val);
        Eigen::Matrix3d Rcy;
        Rcy << c2_val, 0, s2_val,
                0, 1, 0,
                -s2_val, 0, c2_val;

        Eigen::Matrix3d R0_mat = Rc0 * Rcy * Rm0.transpose();
        Eigen::Vector3d t0_mat = tc0 / s0_val - R0_mat * X0;

        DBG("DEBUG_POSE peak[0] d1=%.15e r2=%.15e s0=%.15e\n", d1_val, r2_val, s0_val);
        DBG("DEBUG_POSE R0col0=(%.10e,%.10e,%.10e) t0=(%.10e,%.10e,%.10e)\n",
            R0_mat(0,0), R0_mat(1,0), R0_mat(2,0), t0_mat(0), t0_mat(1), t0_mat(2));

        Eigen::VectorXd d_proj, err_proj;
        project_d_err(R0_mat, t0_mat, 1.0, XXw, xxv, d_proj, err_proj);

        int ninlier = 0;
        for (Eigen::Index i = 0; i < npt; ++i) {
            if (err_proj(i) < config.thv) ++ninlier;
        }

        if (ninlier < thH_val) continue;

        std::vector<int> weights = {2};
        if (ninlier < m * 0.5) weights = {2, 2};
        if (ninlier < m * 0.1) weights = {2, 2, 2};

        GNResult gnr = dsw_gn(R0_mat, t0_mat, XXw, xxv, config.thv,
                                 weights, config.gn_max_iter, config.gn_converge, err_proj);

        if (!gnr.success || gnr.w.size() != npt ||
            !gnr.R.allFinite() || !gnr.t.allFinite()) {
            continue;
        }

        double score = 0;
        for (Eigen::Index i = 0; i < npt; ++i) {
            if (gnr.w(i) > 0.99999) ++score;
        }

        DBG("DEBUG_GN ninlier_before=%d score_after=%.0f gnr_Rcol0=(%.10e,%.10e,%.10e)\n",
            ninlier, score, gnr.R(0,0), gnr.R(1,0), gnr.R(2,0));

        result.scores.push_back(score);
        result.R_candidates.push_back(gnr.R);
        result.t_candidates.push_back(gnr.t);
    }

    return result;
}

} // namespace internal
} // namespace r2ppnp
