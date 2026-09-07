#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "r2ppnp/r2ppnp.h"
#include "r2ppnp/utils.h"

int main() {
    int n = 100;
    Eigen::Matrix3Xd X(3, n);
    Eigen::Matrix2Xd x(2, n);

    Eigen::Matrix3d R_gt = Eigen::Matrix3d::Identity();
    R_gt = Eigen::AngleAxisd(0.1, Eigen::Vector3d(0, 1, 0)) *
           Eigen::AngleAxisd(0.2, Eigen::Vector3d(0, 0, 1));
    Eigen::Vector3d t_gt(0.1, 0.2, 1.0);

    for (int i = 0; i < n; ++i) {
        X.col(i) = Eigen::Vector3d(
            static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0,
            static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0,
            static_cast<double>(rand()) / RAND_MAX * 2.0 + 1.0);

        Eigen::Vector3d Y = R_gt * X.col(i) + t_gt;
        x(0, i) = Y(0) / Y(2);
        x(1, i) = Y(1) / Y(2);
    }

    r2ppnp::Config config;
    config.nr1 = 30;
    config.ransac_p = 0.7;

    r2ppnp::PnPResult result = r2ppnp::r2ppnp(X, x, config);

    if (result.success) {
        std::cout << "Success! num_trials = " << result.num_trials
                  << ", score = " << result.score << std::endl;
        std::cout << "R =\n" << result.R << std::endl;
        std::cout << "t = " << result.t.transpose() << std::endl;

        double dR = r2ppnp::calc_dR(R_gt, result.R);
        double dt = r2ppnp::calc_dt(t_gt, result.t);
        std::cout << "dR = " << dR << " deg, dt = " << dt << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }

    return 0;
}
