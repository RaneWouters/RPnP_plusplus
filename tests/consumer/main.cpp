#include <r2ppnp/r2ppnp.h>

#include <Eigen/Core>

int main() {
    r2ppnp::Config config;
    config.max_trials = 10;
    Eigen::Matrix3Xd world = Eigen::Matrix3Xd::Zero(3, 6);
    Eigen::Matrix2Xd normalized = Eigen::Matrix2Xd::Zero(2, 6);
    const r2ppnp::PnPResult result = r2ppnp::r2ppnp(world, normalized, config);
    return result.success ? 1 : 0;
}
