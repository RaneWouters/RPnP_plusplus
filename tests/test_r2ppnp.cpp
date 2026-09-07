#include "r2ppnp/r2ppnp.h"
#include "r2ppnp/hough_voting.h"
#include "r2ppnp/utils.h"

#include <Eigen/Geometry>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error(
            std::string("check failed at line ") + std::to_string(line) + ": " + expression);
    }
}

#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

bool approximately_equal(double lhs, double rhs, double tolerance = 1e-9) {
    return std::abs(lhs - rhs) <= tolerance;
}

void expect_invalid_argument(const std::function<void()>& function) {
    bool thrown = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    CHECK(thrown);
}

struct Scene {
    Eigen::Matrix3Xd world;
    Eigen::Matrix2Xd pixels;
    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

Scene make_scene(int point_count = 80) {
    Scene scene;
    scene.K << 900.0, 0.0, 320.0,
               0.0, 1100.0, 240.0,
               0.0, 0.0, 1.0;
    scene.R = (
        Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ())).toRotationMatrix();
    scene.t = Eigen::Vector3d(0.2, -0.1, 4.5);
    scene.world.resize(3, point_count);
    scene.pixels.resize(2, point_count);

    std::mt19937 generator(41);
    std::uniform_real_distribution<double> xy(-1.5, 1.5);
    std::uniform_real_distribution<double> z(-0.5, 1.5);
    for (int i = 0; i < point_count; ++i) {
        scene.world.col(i) = Eigen::Vector3d(xy(generator), xy(generator), z(generator));
        const Eigen::Vector3d camera = scene.R * scene.world.col(i) + scene.t;
        const Eigen::Vector3d projected = scene.K * camera;
        scene.pixels.col(i) = projected.head<2>() / projected(2);
    }
    return scene;
}

void test_math_and_coordinates() {
    CHECK(approximately_equal(r2ppnp::safe_acos(1.0), 0.0));
    CHECK(approximately_equal(r2ppnp::safe_acos(-2.0), r2ppnp::kPi));
    CHECK(approximately_equal(r2ppnp::normalize_angle(2.0 * r2ppnp::kPi), 0.0));
    CHECK(approximately_equal(r2ppnp::normalize_angle(-r2ppnp::kPi), -r2ppnp::kPi));

    Eigen::Matrix3d K;
    K << 1000.0, 0.0, 320.0,
         0.0, 800.0, 240.0,
         0.0, 0.0, 1.0;
    Eigen::Matrix2Xd normalized(2, 3);
    normalized << 0.1, -0.2, 0.0,
                  0.3, 0.1, 0.0;
    const Eigen::Matrix2Xd pixels = r2ppnp::xxn2xx(K, normalized);
    const Eigen::Matrix2Xd round_trip = r2ppnp::xx2xxn(K, pixels);
    CHECK((normalized - round_trip).norm() < 1e-12);

    Eigen::Matrix3Xd behind(3, 1);
    behind << 0.0, 0.0, -1.0;
    Eigen::Matrix2Xd observation = Eigen::Matrix2Xd::Zero(2, 1);
    const std::vector<bool> behind_inliers = r2ppnp::calcInliers(
        Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), behind,
        observation, K, 10.0);
    CHECK(!behind_inliers[0]);

    Eigen::MatrixXd histogram = Eigen::MatrixXd::Zero(5, 5);
    histogram(1, 1) = 10.0;
    histogram(3, 3) = 6.0;
    Eigen::MatrixXd dilated;
    r2ppnp::internal::dilate_3x3(histogram, dilated);
    CHECK(r2ppnp::internal::find_local_peaks(histogram, dilated, 1.0, 0.7).size() == 1);
    CHECK(r2ppnp::internal::find_local_peaks(histogram, dilated, 1.0, 0.5).size() == 2);
}

void test_solver_and_result_contract() {
    const Scene scene = make_scene();
    r2ppnp::Config config;
    config.seed = 3;
    config.max_trials = 500;
    const r2ppnp::PnPResult result =
        r2ppnp::r2ppnp_from_pixels(scene.world, scene.pixels, scene.K, 2.0, config);

    CHECK(result.success);
    CHECK(result.message == "success");
    CHECK(result.num_trials > 0 && result.num_trials <= config.max_trials);
    CHECK(result.num_inliers == scene.world.cols());
    CHECK(approximately_equal(result.score, 1.0));
    CHECK(result.inliers.size() == static_cast<std::size_t>(scene.world.cols()));
    CHECK(result.weights.size() == scene.world.cols());
    CHECK(result.errors.size() == scene.world.cols());
    CHECK(result.errors.allFinite());
    CHECK((result.R.transpose() * result.R - Eigen::Matrix3d::Identity()).norm() < 1e-10);
    CHECK(std::abs(result.R.determinant() - 1.0) < 1e-10);
    CHECK(r2ppnp::calc_dR(scene.R, result.R) < 0.05);
    CHECK(r2ppnp::calc_dt(scene.t, result.t) < 1e-3);

    const r2ppnp::PnPResult scaled_K_result =
        r2ppnp::r2ppnp_from_pixels(scene.world, scene.pixels, 2.0 * scene.K, 2.0, config);
    CHECK(scaled_K_result.success);
    CHECK(scaled_K_result.num_trials == result.num_trials);
    CHECK((scaled_K_result.R - result.R).norm() < 1e-10);
    CHECK((scaled_K_result.t - result.t).norm() < 1e-10);
    CHECK(scaled_K_result.num_inliers == result.num_inliers);

    r2ppnp::Config low_resolution_config = config;
    low_resolution_config.nr1 = 3;
    low_resolution_config.max_trials = 5;
    low_resolution_config.finalize = false;
    const r2ppnp::PnPResult low_resolution_result = r2ppnp::r2ppnp_from_pixels(
        scene.world, scene.pixels, scene.K, 2.0, low_resolution_config);
    CHECK(low_resolution_result.num_trials >= 1 && low_resolution_result.num_trials <= 5);

    const r2ppnp::PnPResult refined = r2ppnp::refine(
        scene.world, scene.pixels, scene.K, result.R, result.t, 2.0, config);
    CHECK(refined.success);
    CHECK(refined.num_trials == 0);
    CHECK(refined.num_inliers == scene.world.cols());
    CHECK(refined.score == 1.0);
}

void test_degenerate_inputs_terminate() {
    Eigen::Matrix3Xd world = Eigen::Matrix3Xd::Zero(3, 8);
    Eigen::Matrix2Xd normalized = Eigen::Matrix2Xd::Zero(2, 8);
    r2ppnp::Config config;
    config.max_trials = 100;
    const r2ppnp::PnPResult result = r2ppnp::r2ppnp(world, normalized, config);
    CHECK(!result.success);
    CHECK(result.num_trials == 28); // min(max_trials, 8 choose 2)
    CHECK(result.inliers.size() == 8);
    CHECK(result.weights.size() == 8);
    CHECK(result.errors.size() == 8);
}

void test_invalid_inputs_are_rejected() {
    const Scene scene = make_scene(10);

    Eigen::Matrix2Xd short_pixels = scene.pixels.leftCols(9);
    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(scene.world, short_pixels, scene.K);
    });

    Eigen::Matrix3Xd too_few_world = scene.world.leftCols(5);
    Eigen::Matrix2Xd too_few_pixels = scene.pixels.leftCols(5);
    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(too_few_world, too_few_pixels, scene.K);
    });

    Eigen::Matrix3Xd nan_world = scene.world;
    nan_world(0, 0) = std::numeric_limits<double>::quiet_NaN();
    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(nan_world, scene.pixels, scene.K);
    });

    Eigen::Matrix3d singular_K = Eigen::Matrix3d::Zero();
    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(scene.world, scene.pixels, singular_K);
    });

    Eigen::Matrix3d negative_scale_K = -scene.K;
    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(scene.world, scene.pixels, negative_scale_K);
    });

    r2ppnp::Config invalid_config;
    invalid_config.nr1 = -1;
    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(
            scene.world, scene.pixels, scene.K, 10.0, invalid_config);
    });

    invalid_config = r2ppnp::Config{};
    invalid_config.ransac_p = 1.5;
    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(
            scene.world, scene.pixels, scene.K, 10.0, invalid_config);
    });

    expect_invalid_argument([&] {
        (void)r2ppnp::r2ppnp_from_pixels(scene.world, scene.pixels, scene.K, -1.0);
    });

    const Eigen::Matrix3d invalid_rotation = 2.0 * Eigen::Matrix3d::Identity();
    expect_invalid_argument([&] {
        (void)r2ppnp::refine(
            scene.world, scene.pixels, scene.K, invalid_rotation, scene.t);
    });
}

} // namespace

int main() {
    try {
        test_math_and_coordinates();
        test_solver_and_result_contract();
        test_degenerate_inputs_terminate();
        test_invalid_inputs_are_rejected();
    std::cout << "All RPnP++ tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
