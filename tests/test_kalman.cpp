#include "filters/kalman.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

static bool approx_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps * (1.0 + std::fabs(a) + std::fabs(b));
}

TEST_CASE("KalmanFilter (scalar) predict step matches closed-form") {
    // If your KF is vectorized, use NX=1, NZ=1.
    using KF = filters::LinearKalmanFilter<double, 1, 1>;

    KF kf;
    kf.x(0) = 2.0;
    kf.P(0, 0) = 3.0;

    KF::MatXX F;
    F << 4.0;
    KF::MatXX Q;
    Q << 5.0;

    kf.predict(F, Q);

    // x = F x
    REQUIRE(approx_equal(kf.x(0), 8.0));
    // P = F P F^T + Q = 4*3*4 + 5 = 53
    REQUIRE(approx_equal(kf.P(0, 0), 53.0));
}

TEST_CASE("KalmanFilter (scalar) update step matches closed-form for H=1") {
    using KF = filters::LinearKalmanFilter<double, 1, 1>;

    KF kf;
    kf.x(0) = 0.0;
    kf.P(0, 0) = 1.0;

    KF::VecZ z;
    z << 1.0;
    KF::MatZX H;
    H << 1.0;
    KF::MatZZ R;
    R << 3.0;

    // Closed-form:
    // y = z - Hx = 1 - 0 = 1
    // S = HPH^T + R = 1 + 3 = 4
    // K = P H^T S^{-1} = 1/4 = 0.25
    // x+ = x + K y = 0 + 0.25*1 = 0.25
    // P+ (Joseph) = (1-K)^2 P + K^2 R = 0.75^2*1 + 0.25^2*3 = 0.5625 + 0.1875 = 0.75

    kf.update(z, H, R);

    REQUIRE(approx_equal(kf.x(0), 0.25));
    REQUIRE(approx_equal(kf.P(0, 0), 0.75));
}

TEST_CASE("KalmanFilter update uses H correctly") {
    using KF = filters::LinearKalmanFilter<double, 1, 1>;

    KF kf;
    kf.x(0) = 0.0;
    kf.P(0, 0) = 1.0;

    KF::VecZ z;
    z << 1.0;
    KF::MatZX H;
    H << 2.0; // measurement is z = 2x + v
    KF::MatZZ R;
    R << 3.0;

    // Closed-form with H=2:
    // y = 1 - 2*0 = 1
    // S = 2*1*2 + 3 = 7
    // K = 1*2 / 7 = 2/7
    // x+ = 0 + (2/7)*1 = 2/7
    // P+ (Joseph) scalar:
    // I-KH = 1 - (2/7)*2 = 1 - 4/7 = 3/7
    // P+ = (3/7)^2 * 1 + (2/7)^2 * 3 = 9/49 + 12/49 = 21/49 = 3/7

    kf.update(z, H, R);

    REQUIRE(approx_equal(kf.x(0), 2.0 / 7.0));
    REQUIRE(approx_equal(kf.P(0, 0), 3.0 / 7.0));
}

TEST_CASE("KalmanFilter converges toward constant measurement when R small") {
    using KF = filters::LinearKalmanFilter<double, 1, 1>;

    KF kf;
    kf.x(0) = 0.0;
    kf.P(0, 0) = 10.0;

    KF::MatXX F;
    F << 1.0;
    KF::MatXX Q;
    Q << 1e-6; // nearly constant process
    KF::MatZX H;
    H << 1.0;
    KF::MatZZ R;
    R << 1e-3; // fairly confident measurements

    KF::VecZ z;
    z << 5.0;

    for (int i = 0; i < 50; ++i) {
        kf.predict(F, Q);
        kf.update(z, H, R);
    }

    REQUIRE(std::fabs(kf.x(0) - 5.0) < 1e-2);
    REQUIRE(kf.P(0, 0) < 1e-2);
}
