#include "filters/kalman.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <optional>
#include <type_traits>

using KF1 = filters::KalmanFilter<double, 1>;
using KF3 = filters::KalmanFilter<double, 3>;
using V1 = Eigen::Matrix<double, 1, 1>;

// Filter state and results must be memcpy-able (telemetry, shared memory, checkpoints).
template <class F> constexpr bool kPod = std::is_trivially_copyable_v<F> && std::is_standard_layout_v<F>;
static_assert(kPod<filters::KalmanFilter<double, 6>>);
static_assert(kPod<filters::KalmanUpdate<double, 6, 2>>);
static_assert(kPod<filters::Innovation<double, 2>>);
static_assert(std::is_trivially_copyable_v<std::optional<filters::KalmanUpdate<double, 6, 2>>>);

static bool approx_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps * (1.0 + std::fabs(a) + std::fabs(b));
}

TEST_CASE("KalmanFilter (scalar) predict step matches closed-form") {
    KF1 kf(V1(2.0), V1(3.0));
    kf.predict(V1(4.0), V1(5.0));

    REQUIRE(approx_equal(kf.x()(0), 8.0));          // x = F x
    REQUIRE(approx_equal(kf.P()(0, 0), 53.0));      // P = F P F^T + Q = 4*3*4 + 5
}

TEST_CASE("KalmanFilter (scalar) update matches closed-form for H=1") {
    KF1 kf(V1(0.0), V1(1.0));

    // y = 1, S = 1 + 3 = 4, K = 1/4, x+ = 0.25, P+ (Joseph) = 0.75^2 + 0.25^2 * 3 = 0.75,
    // NIS = y^2 / S = 0.25
    const auto u = kf.update(V1(1.0), V1(1.0), V1(3.0));
    REQUIRE(u);
    REQUIRE(approx_equal(kf.x()(0), 0.25));
    REQUIRE(approx_equal(kf.P()(0, 0), 0.75));
    REQUIRE(approx_equal(u->nu[0], 1.0));
    REQUIRE(approx_equal(u->S[0], 4.0));
    REQUIRE(approx_equal(u->K[0], 0.25));
    REQUIRE(approx_equal(u->nis, 0.25));
}

TEST_CASE("KalmanFilter update uses H correctly") {
    KF1 kf(V1(0.0), V1(1.0));

    // H = 2: S = 7, K = 2/7, x+ = 2/7, I - KH = 3/7, P+ = 9/49 + 12/49 = 3/7
    REQUIRE(kf.update(V1(1.0), V1(2.0), V1(3.0)));
    REQUIRE(approx_equal(kf.x()(0), 2.0 / 7.0));
    REQUIRE(approx_equal(kf.P()(0, 0), 3.0 / 7.0));
}

TEST_CASE("KalmanFilter converges toward constant measurement when R small") {
    KF1 kf(V1(0.0), V1(10.0));
    for (int i = 0; i < 50; ++i) {
        kf.predict(V1(1.0), V1(1e-6));
        REQUIRE(kf.update(V1(5.0), V1(1.0), V1(1e-3)));
    }
    REQUIRE(std::fabs(kf.x()(0) - 5.0) < 1e-2);
    REQUIRE(kf.P()(0, 0) < 1e-2);
}

namespace {
struct Problem {
    KF3::VecX x0{1.0, 2.0, -0.5};
    KF3::MatXX P0, Phi;
    Problem() {
        P0 << 2.0, 0.4, 0.1, 0.4, 1.0, -0.2, 0.1, -0.2, 1.5;
        Phi << 1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.9;
    }
};
} // namespace

TEST_CASE("KalmanFilter nonlinear forms reduce to the linear ones for linear models") {
    const Problem p;
    KF3 lin(p.x0, p.P0), ext(p.x0, p.P0);
    const KF3::MatXX Q = KF3::MatXX::Identity() * 0.01;
    const Eigen::Matrix<double, 1, 3> H(1.0, -0.5, 0.2);

    lin.predict(p.Phi, Q);
    ext.predict_nonlinear(p.Phi * ext.x(), p.Phi, Q);
    REQUIRE(lin.update(V1(0.7), H, V1(0.3)));
    REQUIRE(ext.update_nonlinear(V1(0.7), V1(H * ext.x()), H, V1(0.3)));

    REQUIRE(ext.x().isApprox(lin.x(), 1e-14));
    REQUIRE(ext.P().isApprox(lin.P(), 1e-14));
}

TEST_CASE("KalmanFilter predict with Gamma matches the equivalent full Q") {
    const Problem p;
    KF3 a(p.x0, p.P0), b(p.x0, p.P0);
    Eigen::Matrix<double, 3, 2> Gamma;
    Gamma << 0.5, 0.0, 1.0, 0.1, 0.0, 1.0;
    const Eigen::Matrix2d Q_w = Eigen::Vector2d(0.2, 0.05).asDiagonal();

    a.predict(p.Phi, Gamma, Q_w);
    b.predict(p.Phi, KF3::MatXX(Gamma * Q_w * Gamma.transpose()));
    REQUIRE(a.P().isApprox(b.P(), 1e-14));
}

// One filter taking two sensor sizes: two scalar updates equal one stacked update when the
// measurement noise is uncorrelated.
TEST_CASE("KalmanFilter accepts any measurement size per call") {
    const Problem p;
    KF3 seq(p.x0, p.P0), stacked(p.x0, p.P0);
    const Eigen::Matrix<double, 1, 3> H1(1.0, 0.0, 0.0), H2(0.0, 1.0, 1.0);

    REQUIRE(seq.update(V1(1.2), H1, V1(0.1)));
    REQUIRE(seq.update(V1(2.1), H2, V1(0.4)));

    Eigen::Matrix<double, 2, 3> H;
    H << H1, H2;
    REQUIRE(stacked.update(Eigen::Vector2d(1.2, 2.1), H, Eigen::Matrix2d(Eigen::Vector2d(0.1, 0.4).asDiagonal())));

    REQUIRE(seq.x().isApprox(stacked.x(), 1e-12));
    REQUIRE(seq.P().isApprox(stacked.P(), 1e-12));
}

TEST_CASE("KalmanFilter innovation previews the update without applying it") {
    const Problem p;
    KF3 kf(p.x0, p.P0);
    Eigen::Matrix<double, 2, 3> H;
    H << 1.0, 0.0, 0.5, 0.0, 1.0, 0.0;
    const Eigen::Vector2d z(1.5, 1.0);
    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * 0.2;

    const auto inn = kf.innovation(z, H, R);
    REQUIRE(inn);
    REQUIRE(kf.x().isApprox(p.x0));

    const auto u = kf.update(z, H, R);
    REQUIRE(u);
    REQUIRE(inn->nu == u->nu);
    REQUIRE(inn->S == u->S);
    REQUIRE(approx_equal(inn->nis, u->nis));
}

TEST_CASE("KalmanFilter rejects an update whose S is not positive definite") {
    KF1 kf(V1(1.0), V1(0.5));
    REQUIRE_FALSE(kf.update(V1(2.0), V1(1.0), V1(-1.0))); // S = 0.5 - 1 < 0
    REQUIRE(kf.x()(0) == 1.0);
    REQUIRE(kf.P()(0, 0) == 0.5);
}
