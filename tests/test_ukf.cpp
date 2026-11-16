#include "filters/kalman.hpp"
#include "filters/ukf.hpp"

#include <catch2/catch_test_macros.hpp>
#include <type_traits>

using KF = filters::KalmanFilter<double, 2>;
using UKF = filters::UnscentedKalmanFilter<double, 2>;

static_assert(std::is_trivially_copyable_v<filters::UnscentedKalmanFilter<float, 4>> &&
              std::is_standard_layout_v<filters::UnscentedKalmanFilter<float, 4>>);

namespace {
struct Linear {
    KF::VecX x0{0.5, -1.0};
    KF::MatXX P0, Phi;
    KF::MatXX Q = KF::MatXX::Identity() * 0.05;
    Eigen::Matrix<double, 1, 2> H{1.0, 0.3};
    Eigen::Matrix<double, 1, 1> R = Eigen::Matrix<double, 1, 1>::Constant(0.2);
    Linear() {
        P0 << 3.0, 0.5, 0.5, 1.5;
        Phi << 1.0, 0.2, 0.0, 1.0;
    }
    auto f() const {
        return [this](const UKF::MatXsig& X) { return UKF::MatXsig(Phi * X); };
    }
    auto h() const {
        return [this](const UKF::MatXsig& X) { return UKF::MatZsig<1>(H * X); };
    }
};
// alpha = 1 -> lambda = 0: avoids the huge cancelling weights of alpha = 1e-3
const UKF::Params kParams{1.0};
} // namespace

// With linear models the unscented transform is exact, so the UKF must reproduce the KF. Two
// updates after one predict guards against reusing stale sigma points from predict().
TEST_CASE("UKF matches KF on linear models, including sequential updates") {
    const Linear m;
    KF kf(m.x0, m.P0);
    UKF ukf(m.x0, m.P0, kParams);

    kf.predict(m.Phi, m.Q);
    REQUIRE(ukf.predict(m.f(), m.Q));

    for (double zv : {1.0, 0.8}) {
        const Eigen::Matrix<double, 1, 1> z(zv);
        const auto ku = kf.update(z, m.H, m.R);
        const auto uu = ukf.update(z, m.h(), m.R);
        REQUIRE(ku);
        REQUIRE(uu);
        REQUIRE(ukf.x().isApprox(kf.x(), 1e-10));
        REQUIRE(ukf.P().isApprox(kf.P(), 1e-10));
        REQUIRE(std::abs(uu->nis - ku->nis) < 1e-10);
    }
}

TEST_CASE("UKF innovation matches the KF and leaves the state unchanged") {
    const Linear m;
    KF kf(m.x0, m.P0);
    UKF ukf(m.x0, m.P0, kParams);
    const Eigen::Matrix<double, 1, 1> z(0.9);

    const auto ki = kf.innovation(z, m.H, m.R);
    const auto ui = ukf.innovation(z, m.h(), m.R);
    REQUIRE(ki);
    REQUIRE(ui);
    REQUIRE(std::abs(ui->nu[0] - ki->nu[0]) < 1e-10);
    REQUIRE(std::abs(ui->S[0] - ki->S[0]) < 1e-10);
    REQUIRE(ukf.x().isApprox(m.x0));
}

TEST_CASE("UKF reports a covariance that is not positive definite") {
    const Linear m;
    UKF ukf(m.x0, -m.P0, kParams);

    REQUIRE_FALSE(ukf.predict(m.f(), m.Q));
    REQUIRE_FALSE(ukf.update(Eigen::Matrix<double, 1, 1>(1.0), m.h(), m.R));
    REQUIRE(ukf.x().isApprox(m.x0));
    REQUIRE(ukf.P().isApprox(-m.P0));
}
