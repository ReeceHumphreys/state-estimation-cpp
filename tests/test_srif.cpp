#include "filters/kalman.hpp"
#include "filters/srif.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<filters::SquareRootInformationFilter<double, 6>> &&
              std::is_standard_layout_v<filters::SquareRootInformationFilter<double, 6>>);
static_assert(std::is_trivially_copyable_v<filters::SrifUpdate<double, 2>>);

// The SRIF is the Kalman filter in information form, so on the same problem the two must agree.
TEST_CASE("SRIF matches the Kalman filter on a constant-velocity track") {
    using KF = filters::KalmanFilter<double, 2>;
    using SRIF = filters::SquareRootInformationFilter<double, 2>;

    const KF::VecX x0(1.0, -0.5);
    KF::MatXX P0;
    P0 << 4.0, 0.3, 0.3, 2.0;
    KF kf(x0, P0);
    SRIF srif(x0, P0);

    const double dt = 0.5;
    KF::MatXX Phi;
    Phi << 1.0, dt, 0.0, 1.0;
    const Eigen::Vector2d Gamma(dt * dt / 2.0, dt);
    const Eigen::Matrix<double, 1, 1> q = Eigen::Matrix<double, 1, 1>::Constant(0.1);
    const Eigen::Matrix<double, 1, 2> H(1.0, 0.0);
    const Eigen::Matrix<double, 1, 1> Rm = Eigen::Matrix<double, 1, 1>::Constant(0.25);

    const double zs[] = {1.2, 1.1, 0.7, 0.4, 0.2};
    for (int k = 0; k < 5; ++k) {
        if (k % 2 == 0) {
            kf.predict(Phi, Gamma, q);
            REQUIRE(srif.predict(Phi, Gamma, q));
        } else {
            kf.predict(Phi, KF::MatXX::Zero());
            srif.predict(Phi);
        }
        const Eigen::Matrix<double, 1, 1> z(zs[k]);
        const auto ku = kf.update(z, H, Rm);
        const auto su = srif.update(z, H, Rm);
        REQUIRE(ku);
        REQUIRE(su);

        REQUIRE(srif.state().isApprox(kf.x(), 1e-10));
        REQUIRE(srif.covariance().isApprox(kf.P(), 1e-10));
        REQUIRE(std::abs(su->nu[0] - ku->nu[0]) < 1e-10);
        REQUIRE(std::abs(su->nis - ku->nis) < 1e-10);
    }
}

namespace {
using KF3 = filters::KalmanFilter<double, 3>;
using SRIF3 = filters::SquareRootInformationFilter<double, 3>;
struct Vector {
    KF3::MatXX P0, Phi;
    KF3::VecX x0{0.3, -1.2, 0.8};
    Eigen::Matrix<double, 3, 2> Gamma;
    Eigen::Matrix2d Q, Rm;
    Eigen::Matrix<double, 2, 3> H;
    Vector() {
        P0 << 3.0, 0.4, -0.2, 0.4, 2.0, 0.3, -0.2, 0.3, 1.5;
        Phi << 1.0, 0.2, 0.02, -0.1, 0.9, 0.2, 0.05, 0.0, 1.1;
        Gamma << 0.5, 0.1, 0.2, 1.0, -0.3, 0.4;
        Q << 0.2, 0.05, 0.05, 0.1;
        H << 1.0, 0.0, 0.5, 0.2, -1.0, 0.0;
        Rm << 0.3, 0.1, 0.1, 0.4;
    }
};
} // namespace

// NW > 1 and NZ > 1 exercise the noise reordering and the structured elimination across columns.
TEST_CASE("SRIF matches the Kalman filter with vector process noise and measurements") {
    const Vector m;
    KF3 kf(m.x0, m.P0);
    SRIF3 srif(m.x0, m.P0);
    SRIF3 srif_inv(m.x0, m.P0);
    const KF3::MatXX Phi_inv = m.Phi.inverse();

    for (int k = 0; k < 6; ++k) {
        kf.predict(m.Phi, m.Gamma, m.Q);
        REQUIRE(srif.predict(m.Phi, m.Gamma, m.Q));
        REQUIRE(srif_inv.predict_from_inverse(Phi_inv, m.Gamma, m.Q));
        const Eigen::Vector2d z(0.1 * k, 1.0 - 0.2 * k);

        const auto si = srif.innovation(z, m.H, m.Rm);
        const auto ki = kf.innovation(z, m.H, m.Rm);
        REQUIRE(si);
        REQUIRE(ki);
        REQUIRE(Eigen::Map<const Eigen::Matrix2d>(si->S.data())
                    .isApprox(Eigen::Map<const Eigen::Matrix2d>(ki->S.data()), 1e-10));

        const auto ku = kf.update(z, m.H, m.Rm);
        const auto su = srif.update(z, m.H, m.Rm);
        REQUIRE(srif_inv.update(z, m.H, m.Rm));
        REQUIRE(std::abs(su->nis - ku->nis) < 1e-10);

        REQUIRE(srif.state().isApprox(kf.x(), 1e-10));
        REQUIRE(srif.covariance().isApprox(kf.P(), 1e-10));
        REQUIRE(srif_inv.state().isApprox(kf.x(), 1e-10));
        REQUIRE(srif_inv.covariance().isApprox(kf.P(), 1e-10));
    }
}

TEST_CASE("SRIF reports noise covariances that are not positive definite") {
    const Vector m;
    SRIF3 srif(m.x0, m.P0);
    const KF3::VecX x = srif.state();

    REQUIRE_FALSE(srif.update(Eigen::Vector2d(1.0, 1.0), m.H, Eigen::Matrix2d(-m.Rm)));
    REQUIRE_FALSE(srif.predict(m.Phi, m.Gamma, Eigen::Matrix2d(-m.Q)));
    REQUIRE_FALSE(srif.reset(m.x0, -m.P0));
    REQUIRE(srif.state().isApprox(x));
}
