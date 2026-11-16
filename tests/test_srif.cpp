#include "filters/kalman.hpp"
#include "filters/srif.hpp"

#include <catch2/catch_test_macros.hpp>

// The SRIF is the Kalman filter in information form, so on the same problem the two must agree.
TEST_CASE("SRIF matches the linear Kalman filter on a constant-velocity track") {
    using KF = filters::LinearKalmanFilter<double, 2, 1>;
    using SRIF = filters::SquareRootInformationFilter<double, 2, 1>;

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
    const KF::MatZX H(1.0, 0.0);
    const KF::MatZZ Rm = KF::MatZZ::Constant(0.25);

    const double zs[] = {1.2, 1.1, 0.7, 0.4, 0.2};
    for (int k = 0; k < 5; ++k) {
        if (k % 2 == 0) {
            kf.predict(Phi, Gamma * q * Gamma.transpose());
            srif.predict<1>(Phi, Gamma, q);
        } else {
            kf.predict(Phi, KF::MatXX::Zero());
            srif.predict(Phi);
        }
        const KF::VecZ z = KF::VecZ::Constant(zs[k]);
        kf.update(z, H, Rm);
        srif.update(z, H, Rm);

        REQUIRE(srif.state().isApprox(kf.x, 1e-10));
        REQUIRE(srif.covariance().isApprox(kf.P, 1e-10));
    }
}
