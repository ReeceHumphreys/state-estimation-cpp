#pragma once

#include <Eigen/Dense>
#include <type_traits>

namespace filters {

/// Square-root information filter. Stores the information form R x = b with R upper triangular
/// (P = R^-1 R^-T), and keeps it triangular with Householder QR, which stays well conditioned where
/// the covariance form loses precision.
template <typename T, int NX, int NZ> struct SquareRootInformationFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0 && NZ > 0, "Dimensions must be > 0");

    using VecX = Eigen::Matrix<T, NX, 1>;
    using VecZ = Eigen::Matrix<T, NZ, 1>;
    using MatXX = Eigen::Matrix<T, NX, NX>;
    using MatZZ = Eigen::Matrix<T, NZ, NZ>;
    using MatZX = Eigen::Matrix<T, NZ, NX>;

    MatXX R = MatXX::Identity(); // Square-root information matrix (upper triangular)
    VecX b = VecX::Zero();       // Information state, R x = b

    SquareRootInformationFilter() = default;
    SquareRootInformationFilter(const VecX& x_0, const MatXX& P_0) {
        reset(x_0, P_0);
    }

    void reset(const VecX& x_0, const MatXX& P_0) {
        const MatXX P_inv = P_0.llt().solve(MatXX::Identity());
        R = P_inv.llt().matrixU();
        b = R * x_0;
    }

    VecX state() const {
        return R.template triangularView<Eigen::Upper>().solve(b);
    }

    MatXX covariance() const {
        const MatXX R_inv = R.template triangularView<Eigen::Upper>().solve(MatXX::Identity());
        return R_inv * R_inv.transpose();
    }

    /// Time update without process noise.
    /// @param Phi State transition matrix
    void predict(const MatXX& Phi) {
        const MatXX R_tilde = propagated(Phi);
        Eigen::HouseholderQR<MatXX> qr(R_tilde);
        R = qr.matrixQR().template triangularView<Eigen::Upper>();
        b = qr.householderQ().transpose() * b;
    }

    /// Time update with process noise w ~ N(0, Q) entering as x_{k+1} = Phi x_k + Gamma w.
    /// @param Phi State transition matrix
    /// @param Gamma Process noise mapping (NX x NW)
    /// @param Q Process noise covariance (NW x NW), must be positive definite
    template <int NW>
    void predict(const MatXX& Phi, const Eigen::Matrix<T, NX, NW>& Gamma, const Eigen::Matrix<T, NW, NW>& Q) {
        using MatWW = Eigen::Matrix<T, NW, NW>;
        constexpr int N = NW + NX;

        const MatXX R_tilde = propagated(Phi);
        const MatWW Q_inv = Q.llt().solve(MatWW::Identity());
        const MatWW R_w = Q_inv.llt().matrixU();

        // Triangularise [R_w 0; -R~ Gamma  R~] [w; x] = [0; b]; the lower-right block is the new R.
        Eigen::Matrix<T, N, N> A;
        A.template topLeftCorner<NW, NW>() = R_w;
        A.template topRightCorner<NW, NX>().setZero();
        A.template bottomLeftCorner<NX, NW>() = -R_tilde * Gamma;
        A.template bottomRightCorner<NX, NX>() = R_tilde;
        Eigen::Matrix<T, N, 1> rhs;
        rhs << Eigen::Matrix<T, NW, 1>::Zero(), b;

        Eigen::HouseholderQR<Eigen::Matrix<T, N, N>> qr(A);
        R = qr.matrixQR().template bottomRightCorner<NX, NX>().template triangularView<Eigen::Upper>();
        b = (qr.householderQ().transpose() * rhs).template tail<NX>();
    }

    /// Measurement update for z = H x + v, v ~ N(0, R_meas).
    /// @param z Measurement (for an error-state filter, the prefit residual z - h(x_ref))
    /// @param H Observation matrix
    /// @param R_meas Measurement noise covariance, must be positive definite
    void update(const VecZ& z, const MatZX& H, const MatZZ& R_meas) {
        constexpr int N = NX + NZ;

        // Whiten so the measurement noise is unit: V^-1 z = V^-1 H x + V^-1 v, with R_meas = V V^T.
        const Eigen::LLT<MatZZ> llt(R_meas);
        const auto V = llt.matrixL();
        const MatZX H_w = V.solve(H);
        const VecZ z_w = V.solve(z);

        Eigen::Matrix<T, N, NX> A;
        A << R, H_w;
        Eigen::Matrix<T, N, 1> rhs;
        rhs << b, z_w;

        Eigen::HouseholderQR<Eigen::Matrix<T, N, NX>> qr(A);
        R = qr.matrixQR().template topRows<NX>().template triangularView<Eigen::Upper>();
        b = (qr.householderQ().transpose() * rhs).template head<NX>();
    }

  private:
    /// R Phi^-1, the information matrix mapped through the transition.
    MatXX propagated(const MatXX& Phi) const {
        return Phi.transpose().partialPivLu().solve(R.transpose()).transpose();
    }
};

} // namespace filters
