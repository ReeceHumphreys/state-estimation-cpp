#pragma once

#include "filters/detail/storage.hpp"
#include "filters/results.hpp"

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <type_traits>

namespace filters {

/// Kalman filter, linear and extended. The measurement size NZ is per call, so one filter takes
/// any mix of sensors. For nonlinear models the caller supplies the propagated state / predicted
/// measurement and their Jacobians (predict_nonlinear, update_nonlinear).
///
/// Trivially copyable (memcpy / telemetry safe). Updates that cannot be applied (S not positive
/// definite) return std::nullopt and leave the state unchanged.
template <typename T, int NX> struct KalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0, "NX must be > 0");

    template <int N> using Vec = Eigen::Matrix<T, N, 1>;
    template <int R, int C> using Mat = Eigen::Matrix<T, R, C>;
    using VecX = Vec<NX>;
    using MatXX = Mat<NX, NX>;

    /// x = 0, P = I
    KalmanFilter() = default;
    KalmanFilter(const VecX& x_0, const MatXX& P_0) {
        reset(x_0, P_0);
    }

    void reset(const VecX& x_0, const MatXX& P_0) {
        x() = x_0;
        P() = P_0;
    }

    VecX state() const {
        return x();
    }
    MatXX covariance() const {
        return P();
    }

    /// Mutable views onto the state, e.g. to reset an error state after injecting it.
    Eigen::Map<VecX> x() {
        return Eigen::Map<VecX>(x_.data());
    }
    Eigen::Map<const VecX> x() const {
        return Eigen::Map<const VecX>(x_.data());
    }
    Eigen::Map<MatXX> P() {
        return Eigen::Map<MatXX>(P_.data());
    }
    Eigen::Map<const MatXX> P() const {
        return Eigen::Map<const MatXX>(P_.data());
    }

    /// Linear prediction x <- Phi x, P <- Phi P Phi^T + Q. Q may be singular.
    void predict(const MatXX& Phi, const MatXX& Q) {
        predict_nonlinear(VecX(Phi * x()), Phi, Q);
    }

    /// Linear prediction with process noise w ~ N(0, Q_w) entering as x <- Phi x + Gamma w.
    template <int NW>
    void predict(const MatXX& Phi, const Mat<NX, NW>& Gamma, const Mat<NW, NW>& Q_w) {
        predict(Phi, MatXX(Gamma * Q_w * Gamma.transpose()));
    }

    /// EKF prediction with an externally propagated state.
    /// @param x_pred Propagated state f(x)
    /// @param Phi State transition matrix (Jacobian of f)
    /// @param Q Process noise covariance
    void predict_nonlinear(const VecX& x_pred, const MatXX& Phi, const MatXX& Q) {
        x() = x_pred;
        P() = Phi * P() * Phi.transpose() + Q;
    }

    /// Innovation of z = H x + v, v ~ N(0, R), without applying it (e.g. to gate outliers).
    template <int NZ>
    std::optional<Innovation<T, NZ>> innovation(const Vec<NZ>& z, const Mat<NZ, NX>& H,
                                                const Mat<NZ, NZ>& R) const {
        const Vec<NZ> nu = z - H * x();
        const Mat<NZ, NZ> S = H * P() * H.transpose() + R;
        const Eigen::LLT<Mat<NZ, NZ>> llt(S);
        if (llt.info() != Eigen::Success)
            return std::nullopt;
        return Innovation<T, NZ>{detail::to_array(nu), detail::to_array(S),
                                 llt.matrixL().solve(nu).squaredNorm()};
    }

    /// Linear update for z = H x + v, v ~ N(0, R).
    template <int NZ>
    std::optional<KalmanUpdate<T, NX, NZ>> update(const Vec<NZ>& z, const Mat<NZ, NX>& H,
                                                  const Mat<NZ, NZ>& R) {
        return update_innovation<NZ>(z - H * x(), H, R);
    }

    /// EKF update with an externally predicted measurement.
    /// @param z_hat Predicted measurement h(x)
    /// @param H Observation matrix (Jacobian of h)
    template <int NZ>
    std::optional<KalmanUpdate<T, NX, NZ>> update_nonlinear(const Vec<NZ>& z, const Vec<NZ>& z_hat,
                                                            const Mat<NZ, NX>& H,
                                                            const Mat<NZ, NZ>& R) {
        return update_innovation<NZ>(z - z_hat, H, R);
    }

    /// Update with an explicit innovation (error-state filters): nu = y - H dx, y = z - h(x_ref).
    template <int NZ>
    std::optional<KalmanUpdate<T, NX, NZ>> update_innovation(const Vec<NZ>& nu,
                                                             const Mat<NZ, NX>& H,
                                                             const Mat<NZ, NZ>& R) {
        const Mat<NX, NZ> PHt = P() * H.transpose();
        const Mat<NZ, NZ> S = H * PHt + R;
        const Eigen::LLT<Mat<NZ, NZ>> llt(S);
        if (llt.info() != Eigen::Success)
            return std::nullopt;
        // K = P H^T S^-1, solved as S K^T = H P (P and S symmetric) instead of forming S^-1
        const Mat<NX, NZ> K = llt.solve(PHt.transpose()).transpose();

        // Joseph form keeps P symmetric positive definite under round-off
        const MatXX I_KH = MatXX::Identity() - K * H;
        x() += K * nu;
        P() = I_KH * P() * I_KH.transpose() + K * R * K.transpose();
        return KalmanUpdate<T, NX, NZ>{detail::to_array(nu), detail::to_array(S),
                                       detail::to_array(K), llt.matrixL().solve(nu).squaredNorm()};
    }

  private:
    std::array<T, NX> x_{};
    std::array<T, NX * NX> P_ = detail::identity<T, NX>();
};

} // namespace filters
