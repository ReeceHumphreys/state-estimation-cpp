#pragma once

#include "filters/detail/storage.hpp"
#include "filters/results.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <optional>
#include <type_traits>

namespace filters {

/// Unscented Kalman filter. Models are applied to all 2 NX + 1 sigma points at once (one point
/// per column), so they can vectorize across them. The measurement size NZ is per call.
///
/// Trivially copyable. Steps that cannot be applied (P or S not positive definite) report failure
/// and leave the state unchanged. Sigma points are redrawn from the current (x, P) on every call,
/// so predict() and update() may be called in any order.
template <typename T, int NX> struct UnscentedKalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0, "NX must be > 0");

    static constexpr int kSigmaCount = 2 * NX + 1;

    template <int N> using Vec = Eigen::Matrix<T, N, 1>;
    template <int R, int C> using Mat = Eigen::Matrix<T, R, C>;
    using VecX = Vec<NX>;
    using MatXX = Mat<NX, NX>;
    using MatXsig = Mat<NX, kSigmaCount>; // sigma points, one per column
    template <int NZ> using MatZsig = Mat<NZ, kSigmaCount>;

    /// Sigma-point spread (Wan & van der Merwe).
    struct Params {
        T alpha = T(1e-3);
        T beta = T(2);
        T kappa = T(0);
    };

    /// x = 0, P = I, default Params
    UnscentedKalmanFilter() = default;
    UnscentedKalmanFilter(const VecX& x_0, const MatXX& P_0) {
        reset(x_0, P_0);
    }
    UnscentedKalmanFilter(const VecX& x_0, const MatXX& P_0, const Params& params)
        : params_(params) {
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

    /// @param f Dynamics on all sigma points: MatXsig -> MatXsig
    /// @param Q Process noise covariance
    /// @return false (state unchanged) if P is not positive definite
    template <class F> [[nodiscard]] bool predict(F&& f, const MatXX& Q) {
        MatXsig X;
        if (!sigma_points(X))
            return false;
        const MatXsig Xp = f(X);

        VecSig Wm, Wc;
        weights(Wm, Wc);

        x() = Xp * Wm;
        const MatXsig dX = Xp.colwise() - VecX(x());
        P() = dX * Wc.asDiagonal() * dX.transpose() + Q;
        return true;
    }

    /// Innovation of z = h(x) + v, v ~ N(0, R), without applying it (e.g. to gate outliers).
    /// @param h Measurement model on all sigma points: MatXsig -> MatZsig<NZ>
    template <int NZ, class H>
    std::optional<Innovation<T, NZ>> innovation(const Vec<NZ>& z, H&& h,
                                                const Mat<NZ, NZ>& R) const {
        const auto m = predict_measurement<NZ>(h, R);
        if (!m)
            return std::nullopt;
        const Vec<NZ> nu = z - m->z_hat;
        return Innovation<T, NZ>{detail::to_array(nu), detail::to_array(m->S),
                                 m->llt.matrixL().solve(nu).squaredNorm()};
    }

    /// @param h Measurement model on all sigma points: MatXsig -> MatZsig<NZ>
    template <int NZ, class H>
    std::optional<KalmanUpdate<T, NX, NZ>> update(const Vec<NZ>& z, H&& h, const Mat<NZ, NZ>& R) {
        const auto m = predict_measurement<NZ>(h, R);
        if (!m)
            return std::nullopt;
        const Vec<NZ> nu = z - m->z_hat;
        const Mat<NX, NZ> K = m->llt.solve(m->Pxz.transpose()).transpose();

        x() += K * nu;
        P() -= K * m->S * K.transpose();
        return KalmanUpdate<T, NX, NZ>{detail::to_array(nu), detail::to_array(m->S),
                                       detail::to_array(K),
                                       m->llt.matrixL().solve(nu).squaredNorm()};
    }

  private:
    using VecSig = Vec<kSigmaCount>;

    std::array<T, NX> x_{};
    std::array<T, NX * NX> P_ = detail::identity<T, NX>();
    Params params_;

    template <int NZ> struct PredictedMeasurement {
        Vec<NZ> z_hat;
        Mat<NZ, NZ> S;
        Mat<NX, NZ> Pxz;
        Eigen::LLT<Mat<NZ, NZ>> llt; // of S
    };

    template <int NZ, class H>
    std::optional<PredictedMeasurement<NZ>> predict_measurement(H& h, const Mat<NZ, NZ>& R) const {
        MatXsig X;
        if (!sigma_points(X))
            return std::nullopt;
        const MatZsig<NZ> Z = h(X);

        VecSig Wm, Wc;
        weights(Wm, Wc);

        PredictedMeasurement<NZ> m;
        m.z_hat = Z * Wm;
        const MatZsig<NZ> dZ = Z.colwise() - m.z_hat;
        const MatXsig dX = X.colwise() - VecX(x());
        m.S = dZ * Wc.asDiagonal() * dZ.transpose() + R;
        m.Pxz = dX * Wc.asDiagonal() * dZ.transpose();
        m.llt.compute(m.S);
        if (m.llt.info() != Eigen::Success)
            return std::nullopt;
        return m;
    }

    T lambda() const {
        return params_.alpha * params_.alpha * (T(NX) + params_.kappa) - T(NX);
    }

    void weights(VecSig& Wm, VecSig& Wc) const {
        const T c = T(NX) + lambda();

        Wm.setConstant(T(1) / (T(2) * c));
        Wc = Wm;

        Wm(0) = lambda() / c;
        Wc(0) = lambda() / c + (T(1) - params_.alpha * params_.alpha + params_.beta);
    }

    bool sigma_points(MatXsig& X) const {
        const Eigen::LLT<MatXX> llt(P());
        if (llt.info() != Eigen::Success)
            return false;
        const MatXX U = MatXX(llt.matrixU()) * std::sqrt(T(NX) + lambda());

        X.col(0) = x();
        for (int i = 0; i < NX; ++i) {
            X.col(1 + i) = x() + U.row(i).transpose();
            X.col(1 + NX + i) = x() - U.row(i).transpose();
        }
        return true;
    }
};

} // namespace filters
