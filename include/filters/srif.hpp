#pragma once

#include "filters/detail/storage.hpp"
#include "filters/results.hpp"

#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <cmath>
#include <optional>
#include <type_traits>

namespace filters {

/// Square-root information filter. Stores the information form R x = b with R upper triangular
/// (P = R^-1 R^-T) and keeps it triangular with Householder QR, which stays well conditioned where
/// the covariance form loses precision. The measurement size NZ is per call.
///
/// Trivially copyable. Steps whose noise covariance is not positive definite report failure and
/// leave the state unchanged.
template <typename T, int NX> struct SquareRootInformationFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0, "NX must be > 0");

    template <int N> using Vec = Eigen::Matrix<T, N, 1>;
    template <int R, int C> using Mat = Eigen::Matrix<T, R, C>;
    using VecX = Vec<NX>;
    using MatXX = Mat<NX, NX>;

    /// x = 0, P = I
    SquareRootInformationFilter() = default;
    /// @pre P_0 positive definite (asserted); use reset() to check instead
    SquareRootInformationFilter(const VecX& x_0, const MatXX& P_0) {
        [[maybe_unused]] const bool ok = reset(x_0, P_0);
        assert(ok && "P_0 must be positive definite");
    }

    /// @return false (state unchanged) if P_0 is not positive definite
    [[nodiscard]] bool reset(const VecX& x_0, const MatXX& P_0) {
        const Eigen::LLT<MatXX> llt_P(P_0);
        if (llt_P.info() != Eigen::Success)
            return false;
        const Eigen::LLT<MatXX> llt_info(llt_P.solve(MatXX::Identity()));
        if (llt_info.info() != Eigen::Success)
            return false;
        R_mut() = llt_info.matrixU();
        b_mut() = sqrt_information() * x_0;
        return true;
    }

    VecX state() const {
        return sqrt_information().template triangularView<Eigen::Upper>().solve(
            information_vector());
    }

    MatXX covariance() const {
        const MatXX R_inv =
            sqrt_information().template triangularView<Eigen::Upper>().solve(MatXX::Identity());
        return R_inv * R_inv.transpose();
    }

    /// Square-root information matrix R (upper triangular), R^T R = P^-1.
    Eigen::Map<const MatXX> sqrt_information() const {
        return Eigen::Map<const MatXX>(R_.data());
    }
    /// Information vector b, R x = b.
    Eigen::Map<const VecX> information_vector() const {
        return Eigen::Map<const VecX>(b_.data());
    }

    /// Time update without process noise.
    void predict(const MatXX& Phi) {
        retriangularise(through(Phi), information_vector());
    }

    /// Time update with process noise w ~ N(0, Q_w) entering as x <- Phi x + Gamma w.
    /// @return false (state unchanged) if Q_w is not positive definite
    template <int NW>
    [[nodiscard]] bool predict(const MatXX& Phi, const Mat<NX, NW>& Gamma,
                               const Mat<NW, NW>& Q_w) {
        return predict_noisy(through(Phi), Gamma, Q_w);
    }

    /// predict(), taking Phi^-1 instead of Phi to skip an LU factorisation. Worth it when the
    /// dynamics give the inverse cheaply (backward propagation, symplectic structure).
    void predict_from_inverse(const MatXX& Phi_inv) {
        retriangularise(sqrt_information() * Phi_inv, information_vector());
    }

    template <int NW>
    [[nodiscard]] bool predict_from_inverse(const MatXX& Phi_inv, const Mat<NX, NW>& Gamma,
                                            const Mat<NW, NW>& Q_w) {
        return predict_noisy(MatXX(sqrt_information() * Phi_inv), Gamma, Q_w);
    }

    /// Innovation of z = H x + v, v ~ N(0, R), without applying it. Forms S, which the update
    /// itself never needs.
    template <int NZ>
    std::optional<Innovation<T, NZ>> innovation(const Vec<NZ>& z, const Mat<NZ, NX>& H,
                                                const Mat<NZ, NZ>& R) const {
        // H P H^T = G^T G with G = R_info^-T H^T
        const Mat<NX, NZ> G = sqrt_information()
                                  .transpose()
                                  .template triangularView<Eigen::Lower>()
                                  .solve(H.transpose());
        const Mat<NZ, NZ> S = G.transpose() * G + R;
        const Eigen::LLT<Mat<NZ, NZ>> llt(S);
        if (llt.info() != Eigen::Success)
            return std::nullopt;
        const Vec<NZ> nu = z - H * state();
        return Innovation<T, NZ>{detail::to_array(nu), detail::to_array(S),
                                 llt.matrixL().solve(nu).squaredNorm()};
    }

    /// Measurement update for z = H x + v, v ~ N(0, R).
    /// @param z Measurement (for an error-state filter, the prefit residual z - h(x_ref))
    /// @return std::nullopt (state unchanged) if R is not positive definite
    template <int NZ>
    std::optional<SrifUpdate<T, NZ>> update(const Vec<NZ>& z, const Mat<NZ, NX>& H,
                                            const Mat<NZ, NZ>& R) {
        // Whiten so the measurement noise is unit: V^-1 z = V^-1 H x + V^-1 v, with R = V V^T.
        const Eigen::LLT<Mat<NZ, NZ>> llt(R);
        if (llt.info() != Eigen::Success)
            return std::nullopt;
        const auto V = llt.matrixL();
        const Vec<NZ> nu = z - H * state();

        // [R_info | b] is already triangular, so only the whitened measurement rows need
        // eliminating. What is left in their right-hand side is the whitened residual, whose
        // squared norm is the NIS.
        Mat<NX, NX + 1> top;
        top << sqrt_information(), information_vector();
        Mat<NZ, NX + 1> bottom;
        bottom << V.solve(H), V.solve(z);
        eliminate<NX>(top, bottom);
        R_mut() = top.template leftCols<NX>();
        b_mut() = top.col(NX);
        return SrifUpdate<T, NZ>{detail::to_array(nu), bottom.col(NX).squaredNorm()};
    }

  private:
    std::array<T, NX * NX> R_ = detail::identity<T, NX>();
    std::array<T, NX> b_{};

    Eigen::Map<MatXX> R_mut() {
        return Eigen::Map<MatXX>(R_.data());
    }
    Eigen::Map<VecX> b_mut() {
        return Eigen::Map<VecX>(b_.data());
    }

    /// R Phi^-1, the information matrix mapped through the transition.
    MatXX through(const MatXX& Phi) const {
        return Phi.transpose().partialPivLu().solve(sqrt_information().transpose()).transpose();
    }

    template <int NW>
    bool predict_noisy(const MatXX& R_tilde, const Mat<NX, NW>& Gamma, const Mat<NW, NW>& Q_w) {
        using MatWW = Mat<NW, NW>;
        constexpr int N = NW + NX;

        const Eigen::LLT<MatWW> llt(Q_w);
        if (llt.info() != Eigen::Success)
            return false;

        // System over [w; x]: R_w w = 0 (unit noise) and -R~ Gamma w + R~ x = b, R~ = R Phi^-1.
        // R_w = L^-1 for Q_w = L L^T is lower triangular; reversing the order of w (and the rows)
        // makes it upper triangular, so the w columns can be eliminated with structured QR.
        Mat<NW, N + 1> top = Mat<NW, N + 1>::Zero();
        top.template leftCols<NW>() = MatWW(llt.matrixL().solve(MatWW::Identity())).reverse();
        Mat<NX, N + 1> bottom;
        bottom << -(R_tilde * Gamma).rowwise().reverse(), R_tilde, information_vector();
        eliminate<NW>(top, bottom);

        // The w-free rows that remain are a dense NX x NX system in x.
        retriangularise(bottom.template middleCols<NX>(NW), bottom.col(N));
        return true;
    }

    /// Householder-zero the first K columns of `bottom` into the upper-triangular leading K x K
    /// block of `top`. The last column of both is the right-hand side. Each reflector only touches
    /// one row of `top`, so this costs O(K * rows(bottom) * cols) instead of a full QR.
    template <int K, int C, int M>
    static void eliminate(Mat<K, C>& top, Mat<M, C>& bottom) {
        for (int j = 0; j < K; ++j) {
            const T sigma = bottom.col(j).squaredNorm();
            if (sigma == T(0))
                continue; // column already eliminated
            const T a = top(j, j);
            const T mu = std::sqrt(a * a + sigma);
            // v = [a - mu; bottom.col(j)], with a - mu formed without cancellation (Parlett), so
            // the new diagonal is +mu
            const T v0 = a <= T(0) ? a - mu : -sigma / (a + mu);
            const T tau = T(2) / (v0 * v0 + sigma);
            const Vec<M> v = bottom.col(j);
            const Mat<1, C> s = v0 * top.row(j) + v.transpose() * bottom;
            top.row(j) -= (tau * v0) * s;
            bottom -= (tau * v) * s;
            top(j, j) = mu;
            bottom.col(j).setZero();
        }
    }

    /// QR-triangularise the dense system A x = rhs into R x = b.
    template <class DA, class DB>
    void retriangularise(const Eigen::MatrixBase<DA>& A, const Eigen::MatrixBase<DB>& rhs) {
        const Eigen::HouseholderQR<MatXX> qr(A);
        R_mut() = qr.matrixQR().template triangularView<Eigen::Upper>();
        b_mut() = qr.householderQ().transpose() * VecX(rhs);
    }
};

} // namespace filters
