#pragma once

#include "data_structures/ring_buffer.hpp"

#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace filters {

template <typename T, int NX, int NZ, std::uint32_t NHIST = 0> struct LinearKalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0 && NZ > 0, "Dimensions must be > 0");

  public:
    using VecX = Eigen::Matrix<T, NX, 1>;
    using VecZ = Eigen::Matrix<T, NZ, 1>;
    using MatXX = Eigen::Matrix<T, NX, NX>;
    using MatZZ = Eigen::Matrix<T, NZ, NZ>;
    using MatZX = Eigen::Matrix<T, NZ, NX>;
    using MatXZ = Eigen::Matrix<T, NX, NZ>;

    static constexpr bool kHasHistory = (NHIST > 0);
    struct HistoryRow {
        T t = std::numeric_limits<T>::quiet_NaN();
        VecX dx_post = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX P_post = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ y_prefit = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ nu_innov = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ postfit = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        int station_id = -1; // Ground station ID for color-coding plots

        VecX x_true = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        VecX x_hat = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
    };

    using HistoryBuffer = std::vector<HistoryRow>;

    const HistoryBuffer& history() const {
        return hist_;
    }
    HistoryBuffer& history_mut() {
        return hist_;
    }
    void reserve_history(std::size_t n) {
        hist_.reserve(n);
    }
    void push_history(const HistoryRow& row) {
        if constexpr (NHIST > 0) {
            hist_.push_back(row);
        }
    }

    struct PredictStep {
        VecX x_prior = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX P_prior = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX Phi = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX Q = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
    };

    struct UpdateStep {
        VecZ z = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ y =
            VecZ::Constant(std::numeric_limits<T>::quiet_NaN()); // prefit residual (z - H x_prior)
        VecZ nu =
            VecZ::Constant(std::numeric_limits<T>::quiet_NaN()); // innovation actually applied
        MatZX H = MatZX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatZZ R = MatZZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatZZ S = MatZZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXZ K = MatXZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ postfit_lin =
            VecZ::Constant(std::numeric_limits<T>::quiet_NaN()); // (z - H x_post) linear
    };

    // ------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------
    VecX x = VecX::Zero();       // State estimate
    MatXX P = MatXX::Identity(); // Estimate covariance

    LinearKalmanFilter() = default;
    LinearKalmanFilter(const VecX& x_0, const MatXX& P_0) : x(x_0), P(P_0) {}

    void reset(const VecX& x_0, const MatXX& P_0) {
        x = x_0;
        P = P_0;
        if constexpr (NHIST > 0) {
            pred_hist.reset();
            upd_hist.reset();
            hist_.clear();
        }
    }

    // internal step history access
    std::uint32_t predict_history_size() const {
        if constexpr (NHIST > 0)
            return pred_hist.size;
        return 0;
    }
    std::uint32_t update_history_size() const {
        if constexpr (NHIST > 0)
            return upd_hist.size;
        return 0;
    }

    const PredictStep& predict_history_oldest(std::uint32_t i) const {
        static_assert(NHIST > 0, "History disabled (NHIST=0)");
        return pred_hist.at_oldest(i);
    }
    const UpdateStep& update_history_oldest(std::uint32_t i) const {
        static_assert(NHIST > 0, "History disabled (NHIST=0)");
        return upd_hist.at_oldest(i);
    }

    /// Perform a single Kalman filter prediction
    /// @param state_transition State transition model (Phi)
    /// @param Q Process noise covariance
    PredictStep predict(const MatXX& state_transition, const MatXX& Q) {
        PredictStep step;
        step.x_prior = x;
        step.P_prior = P;
        step.Phi = state_transition;
        step.Q = Q;

        x = state_transition * x;
        P = state_transition * P * state_transition.transpose() + Q;

        if constexpr (NHIST > 0)
            pred_hist.push(step);
        return step;
    }

    /// Perform a single Kalman filter update
    /// @param z Measurement
    /// @param H Observation model
    /// @param R Measurement noise covariance
    UpdateStep update(const VecZ& z, const MatZX& H, const MatZZ& R) {
        const VecZ y = z - H * x; // prefit residual
        return update_innovation(y, H, R, z, y); // linear case: innovation is the prefit residual
    }

    /// Update with explicit innovation (for LKF/error-state filters)
    /// @param nu Innovation: nu = y - H*dx where y = z - h(X_ref)
    /// @param H Observation matrix (Jacobian)
    /// @param R Measurement noise covariance
    /// @param z Measurement
    /// @param y_prefit Prefit residual: y = z - h(X_ref)
    UpdateStep update_innovation(const VecZ& nu, const MatZX& H, const MatZZ& R, const VecZ& z,
                                 const VecZ& y_prefit) {
        const MatZZ S = H * P * H.transpose() + R;
        const Eigen::LDLT<MatZZ> ldlt(S);
        const MatXZ K = (P * H.transpose()) * ldlt.solve(MatZZ::Identity());

        const VecX x_prior = x;
        const MatXX P_prior = P;

        x = x_prior + K * nu;
        P = update_covariance(K, H, P_prior, R);

        UpdateStep step;
        step.z = z;
        step.y = y_prefit;
        step.nu = nu;
        step.H = H;
        step.R = R;
        step.S = S;
        step.K = K;
        step.postfit_lin = y_prefit - H * x;

        if constexpr (NHIST > 0)
            upd_hist.push(step);
        return step;
    }

  private:
    [[no_unique_address]] HistoryBuffer hist_{};

    static MatXX update_covariance(const MatXZ& K, const MatZX& H, const MatXX& P_prior,
                                   const MatZZ& R) {
        const MatXX I = MatXX::Identity();
        const MatXX I_KH = I - K * H;
        return I_KH * P_prior * I_KH.transpose() + K * R * K.transpose();
    }

    [[no_unique_address]] RingBuffer<PredictStep, (NHIST > 0 ? NHIST : 1)> pred_hist{};
    [[no_unique_address]] RingBuffer<UpdateStep, (NHIST > 0 ? NHIST : 1)> upd_hist{};
};

// ============================================================================
// Extended Kalman Filter (EKF) + optional fixed-size history
// - History exposes the same HistoryRow stream for HW-style logging.
// ============================================================================

template <typename T, int NX, int NZ, std::uint32_t NHIST = 0> struct ExtendedKalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0 && NZ > 0, "Dimensions must be > 0");

  public:
    using VecX = Eigen::Matrix<T, NX, 1>;
    using VecZ = Eigen::Matrix<T, NZ, 1>;
    using MatXX = Eigen::Matrix<T, NX, NX>;
    using MatZZ = Eigen::Matrix<T, NZ, NZ>;
    using MatZX = Eigen::Matrix<T, NZ, NX>;
    using MatXZ = Eigen::Matrix<T, NX, NZ>;

    static constexpr bool kHasHistory = (NHIST > 0);

    struct HistoryRow {
        T t = std::numeric_limits<T>::quiet_NaN();
        VecX dx_post = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX P_post = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ y_prefit = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ nu_innov = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ postfit = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        int station_id = -1; // Ground station ID for color-coding plots

        VecX x_true = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        VecX x_hat = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
    };

    using HistoryBuffer = std::vector<HistoryRow>;

    const HistoryBuffer& history() const {
        return hist_;
    }
    HistoryBuffer& history_mut() {
        return hist_;
    }
    void reserve_history(std::size_t n) {
        hist_.reserve(n);
    }
    void push_history(const HistoryRow& row) {
        if constexpr (NHIST > 0) {
            hist_.push_back(row);
        }
    }

    // ------------------------------------------------------------------------
    // Internal step logs (optional)
    // ------------------------------------------------------------------------
    struct PredictStep {
        VecX x_prior = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX P_prior = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
        VecX x_pred = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX Phi = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX Q = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
    };

    struct UpdateStep {
        VecZ z = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ z_hat = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ nu = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatZX H = MatZX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatZZ R = MatZZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatZZ S = MatZZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXZ K = MatXZ::Constant(std::numeric_limits<T>::quiet_NaN());
    };

    // ------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------
    VecX x = VecX::Zero();
    MatXX P = MatXX::Identity();

    ExtendedKalmanFilter() = default;
    ExtendedKalmanFilter(const VecX& x_0, const MatXX& P_0) : x(x_0), P(P_0) {}

    void reset(const VecX& x_0, const MatXX& P_0) {
        x = x_0;
        P = P_0;
        if constexpr (NHIST > 0) {
            pred_hist.reset();
            upd_hist.reset();
            hist_.clear();
        }
    }

    std::uint32_t predict_history_size() const {
        if constexpr (NHIST > 0)
            return pred_hist.size;
        return 0;
    }
    std::uint32_t update_history_size() const {
        if constexpr (NHIST > 0)
            return upd_hist.size;
        return 0;
    }

    const PredictStep& predict_history_oldest(std::uint32_t i) const {
        static_assert(NHIST > 0, "History disabled (NHIST=0)");
        return pred_hist.at_oldest(i);
    }
    const UpdateStep& update_history_oldest(std::uint32_t i) const {
        static_assert(NHIST > 0, "History disabled (NHIST=0)");
        return upd_hist.at_oldest(i);
    }

    /// Discrete EKF prediction using externally computed transition
    PredictStep predict(const VecX& x_pred, const Eigen::Ref<const MatXX>& Phi, const MatXX& Q) {
        PredictStep step;
        step.x_prior = x;
        step.P_prior = P;
        step.x_pred = x_pred;
        step.Phi = Phi;
        step.Q = Q;

        x = x_pred;
        P = Phi * P * Phi.transpose() + Q;

        if constexpr (NHIST > 0)
            pred_hist.push(step);
        return step;
    }

    /// EKF prediction for linear error-state propagation x <- Phi x
    PredictStep predict(const Eigen::Ref<const MatXX>& Phi, const MatXX& Q) {
        const VecX x_pred = Phi * x;
        return predict(x_pred, Phi, Q);
    }

    /// Discrete EKF update using externally computed z_hat and H
    UpdateStep update(const VecZ& z, const VecZ& z_hat, const Eigen::Ref<const MatZX>& H,
                      const MatZZ& R) {
        const MatZZ S = H * P * H.transpose() + R;
        const Eigen::LDLT<MatZZ> ldlt(S);
        const MatXZ K = (P * H.transpose()) * ldlt.solve(MatZZ::Identity());

        const VecX x_prior = x;
        const MatXX P_prior = P;

        const VecZ nu = z - z_hat;

        x = x_prior + K * nu;
        P = update_covariance(K, H, P_prior, R);

        UpdateStep step;
        step.z = z;
        step.z_hat = z_hat;
        step.nu = nu;
        step.H = H;
        step.R = R;
        step.S = S;
        step.K = K;

        if constexpr (NHIST > 0)
            upd_hist.push(step);
        return step;
    }

    /// EKF update with explicit innovation (for error-state EKF)
    /// @param nu Innovation: nu = y - H*dx where y = z - h(X_ref)
    /// @param H Observation matrix (Jacobian)
    /// @param R Measurement noise covariance
    /// @param z Measurement
    /// @param z_hat_ref Predicted measurement from reference: h(X_ref)
    UpdateStep update_innovation(const VecZ& nu, const Eigen::Ref<const MatZX>& H, const MatZZ& R,
                                 const VecZ& z, const VecZ& z_hat_ref) {
        const MatZZ S = H * P * H.transpose() + R;
        const Eigen::LDLT<MatZZ> ldlt(S);
        const MatXZ K = (P * H.transpose()) * ldlt.solve(MatZZ::Identity());

        const VecX x_prior = x;
        const MatXX P_prior = P;

        x = x_prior + K * nu;
        P = update_covariance(K, H, P_prior, R);

        UpdateStep step;
        step.z = z;
        step.z_hat = z_hat_ref;
        step.nu = nu;
        step.H = H;
        step.R = R;
        step.S = S;
        step.K = K;

        if constexpr (NHIST > 0)
            upd_hist.push(step);
        return step;
    }

  private:
    [[no_unique_address]] HistoryBuffer hist_{};

    static MatXX update_covariance(const MatXZ& K, const MatZX& H, const MatXX& P_prior,
                                   const MatZZ& R) {
        const MatXX I = MatXX::Identity();
        const MatXX I_KH = I - K * H;
        return I_KH * P_prior * I_KH.transpose() + K * R * K.transpose();
    }

    [[no_unique_address]] RingBuffer<PredictStep, (NHIST > 0 ? NHIST : 1)> pred_hist{};
    [[no_unique_address]] RingBuffer<UpdateStep, (NHIST > 0 ? NHIST : 1)> upd_hist{};
};

template <typename T, int NX, int NZ, std::uint32_t NHIST = 0> struct UnscentedKalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0 && NZ > 0, "Dimensions must be > 0");

  public:
    using VecX = Eigen::Matrix<T, NX, 1>;
    using VecZ = Eigen::Matrix<T, NZ, 1>;
    using MatXX = Eigen::Matrix<T, NX, NX>;
    using MatZZ = Eigen::Matrix<T, NZ, NZ>;
    using MatXZ = Eigen::Matrix<T, NX, NZ>;
    using MatZX = Eigen::Matrix<T, NZ, NX>;

    static constexpr int kSigmaCount = 2 * NX + 1;

    using MatXsig = Eigen::Matrix<T, NX, kSigmaCount>;
    using MatZsig = Eigen::Matrix<T, NZ, kSigmaCount>;
    using VecSig = Eigen::Matrix<T, kSigmaCount, 1>;

    struct PredictStep {
        VecX x_prior = VecX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX P_prior = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXX Q = MatXX::Constant(std::numeric_limits<T>::quiet_NaN());
    };

    struct UpdateStep {
        VecZ z = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ z_hat = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatZZ R = MatZZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatZZ S = MatZZ::Constant(std::numeric_limits<T>::quiet_NaN());
        MatXZ K = MatXZ::Constant(std::numeric_limits<T>::quiet_NaN());
        VecZ nu = VecZ::Constant(std::numeric_limits<T>::quiet_NaN());
    };

    static constexpr bool kHasHistory = (NHIST > 0);

    VecX x = VecX::Zero();
    MatXX P = MatXX::Identity();

    bool has_prediction = false;

    T alpha = T(1e-3);
    T beta = T(2);
    T kappa = T(0);

    void reset(const VecX& x0, const MatXX& P0) {
        x = x0;
        P = P0;
        has_prediction = false;
        X_pred.setZero();
        if constexpr (NHIST > 0) {
            pred_hist.reset();
            upd_hist.reset();
        }
    }

    std::uint32_t predict_history_size() const {
        if constexpr (NHIST > 0)
            return pred_hist.size;
        return 0;
    }
    std::uint32_t update_history_size() const {
        if constexpr (NHIST > 0)
            return upd_hist.size;
        return 0;
    }

    const PredictStep& predict_history_oldest(std::uint32_t i) const {
        static_assert(NHIST > 0, "History disabled (NHIST=0)");
        return pred_hist.at_oldest(i);
    }
    const UpdateStep& update_history_oldest(std::uint32_t i) const {
        static_assert(NHIST > 0, "History disabled (NHIST=0)");
        return upd_hist.at_oldest(i);
    }

    template <class FFunc> PredictStep predict(FFunc&& f, const MatXX& Q) {
        PredictStep step;
        step.x_prior = x;
        step.P_prior = P;
        step.Q = Q;

        const MatXsig X = sigma_points(x, P);

        MatXsig Xp;
        for (int i = 0; i < kSigmaCount; ++i)
            Xp.col(i) = f(X.col(i));

        VecSig Wm, Wc;
        weights(Wm, Wc);

        x = Xp * Wm;

        P = MatXX::Zero();
        for (int i = 0; i < kSigmaCount; ++i) {
            const VecX dx = Xp.col(i) - x;
            P += Wc(i) * (dx * dx.transpose());
        }
        P += Q;

        X_pred = Xp;
        has_prediction = true;

        if constexpr (NHIST > 0)
            pred_hist.push(step);
        return step;
    }

    template <class HFunc> UpdateStep update(const VecZ& z, HFunc&& h, const MatZZ& R) {
        assert(has_prediction && "UKF update() called before predict()");

        VecSig Wm, Wc;
        weights(Wm, Wc);

        MatZsig Z;
        for (int i = 0; i < kSigmaCount; ++i)
            Z.col(i) = h(X_pred.col(i));

        const VecZ z_hat = Z * Wm;

        MatZZ S = MatZZ::Zero();
        MatXZ Pxz = MatXZ::Zero();
        for (int i = 0; i < kSigmaCount; ++i) {
            const VecZ dz = Z.col(i) - z_hat;
            const VecX dx = X_pred.col(i) - x;
            S += Wc(i) * (dz * dz.transpose());
            Pxz += Wc(i) * (dx * dz.transpose());
        }
        S += R;

        const Eigen::LDLT<MatZZ> ldlt(S);
        const MatXZ K = Pxz * ldlt.solve(MatZZ::Identity());

        const VecZ nu = z - z_hat;

        x = x + K * nu;
        P = P - K * S * K.transpose();

        UpdateStep step;
        step.z = z;
        step.z_hat = z_hat;
        step.R = R;
        step.S = S;
        step.K = K;
        step.nu = nu;

        if constexpr (NHIST > 0)
            upd_hist.push(step);
        return step;
    }

  private:
    MatXsig X_pred = MatXsig::Zero();

    void weights(VecSig& Wm, VecSig& Wc) const {
        const T lambda = alpha * alpha * (T(NX) + kappa) - T(NX);
        const T c = T(NX) + lambda;

        Wm.setConstant(T(1) / (T(2) * c));
        Wc = Wm;

        Wm(0) = lambda / c;
        Wc(0) = lambda / c + (T(1) - alpha * alpha + beta);
    }

    MatXsig sigma_points(const VecX& x0, const MatXX& P0) const {
        const T lambda = alpha * alpha * (T(NX) + kappa) - T(NX);
        const T c = T(NX) + lambda;

        Eigen::LLT<MatXX> llt(P0);
        MatXX U = llt.matrixU();
        U *= std::sqrt(c);

        MatXsig X;
        X.col(0) = x0;
        for (int i = 0; i < NX; ++i) {
            X.col(1 + i) = x0 + U.row(i).transpose();
            X.col(1 + NX + i) = x0 - U.row(i).transpose();
        }
        return X;
    }

    [[no_unique_address]] RingBuffer<PredictStep, (NHIST > 0 ? NHIST : 1)> pred_hist{};
    [[no_unique_address]] RingBuffer<UpdateStep, (NHIST > 0 ? NHIST : 1)> upd_hist{};
};

} // namespace filters
