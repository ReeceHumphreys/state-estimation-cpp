#pragma once

#include <Eigen/Dense>
#include <type_traits>

namespace filters {

template <typename T, int NX, int NZ> struct LinearKalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0 && NZ > 0, "Dimensions must be > 0");

  public:
    using VecX = Eigen::Matrix<T, NX, 1>;
    using VecZ = Eigen::Matrix<T, NZ, 1>;
    using MatXX = Eigen::Matrix<T, NX, NX>;
    using MatZZ = Eigen::Matrix<T, NZ, NZ>;
    using MatZX = Eigen::Matrix<T, NZ, NX>;
    using MatXZ = Eigen::Matrix<T, NX, NZ>;

    VecX x = VecX::Zero();       // State estimate
    MatXX P = MatXX::Identity(); // Estimate covariance

    /// Constructor
    LinearKalmanFilter() = default;
    LinearKalmanFilter(const VecX& x_0, const MatXX& P_0) : x(x_0), P(P_0) {}

    /// Perform a single Kalman filter prediction
    /// @param state_transition State transition model
    /// @param Q Process noise covariance
    void predict(const MatXX& state_transition, const MatXX& Q) {
        x = state_transition * x;
        P = state_transition * P * state_transition.transpose() + Q;
    }

    /// Perform a single Kalman filter update
    /// @param z Measurement
    /// @param H Observation model
    /// @param R Measurement noise covariance
    void update(const VecZ& z, const MatZX& H, const MatZZ& R) {
        const MatXZ K = kalman_gain(P, H, R);
        x = current_state_estimate(x, z, H, K);
        P = update_covariance(K, H, P, R);
    }

  private:
    /// Kalman gain equation
    /// K = P H / (H P H + R)
    MatXZ kalman_gain(const MatXX& P_prior, const MatZX& H, const MatZZ& R) {
        const MatZZ S = H * P_prior * H.transpose() + R;
        const Eigen::LDLT<MatZZ> ldlt(S);
        return (P_prior * H.transpose()) * ldlt.solve(MatZZ::Identity());
    }

    /// State update equation
    /// x = x_prior + K (z - H x_prior)
    VecX current_state_estimate(const VecX& x_prior, const VecZ& z, const MatZX& H,
                                const MatXZ& K) {
        return x_prior + K * innovation(z, H, x_prior);
    }

    /// Innovation equation
    VecZ innovation(const VecZ& z, const MatZX& H, const VecX& x_prior) {
        return z - H * x_prior;
    }

    /// Covariance update equation (Joseph form)
    /// P = (I - K H) P (I - K H) + K R K
    MatXX update_covariance(const MatXZ& K, const MatZX& H, const MatXX& P_prior, const MatZZ& R) {
        const MatXX I = MatXX::Identity();
        const MatXX I_KH = I - K * H;
        return I_KH * P_prior * I_KH.transpose() + K * R * K.transpose();
    }
};

template <typename T, int NX, int NZ> struct ExtendedKalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0 && NZ > 0, "Dimensions must be > 0");

  public:
    using VecX = Eigen::Matrix<T, NX, 1>;
    using VecZ = Eigen::Matrix<T, NZ, 1>;
    using MatXX = Eigen::Matrix<T, NX, NX>;
    using MatZZ = Eigen::Matrix<T, NZ, NZ>;
    using MatZX = Eigen::Matrix<T, NZ, NX>;
    using MatXZ = Eigen::Matrix<T, NX, NZ>;

    VecX x = VecX::Zero();       // State estimate
    MatXX P = MatXX::Identity(); // Estimate covariance

    ExtendedKalmanFilter() = default;
    ExtendedKalmanFilter(const VecX& x_0, const MatXX& P_0) : x(x_0), P(P_0) {}

    /// EKF predict:
    /// x_k|k-1 = f(x_k-1|k-1)
    /// P_k|k-1 = F P F^T + Q, where F = df/dx at linearization point
    template <class FFunc, class FJac> void predict(FFunc&& f, FJac&& jacobian_f, const MatXX& Q) {
        const VecX x_prev = x;

        x = f(x_prev);

        const MatXX F = jacobian_f(x_prev); // linearize at x_prev (common choice)
        P = F * P * F.transpose() + Q;
    }

    /// EKF update:
    /// z_hat = h(x_k|k-1)
    /// H = dh/dx at linearization point
    template <class HFunc, class HJac>
    void update(const VecZ& z, HFunc&& h, HJac&& jacobian_h, const MatZZ& R) {
        const VecX x_prior = x;
        const VecZ z_hat = h(x_prior);
        const MatZX H = jacobian_h(x_prior);

        const MatXZ K = kalman_gain(P, H, R);
        x = current_state_estimate(x_prior, z, z_hat, K);
        P = update_covariance(K, H, P, R);
    }

  private:
    /// Kalman gain:
    /// K = P H^T (H P H^T + R)^-1
    MatXZ kalman_gain(const MatXX& P_prior, const MatZX& H, const MatZZ& R) const {
        const MatZZ S = H * P_prior * H.transpose() + R;
        const Eigen::LDLT<MatZZ> ldlt(S);
        return (P_prior * H.transpose()) * ldlt.solve(MatZZ::Identity());
    }

    /// EKF state update:
    /// x = x_prior + K (z - z_hat)
    VecX current_state_estimate(const VecX& x_prior, const VecZ& z, const VecZ& z_hat,
                                const MatXZ& K) const {
        return x_prior + K * innovation(z, z_hat);
    }

    /// EKF innovation:
    /// y = z - z_hat
    VecZ innovation(const VecZ& z, const VecZ& z_hat) const {
        return z - z_hat;
    }

    /// Joseph covariance update (same as your LKF):
    /// P = (I - K H) P (I - K H)^T + K R K^T
    MatXX update_covariance(const MatXZ& K, const MatZX& H, const MatXX& P_prior,
                            const MatZZ& R) const {
        const MatXX I = MatXX::Identity();
        const MatXX I_KH = I - K * H;
        return I_KH * P_prior * I_KH.transpose() + K * R * K.transpose();
    }
};

template <typename T, int NX, int NZ> struct UnscentedKalmanFilter {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(NX > 0 && NZ > 0, "Dimensions must be > 0");

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

    VecX x = VecX::Zero();
    MatXX P = MatXX::Identity();
    bool has_prediction = false;

    // UKF tuning
    T alpha = T(1e-3);
    T beta = T(2); // optimal for Gaussian
    T kappa = T(0);

    template <class FFunc> void predict(FFunc&& f, const MatXX& Q) {
        MatXsig X = sigma_points(x, P);
        MatXsig Xp;

        for (int i = 0; i < kSigmaCount; ++i) {
            Xp.col(i) = f(X.col(i));
        }

        VecSig Wm, Wc;
        weights(Wm, Wc);

        x = Xp * Wm; // weighted mean
        P = MatXX::Zero();
        for (int i = 0; i < kSigmaCount; ++i) {
            const VecX dx = Xp.col(i) - x;
            P += Wc(i) * (dx * dx.transpose());
        }
        P += Q;

        // Save propagated sigma points for update step (optional but common)
        X_pred = Xp;

        has_prediction = true;
    }

    template <class HFunc> void update(const VecZ& z, HFunc&& h, const MatZZ& R) {
        assert(has_prediction && "EKF update() called before predict()");
        VecSig Wm, Wc;
        weights(Wm, Wc);

        MatZsig Z;
        for (int i = 0; i < kSigmaCount; ++i) {
            Z.col(i) = h(X_pred.col(i));
        }

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

        x = x + K * (z - z_hat);
        P = P - K * S * K.transpose();
    }

  private:
    // store propagated sigma points from predict()
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
        MatXX L = llt.matrixL();
        L *= std::sqrt(c);

        MatXsig X;
        X.col(0) = x0;
        for (int i = 0; i < NX; ++i) {
            X.col(1 + i) = x0 + L.col(i);
            X.col(1 + NX + i) = x0 - L.col(i);
        }
        return X;
    }
};

} // namespace filters
