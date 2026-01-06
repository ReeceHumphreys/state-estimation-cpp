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

    /// Constructor
    ExtendedKalmanFilter() = default;
    ExtendedKalmanFilter(const VecX& x_0, const MatXX& P_0) : x(x_0), P(P_0) {}

    /// Perform a single EKF prediction
    ///   x_k|k-1 = f(x_{k-1|k-1})
    /// @tparam FFunc Callable type implementing f(x) -> VecX
    /// @tparam FJac  Callable type implementing jacobian_f(x) -> MatXX
    /// @param f Nonlinear state transition function
    /// @param jacobian_f Jacobian of f with respect to x
    /// @param Q Process noise covariance (state-space), NX x NX
    template <class FFunc, class FJac> void predict(FFunc&& f, FJac&& jacobian_f, const MatXX& Q) {
        // Save the prior state used for linearization and propagation
        const VecX x_prev = x;

        // Nonlinear state propagation
        x = f(x_prev);

        // Linearize about x_prev
        const MatXX F = jacobian_f(x_prev);

        // Covariance propagation using the local linear model
        P = F * P * F.transpose() + Q;
    }

    /// Perform a single EKF measurement update
    /// @tparam HFunc Callable type implementing h(x) -> VecZ
    /// @tparam HJac  Callable type implementing jacobian_h(x) -> MatZX
    /// @param z Measurement vector, NZ x 1
    /// @param h Nonlinear measurement function
    /// @param jacobian_h Jacobian of h with respect to x
    /// @param R Measurement noise covariance, NZ x NZ
    template <class HFunc, class HJac>
    void update(const VecZ& z, HFunc&& h, HJac&& jacobian_h, const MatZZ& R) {
        // Capture the prior state used to compute z_hat and H
        const VecX x_prior = x;

        // Predicted measurement and linearization
        const VecZ z_hat = h(x_prior);
        const MatZX H = jacobian_h(x_prior);

        // Standard Kalman update using residual y = z - z_hat
        const MatXZ K = kalman_gain(P, H, R);
        x = current_state_estimate(x_prior, z, z_hat, K);
        P = update_covariance(K, H, P, R);
    }

  private:
    /// Kalman gain equation
    /// S = H P H^T + R
    /// K = P H^T S^{-1}
    /// @param P_prior Prior estimate covariance, NX x NX
    /// @param H Measurement Jacobian / observation matrix, NZ x NX
    /// @param R Measurement noise covariance, NZ x NZ
    /// @return Kalman gain, NX x NZ
    MatXZ kalman_gain(const MatXX& P_prior, const MatZX& H, const MatZZ& R) const {
        const MatZZ S = H * P_prior * H.transpose() + R;
        const Eigen::LDLT<MatZZ> ldlt(S);
        return (P_prior * H.transpose()) * ldlt.solve(MatZZ::Identity());
    }

    /// State update equation
    /// x_k|k = x_k|k-1 + K (z - z_hat)
    /// @param x_prior Prior state estimate (x_k|k-1), NX x 1
    /// @param z Measurement vector, NZ x 1
    /// @param z_hat Predicted measurement h(x_prior), NZ x 1
    /// @param K Kalman gain, NX x NZ
    /// @return Updated state estimate (x_k|k), NX x 1
    VecX current_state_estimate(const VecX& x_prior, const VecZ& z, const VecZ& z_hat,
                                const MatXZ& K) const {
        return x_prior + K * innovation(z, z_hat);
    }

    /// Innovation equation
    /// y = z - z_hat
    /// @param z Measurement vector, NZ x 1
    /// @param z_hat Predicted measurement h(x_prior), NZ x 1
    /// @return Innovation / residual, NZ x 1
    VecZ innovation(const VecZ& z, const VecZ& z_hat) const {
        return z - z_hat;
    }

    /// Covariance update equation (Joseph form)
    /// @param K Kalman gain, NX x NZ
    /// @param H Measurement Jacobian, NZ x NX
    /// @param P_prior Prior covariance, NX x NX
    /// @param R Measurement noise covariance, NZ x NZ
    /// @return Updated covariance, NX x NX
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

  public:
    using VecX = Eigen::Matrix<T, NX, 1>;
    using VecZ = Eigen::Matrix<T, NZ, 1>;
    using MatXX = Eigen::Matrix<T, NX, NX>;
    using MatZZ = Eigen::Matrix<T, NZ, NZ>;
    using MatXZ = Eigen::Matrix<T, NX, NZ>;
    using MatZX = Eigen::Matrix<T, NZ, NX>;

    /// Number of sigma points used by the standard UKF:
    /// 2*NX + 1
    static constexpr int kSigmaCount = 2 * NX + 1;

    /// Sigma point containers:
    /// Xsig: NX x (2NX+1), Zsig: NZ x (2NX+1)
    using MatXsig = Eigen::Matrix<T, NX, kSigmaCount>;
    using MatZsig = Eigen::Matrix<T, NZ, kSigmaCount>;

    /// Weight vector containers
    using VecSig = Eigen::Matrix<T, kSigmaCount, 1>;

    VecX x = VecX::Zero();       // State estimate
    MatXX P = MatXX::Identity(); // Estimate covariance

    /// If predict() has been called and X_pred contains valid sigma points
    bool has_prediction = false;

    /// UKF tuning parameters (Merwe scaled sigma points)
    /// These control the spread and weighting of sigma points:
    ///  - alpha: spread of sigma points around mean
    ///  - beta:  prior knowledge about distribution
    ///  - kappa: secondary scaling parameter
    T alpha = T(1e-3); // Small spread
    T beta = T(2);     // Gaussian prior
    T kappa = T(0);    // Secondary scaling

    /// Perform a single UKF prediction
    /// @tparam FFunc Callable type implementing f(x) -> VecX
    /// @param f Nonlinear state transition function
    /// @param Q Process noise covariance (state-space), NX x NX
    template <class FFunc> void predict(FFunc&& f, const MatXX& Q) {
        // Generate sigma points about (x,P)
        const MatXsig X = sigma_points(x, P);

        // Propagate sigma points through the nonlinear model
        MatXsig Xp;
        for (int i = 0; i < kSigmaCount; ++i) {
            Xp.col(i) = f(X.col(i));
        }

        // Compute UKF weights
        VecSig Wm, Wc;
        weights(Wm, Wc);

        // Predicted mean: x <- Sigma Wm_i * Xp_i
        x = Xp * Wm;

        // Predicted covariance: P <- Sigma Wc_i * (dx_i dx_i^T) + Q
        P = MatXX::Zero();
        for (int i = 0; i < kSigmaCount; ++i) {
            const VecX dx = Xp.col(i) - x;
            P += Wc(i) * (dx * dx.transpose());
        }
        P += Q;

        // Store propagated sigma points for the measurement update (common UKF pattern)
        X_pred = Xp;
        has_prediction = true;
    }

    /// Perform a single UKF measurement update
    /// @tparam HFunc Callable type implementing h(x) -> VecZ
    /// @param z Measurement vector, NZ x 1
    /// @param h Nonlinear measurement function
    /// @param R Measurement noise covariance, NZ x NZ
    template <class HFunc> void update(const VecZ& z, HFunc&& h, const MatZZ& R) {
        // UKF update requires the predicted sigma points (from predict()).
        // Use an assertion to catch incorrect call ordering early.
        assert(has_prediction && "UKF update() called before predict()");

        // Compute UKF weights
        VecSig Wm, Wc;
        weights(Wm, Wc);

        // Transform predicted sigma points through measurement model: Z_i = h(X'_i)
        MatZsig Z;
        for (int i = 0; i < kSigmaCount; ++i) {
            Z.col(i) = h(X_pred.col(i));
        }

        // Predicted measurement mean: z_hat <- Sigma Wm_i Z_i
        const VecZ z_hat = Z * Wm;

        // Innovation covariance S and cross covariance Pxz
        MatZZ S = MatZZ::Zero();
        MatXZ Pxz = MatXZ::Zero();
        for (int i = 0; i < kSigmaCount; ++i) {
            const VecZ dz = Z.col(i) - z_hat;  // measurement deviation
            const VecX dx = X_pred.col(i) - x; // state deviation (about predicted mean)
            S += Wc(i) * (dz * dz.transpose());
            Pxz += Wc(i) * (dx * dz.transpose());
        }
        S += R;

        // Kalman gain: K = Pxz S^{-1} (compute via LDLT solve)
        const Eigen::LDLT<MatZZ> ldlt(S);
        const MatXZ K = Pxz * ldlt.solve(MatZZ::Identity());

        // State and covariance update (standard UKF form)
        x = x + K * (z - z_hat);
        P = P - K * S * K.transpose();
    }

  private:
    /// Stored predicted sigma points X'_i (from predict()) for reuse in update()
    MatXsig X_pred = MatXsig::Zero();

    /// Compute UKF weights (Merwe scaled sigma points)
    /// @param Wm Output mean weights, length (2NX+1)
    /// @param Wc Output covariance weights, length (2NX+1)
    void weights(VecSig& Wm, VecSig& Wc) const {
        const T lambda = alpha * alpha * (T(NX) + kappa) - T(NX);
        const T c = T(NX) + lambda;

        Wm.setConstant(T(1) / (T(2) * c));
        Wc = Wm;

        Wm(0) = lambda / c;
        Wc(0) = lambda / c + (T(1) - alpha * alpha + beta);
    }

    /// Generate sigma points for the UKF (Merwe scaled sigma points)
    /// @param x0 Mean state, NX x 1
    /// @param P0 State covariance, NX x NX
    /// @return Sigma point matrix X, NX x (2NX+1)
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
