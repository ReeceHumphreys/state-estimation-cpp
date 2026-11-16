#pragma once

#include <array>

namespace filters {

// Results returned by the filters. Plain trivially copyable structs (memcpy / telemetry safe);
// matrix fields are column-major, e.g. Eigen::Map<const Eigen::Matrix<T, NX, NZ>>(u.K.data()).

/// Measurement innovation, before or as it is applied.
template <typename T, int NZ> struct Innovation {
    std::array<T, NZ> nu;     // innovation z - h(x)
    std::array<T, NZ * NZ> S; // innovation covariance
    T nis;                    // normalised innovation squared nu^T S^-1 nu, ~ chi^2(NZ)
};

/// A Kalman (KF / EKF / UKF) measurement update that was applied.
template <typename T, int NX, int NZ> struct KalmanUpdate {
    std::array<T, NZ> nu;
    std::array<T, NZ * NZ> S;
    std::array<T, NX * NZ> K; // gain
    T nis;
};

/// A square-root information filter measurement update that was applied. The SRIF never forms S
/// or K; use SquareRootInformationFilter::innovation() before updating if S is needed.
template <typename T, int NZ> struct SrifUpdate {
    std::array<T, NZ> nu;
    T nis;
};

} // namespace filters
