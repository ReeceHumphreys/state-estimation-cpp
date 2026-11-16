#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>

namespace filters::detail {

/// Column-major N x N identity as plain storage.
template <typename T, int N> constexpr std::array<T, std::size_t(N) * N> identity() {
    std::array<T, std::size_t(N) * N> a{};
    for (int i = 0; i < N; ++i)
        a[std::size_t(i) * N + i] = T(1);
    return a;
}

/// Copy a fixed-size Eigen value into column-major plain storage.
template <class Derived> auto to_array(const Eigen::MatrixBase<Derived>& m) {
    std::array<typename Derived::Scalar, Derived::SizeAtCompileTime> a;
    Eigen::Map<typename Derived::PlainObject>(a.data()) = m;
    return a;
}

} // namespace filters::detail
