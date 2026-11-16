#pragma once

#include "data_structures/ring_buffer.hpp"

#include <cstdint>
#include <type_traits>

namespace estimators {

template <typename T> struct RunningAverage {

    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

    T process(T x) {
        ++count;
        mean += (x - mean) / static_cast<T>(count);
        return mean;
    }

    void reset() {
        mean = T{};
        count = 0;
    }

  private:
    T mean = T{};
    std::uint64_t count = 0; // a float count stops incrementing at 2^24
};

template <typename T, std::uint32_t N> struct MovingAverage {

    static_assert(std::is_floating_point_v<T>, "T must be floating point");
    static_assert(N > 0, "N must be greater than zero");

    T process(T value) {
        const T outgoing = buffer.push(value);
        sum += value;

        if (buffer.size() == N) {
            sum -= outgoing;
            return sum / static_cast<T>(N);
        } else {
            return sum / static_cast<T>(buffer.size());
        }
    }

    void reset() {
        buffer.reset();
        sum = T{};
    }

  private:
    data_structures::RingBuffer<T, N> buffer;
    T sum = {};
};

template <typename T> struct ExponentialMovingAverage {
    static_assert(std::is_floating_point_v<T>, "T must be floating point");

    explicit ExponentialMovingAverage(T alpha) : alpha_(alpha) {}

    T process(T x) {
        if (!initialized_) {
            y_ = x;
            initialized_ = true;
        } else {
            y_ = (T{1} - alpha_) * y_ + alpha_ * x;
        }
        return y_;
    }

    void reset() {
        y_ = T{};
        initialized_ = false;
    }

    void set_alpha(T a) {
        alpha_ = a;
    }

    T alpha() const {
        return alpha_;
    }

  private:
    T alpha_{T{0.1}};
    T y_{};
    bool initialized_{false};
};

} // namespace estimators
