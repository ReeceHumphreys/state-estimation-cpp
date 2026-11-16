#pragma once

#include <array>
#include <cassert>
#include <cstdint>

namespace data_structures {

/// Fixed-capacity ring buffer: push() overwrites the oldest element once full.
template <typename T, std::uint32_t N> struct RingBuffer {
    static_assert(N > 0, "N must be greater than zero");

    /// Push a value; returns the element it overwrote (T{} while not yet full).
    T push(T value) {
        T old = buffer_[idx_];
        buffer_[idx_] = value;

        if (++idx_ == N)
            idx_ = 0;
        if (size_ < N)
            ++size_;
        return old;
    }

    void reset() {
        buffer_.fill(T{});
        idx_ = 0;
        size_ = 0;
    }

    std::uint32_t size() const {
        return size_;
    }
    bool full() const {
        return size_ == N;
    }
    bool empty() const {
        return size_ == 0;
    }

    /// Chronological indexing: i = 0 is the oldest, i = size() - 1 the newest.
    const T& at_oldest(std::uint32_t i) const {
        assert(i < size_);
        const std::uint32_t j = (full() ? idx_ : 0) + i;
        return buffer_[(j >= N) ? (j - N) : j];
    }

    /// Reverse chronological indexing: i = 0 is the newest.
    const T& at_newest(std::uint32_t i) const {
        assert(i < size_);
        const std::uint32_t newest = (idx_ == 0) ? (N - 1) : (idx_ - 1);
        return buffer_[(newest >= i) ? (newest - i) : (newest + N - i)];
    }

  private:
    std::array<T, N> buffer_{};
    std::uint32_t idx_ = 0;
    std::uint32_t size_ = 0;
};

} // namespace data_structures
