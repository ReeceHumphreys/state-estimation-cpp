#pragma once

#include <array>
#include <cassert>
#include <cstdint>

template <typename T, std::uint32_t N> struct RingBuffer {
    static_assert(N > 0, "N must be greater than zero");

    std::array<T, N> buffer{};
    std::uint32_t idx = 0;
    std::uint32_t size = 0;

    T push(T value) {
        T old = buffer[idx];
        buffer[idx] = value;

        if (++idx == N)
            idx = 0;
        if (size < N)
            ++size;
        return old;
    }

    void reset() {
        buffer.fill(T{});
        idx = 0;
        size = 0;
    }

    bool full() const {
        return size == N;
    }
    bool empty() const {
        return size == 0;
    }

    // Chronological indexing: i=0 is oldest, i=size-1 is newest
    const T& at_oldest(std::uint32_t i) const {
        assert(i < size);
        // oldest element index:
        const std::uint32_t start = full() ? idx : 0;
        const std::uint32_t j = start + i;
        return buffer[(j >= N) ? (j - N) : j];
    }

    // Reverse chronological: i=0 newest
    const T& at_newest(std::uint32_t i) const {
        assert(i < size);
        // newest element is idx-1 (wrap)
        const std::uint32_t newest = (idx == 0) ? (N - 1) : (idx - 1);
        const std::int32_t j = static_cast<std::int32_t>(newest) - static_cast<std::int32_t>(i);
        return buffer[(j < 0) ? (static_cast<std::uint32_t>(j + N))
                              : static_cast<std::uint32_t>(j)];
    }
};
