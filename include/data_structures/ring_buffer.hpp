#include <array>
#include <cstdint>

template <typename T, std::uint32_t N> struct RingBuffer {
    static_assert(N > 0, "N must be greater than zero");

    std::array<T, N> buffer{};
    std::uint32_t idx = 0;
    std::uint32_t size = 0;

    /// Add a value to the buffer, returning the overwritten value
    T push(T value) {
        T old = buffer[idx];
        buffer[idx] = value;

        if (++idx == N) {
            idx = 0;
        }

        if (size < N) {
            ++size;
        }

        return old;
    }

    /// Reset the buffer to initial state
    void reset() {
        buffer.fill(T{});
        idx = 0;
        size = 0;
    }
};
