#include "estimators/averages.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

static bool approx_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps * (1.0 + std::fabs(a) + std::fabs(b));
}

TEST_CASE("RunningAverage computes exact arithmetic mean") {
    estimators::RunningAverage<double> avg;

    SECTION("first sample equals input") {
        avg.reset();
        REQUIRE(approx_equal(avg.process(10.0), 10.0));
    }

    SECTION("sequence mean is exact") {
        avg.reset();
        REQUIRE(approx_equal(avg.process(1.0), 1.0));
        REQUIRE(approx_equal(avg.process(2.0), 1.5));
        REQUIRE(approx_equal(avg.process(3.0), 2.0));
        REQUIRE(approx_equal(avg.process(4.0), 2.5));
    }
}

TEST_CASE("MovingAverage computes sliding window mean with warm-up") {
    constexpr std::uint32_t N = 3;
    estimators::MovingAverage<double, N> ma;

    SECTION("warm-up uses mean over available samples") {
        REQUIRE(approx_equal(ma.process(1.0), 1.0)); // [1] -> 1
        REQUIRE(approx_equal(ma.process(2.0), 1.5)); // [1,2] -> 1.5
        REQUIRE(approx_equal(ma.process(3.0), 2.0)); // [1,2,3] -> 2
    }

    SECTION("steady-state slides over last N samples") {
        // Fill
        REQUIRE(approx_equal(ma.process(1.0), 1.0));
        REQUIRE(approx_equal(ma.process(2.0), 1.5));
        REQUIRE(approx_equal(ma.process(3.0), 2.0)); // [1,2,3]

        // Slide
        REQUIRE(approx_equal(ma.process(4.0), 3.0)); // [2,3,4] -> 3
        REQUIRE(approx_equal(ma.process(5.0), 4.0)); // [3,4,5] -> 4
        REQUIRE(approx_equal(ma.process(6.0), 5.0)); // [4,5,6] -> 5
    }

    SECTION("reset clears history") {
        (void)ma.process(10.0);
        (void)ma.process(20.0);
        (void)ma.process(30.0);

        ma.reset();

        REQUIRE(approx_equal(ma.process(7.0), 7.0));
        REQUIRE(approx_equal(ma.process(8.0), 7.5));
    }
}
