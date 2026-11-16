#include "data_structures/ring_buffer.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("RingBuffer overwrites the oldest element and indexes both ways") {
    data_structures::RingBuffer<int, 3> rb;
    REQUIRE(rb.empty());

    REQUIRE(rb.push(1) == 0);
    REQUIRE(rb.push(2) == 0);
    REQUIRE(rb.push(3) == 0);
    REQUIRE(rb.full());
    REQUIRE(rb.push(4) == 1); // returns the value it overwrote
    REQUIRE(rb.push(5) == 2);

    REQUIRE(rb.size() == 3);
    REQUIRE(rb.at_oldest(0) == 3);
    REQUIRE(rb.at_oldest(2) == 5);
    REQUIRE(rb.at_newest(0) == 5);
    REQUIRE(rb.at_newest(2) == 3);

    rb.reset();
    REQUIRE(rb.empty());
    rb.push(7);
    REQUIRE(rb.at_oldest(0) == 7);
    REQUIRE(rb.at_newest(0) == 7);
}
