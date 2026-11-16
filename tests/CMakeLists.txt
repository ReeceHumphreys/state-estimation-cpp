find_package(Catch2 3 REQUIRED)

# Running average tests
add_executable(test_running_average
    test_running_average.cpp
)
target_link_libraries(test_running_average
    PRIVATE
        se::se
        Catch2::Catch2WithMain
)

# Kalman filter tests
add_executable(test_kalman
    test_kalman.cpp
)
target_link_libraries(test_kalman
    PRIVATE
        se::se
        Catch2::Catch2WithMain
)

# Square-root information filter tests
add_executable(test_srif
    test_srif.cpp
)
target_link_libraries(test_srif
    PRIVATE
        se::se
        Catch2::Catch2WithMain
)

include(CTest)
include(Catch)

catch_discover_tests(test_running_average)
catch_discover_tests(test_kalman)
catch_discover_tests(test_srif)
