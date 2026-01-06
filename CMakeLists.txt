cmake_minimum_required(VERSION 3.20)

project(StateEstimation VERSION 0.1.0 LANGUAGES CXX)

# -----------------------------
# Global C++ settings
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------
# Dependencies
# -----------------------------
find_package(Eigen3 REQUIRED NO_MODULE)

# -----------------------------
# Header-only library target
# -----------------------------
add_library(se INTERFACE)
add_library(se::se ALIAS se)

target_link_libraries(se
  INTERFACE
    Eigen3::Eigen
)

target_include_directories(se
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# -----------------------------
# Tests
# -----------------------------
include(CTest)
enable_testing()
add_subdirectory(tests)

# -----------------------------
# Install + package config
# -----------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install target
install(TARGETS se
  EXPORT SETargets
)

# Install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT SETargets
  FILE SETargets.cmake
  NAMESPACE se::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SE
)

# Version file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/SEConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Config file
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/SEConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SEConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SE
)

# Install config files
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/SEConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/SEConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SE
)
