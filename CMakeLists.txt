cmake_minimum_required(VERSION 3.20)

project(Filters VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(filters INTERFACE)
add_library(filters::filters ALIAS filters)

target_link_libraries(filters
  INTERFACE Eigen3::Eigen
)

target_include_directories(filters
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(CTest)
enable_testing()
add_subdirectory(tests)

# ---------- install rules ----------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS filters
  EXPORT FiltersTargets
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT FiltersTargets
  FILE FiltersTargets.cmake
  NAMESPACE filters::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Filters
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/FiltersConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/FiltersConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/FiltersConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Filters
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/FiltersConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/FiltersConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Filters
)
