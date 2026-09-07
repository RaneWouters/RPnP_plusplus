cmake_minimum_required(VERSION 3.16...3.26)
project(r2ppnp_cpp VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARK "Build syn benchmark" OFF)
option(USE_SYSTEM_EIGEN3 "Use system pre-installed Eigen3" OFF)
option(USE_BUNDLED_EIGEN3 "Use bundled Eigen3 headers" ON)
option(INSTALL_CPP_SDK "Install the C++ library, headers, and CMake package" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(3rdparty/find_dependencies.cmake)
include(cmake/CompilerOptions.cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(r2ppnp STATIC)
add_library(r2ppnp::r2ppnp ALIAS r2ppnp)
target_sources(r2ppnp PRIVATE
    src/r2ppnp.cpp
    src/hough_voting.cpp
    src/optimize_gn.cpp
    src/utils.cpp
)
target_include_directories(r2ppnp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(r2ppnp PUBLIC Eigen3::Eigen)
target_compile_definitions(r2ppnp PUBLIC EIGEN_MPL2_ONLY)

set_global_target_properties(r2ppnp)

if(INSTALL_CPP_SDK)
    install(FILES LICENSE THIRD_PARTY_NOTICES.md
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/r2ppnp)
    install(TARGETS r2ppnp
        EXPORT r2ppnpTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(DIRECTORY include/r2ppnp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    if(R2PPNP_EIGEN_BUNDLED_ACTIVE)
        install(DIRECTORY 3rdparty/eigen3/Eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/eigen3)
        install(FILES
            3rdparty/eigen3/COPYING.README
            3rdparty/eigen3/COPYING.MPL2
            3rdparty/eigen3/COPYING.BSD
            3rdparty/eigen3/COPYING.MINPACK
            3rdparty/eigen3/COPYING.GPL
            3rdparty/eigen3/COPYING.LGPL
            DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/r2ppnp/eigen3
        )
    endif()
    install(EXPORT r2ppnpTargets
        FILE r2ppnpTargets.cmake
        NAMESPACE r2ppnp::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/r2ppnp
    )
    configure_package_config_file(
        cmake/r2ppnpConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/r2ppnpConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/r2ppnp
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/r2ppnpConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/r2ppnpConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/r2ppnpConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/r2ppnp
    )
endif()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_BENCHMARK)
    message(FATAL_ERROR
        "The benchmark is Python-based; run benchmark/run_comparison.py from the repository root.")
endif()
