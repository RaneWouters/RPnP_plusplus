# Eigen3 dependency resolution (bundled, system, or FetchContent)
function(setup_eigen3)
    set(R2PPNP_EIGEN_BUNDLED_ACTIVE OFF PARENT_SCOPE)
    if(USE_BUNDLED_EIGEN3)
        # Use bundled Eigen3 headers (header-only, no compilation needed)
        if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/eigen3/Eigen/Core")
            add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/eigen3")
            set(R2PPNP_EIGEN_BUNDLED_ACTIVE ON PARENT_SCOPE)
            message(STATUS "Using bundled Eigen3 headers")
            return()
        else()
            message(WARNING "Bundled Eigen3 not found at ${CMAKE_CURRENT_LIST_DIR}/eigen3, falling back")
        endif()
    endif()

    if(USE_SYSTEM_EIGEN3)
        find_package(Eigen3 QUIET NO_MODULE)
        if(TARGET Eigen3::Eigen)
            message(STATUS "Using system Eigen3")
            return()
        endif()
    endif()

    # Last resort: FetchContent download
    message(STATUS "Downloading Eigen3 via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084ac88434fdbd9d3f6ebc0ff3c58f836e39283c7fa715103d71d31de416
    )
    FetchContent_MakeAvailable(Eigen3)
    message(STATUS "Eigen3 downloaded and built")
endfunction()

setup_eigen3()
