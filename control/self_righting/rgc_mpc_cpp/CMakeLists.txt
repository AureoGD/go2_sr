cmake_minimum_required(VERSION 3.18)
project(rgc_mpc_cpp LANGUAGES C CXX)

# Built by build.sh (which also builds the pinned BLASFEO / HPIPM into third_party/install).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(RGC_DEPS_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/third_party/install" CACHE PATH "BLASFEO/HPIPM install prefix")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

foreach(lib hpipm blasfeo)
    if(NOT EXISTS "${RGC_DEPS_PREFIX}/lib/lib${lib}.a")
        message(FATAL_ERROR "lib${lib}.a not found in ${RGC_DEPS_PREFIX}/lib (run build.sh)")
    endif()
endforeach()

# ---------------------------------------------------------------- C++ core (no Python inside)
add_library(rgc_mpc_core STATIC src/hpipm_stage_qp_solver.cpp)
target_include_directories(rgc_mpc_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${RGC_DEPS_PREFIX}/include ${RGC_DEPS_PREFIX}/include/hpipm ${RGC_DEPS_PREFIX}/include/blasfeo)
target_link_libraries(rgc_mpc_core
    PUBLIC Eigen3::Eigen
    PRIVATE ${RGC_DEPS_PREFIX}/lib/libhpipm.a ${RGC_DEPS_PREFIX}/lib/libblasfeo.a m)
set_target_properties(rgc_mpc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(rgc_mpc_core PRIVATE -O3 -Wall -Wextra)

# ---------------------------------------------------------------- Python module
pybind11_add_module(_rgc_mpc_cpp bindings/module.cpp)
target_link_libraries(_rgc_mpc_cpp PRIVATE rgc_mpc_core)
# the compiled module lands next to the Python package files
set_target_properties(_rgc_mpc_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
