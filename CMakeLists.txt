cmake_minimum_required(VERSION 3.23)
project(KDTree LANGUAGES CXX CUDA)

#set(CMAKE_CUDA_FLAGS "-G")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_package(OpenMP REQUIRED)

# Library
add_library(KDTree KDTree.cu KDTree.cuh Math.h KDTreeDevice.cu KDTreeDevicePtr.cuh Math.h)
set_property(TARGET KDTree PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(KDTree OpenMP::OpenMP_CXX)
target_include_directories(KDTree PUBLIC .)

if(KDTREE_TRACE)
    target_compile_definitions(KDTree PRIVATE KDTREE_TRACE=1)
endif()

# Test
add_subdirectory(tests)