cmake_minimum_required(VERSION 3.23)
project(KDTree)

set(CMAKE_CXX_STANDARD 17)

add_library(KDTree KDTree.cpp KDTree.h Utils.h)

add_executable(KDTree-test main.cpp)
target_link_libraries(KDTree-test KDTree)

find_package(OpenMP REQUIRED)
target_link_libraries(KDTree OpenMP::OpenMP_CXX)

find_package(Houdini)
if(Houdini_FOUND)
    add_executable(KDTree-Houdini-Test test-houdini.cpp)
    target_link_libraries(KDTree-Houdini-Test Houdini KDTree)
    target_link_libraries(KDTree-Houdini-Test Houdini tbb)
    houdini_configure_target(KDTree-Houdini-Test)
endif()