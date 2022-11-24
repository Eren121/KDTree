cmake_minimum_required(VERSION 3.23)
project(KDTree)

set(CMAKE_CXX_STANDARD 17)

add_executable(KDTree main.cpp KDTree.cpp KDTree.h)

find_package(OpenMP REQUIRED)
target_link_libraries(KDTree OpenMP::OpenMP_CXX)