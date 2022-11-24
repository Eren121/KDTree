cmake_minimum_required(VERSION 3.23)
project(KDTree)

set(CMAKE_CXX_STANDARD 17)

add_executable(KDTree main.cpp KDTree.cpp KDTree.h viewer.cpp viewer.h)

set(Easy3D_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Easy3D/build")
find_package(Easy3D COMPONENTS viewer REQUIRED)     # request Easy3D (recommended to request only needed components)
target_link_libraries(KDTree easy3d::viewer) # link to necessary Easy3D modules (add more if needed, e.g., algo)