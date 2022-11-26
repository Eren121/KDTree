project(KDTree-test LANGUAGES CXX CUDA)

set(SRC test.cu QueryOp.h)

# Enable tests for Houdini if Houdini is found
find_package(Houdini)
if(Houdini_FOUND)
    list(APPEND SRC test_houdini.cu)
endif()

message(${SRC})
add_executable(KDTree-test ${SRC})
target_link_libraries(KDTree-test PRIVATE KDTree)
set_property(TARGET KDTree-test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(catch2)
target_link_libraries(KDTree-test PRIVATE Catch2WithMain)

if(Houdini_FOUND)
    # Isolate Houdini because it work badly with nvcc
    add_library(KDTree-test-houdini houdini.cpp houdini.hpp)
    target_link_libraries(KDTree-test-houdini PRIVATE Houdini)
    target_link_libraries(KDTree-test-houdini PRIVATE tbb) # needed for some reason on my computer
    target_link_libraries(KDTree-test PRIVATE KDTree-test-houdini)
endif()

target_compile_definitions(KDTree-test PRIVATE MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models")