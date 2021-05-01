cmake_minimum_required(VERSION 3.19)

project(NNTK LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/vendor/Catch2/contrib)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(vendor/Catch2)
add_subdirectory(test)
