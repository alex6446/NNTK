project(NNTK-Examples LANGUAGES CXX CUDA)

set(EXAMPLES_SRC
    #${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/cuda.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/Base.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/GPU/CUDA/Memory.cpp
)

add_executable(nntk-examples ${EXAMPLES_SRC})
target_include_directories(nntk-examples PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(nntk-examples PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(${EXAMPLES_SRC} PROPERTIES LANGUAGE CUDA)
