project(NNTK-Test LANGUAGES CXX CUDA)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(TEST_SRC
    ${PROJECT_SOURCE_DIR}/CatchMain.cpp
    ${PROJECT_SOURCE_DIR}/MX/Array/Operations.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/Base.cpp
    ${CMAKE_SOURCE_DIR}/src/Core/GPU/CUDA/Memory.cpp
    ${CMAKE_SOURCE_DIR}/src/Functions/Activation.cpp
    ${CMAKE_SOURCE_DIR}/src/Functions/Loss.cpp
)

add_executable(nntk-test ${TEST_SRC})
target_include_directories(nntk-test PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(nntk-test PRIVATE Catch2::Catch2)

set_target_properties(nntk-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(${TEST_SRC} PROPERTIES LANGUAGE CUDA)

include(CTest)
include(Catch)
catch_discover_tests(nntk-test)

