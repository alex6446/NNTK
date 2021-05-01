set(SRCROOT ${PROJECT_SOURCE_DIR}/test)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(SRC_TEST
    ${SRCROOT}/CatchMain.cpp
    ${SRCROOT}/MX/Array/Operations.cpp
)
set(TARGET_TEST test-nntk)

add_executable(${TARGET_TEST} ${SRC_TEST})
target_link_libraries(${TARGET_TEST} PRIVATE nntk)
target_link_libraries(${TARGET_TEST} PRIVATE Catch2::Catch2)
nntk_setup(${TARGET_TEST} "${SRC_TEST}")

include(CTest)
include(Catch)
catch_discover_tests(${TARGET_TEST})

