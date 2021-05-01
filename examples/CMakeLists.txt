macro(nntk_add_example target sources)

    add_executable(${target} ${sources})
    target_link_libraries(${target} PRIVATE nntk)
    nntk_setup(${target} "${sources}")

endmacro()

set(SRCROOT ${PROJECT_SOURCE_DIR}/examples)

nntk_add_example(examples-cuda ${SRCROOT}/cuda.cpp)
nntk_add_example(examples-main ${SRCROOT}/main.cpp)

