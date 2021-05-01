macro(nntk_setup target sources)

    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)

    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE CUDA)

endmacro()

