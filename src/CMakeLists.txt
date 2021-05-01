include(${PROJECT_SOURCE_DIR}/cmake/Macros.cmake)

set(SRCROOT ${PROJECT_SOURCE_DIR}/src)

set(SRC_NNTK
    ${SRCROOT}/Core/GPU/CUDA/Memory.cu
    ${SRCROOT}/Core/Device.cpp
    ${SRCROOT}/Functions/Activation.cpp
    ${SRCROOT}/Functions/Loss.cpp
)

add_library(nntk STATIC ${SRC_NNTK})
nntk_setup(nntk "${SRC_NNTK}")
