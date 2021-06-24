#pragma once

#ifdef NN_GPU_ENABLED
#define NN_GPU_FUNCTION_CALL(condition, function, arguments) \
    { \
        if (NN_USE_GPU && condition) { \
            function##_gpu arguments; \
        } else { \
            function##_cpu arguments; \
        } \
    }
#else
#define NN_GPU_FUNCTION_CALL(condition, function, arguments) \
    function##_cpu arguments;
#endif


namespace NN
{

enum Device { CPU, GPU };
extern bool NN_USE_GPU;

} // namespace NN


