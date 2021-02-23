#pragma once

#include <stdexcept>

#define NN_RUNTIME_ERROR(condition, message) if (condition) { throw std::runtime_error(std::string(__FUNCTION__) + ":" + message); }

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

    using nn_type = float;

    enum class DeviceType { CPU, GPU };

    extern bool NN_USE_GPU;

} // namespace NN
