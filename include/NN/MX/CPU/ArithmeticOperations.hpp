#include "NN/Core/Base.hpp"

namespace NN
{
namespace MX
{

    template<typename T>
    class Array;

namespace internal
{

    template<typename T>
    using size_type = typename Array<T>::size_type;

#define MX_ARRAY_OPERATION_AA_CPU(op, name, check) \
    template<typename T> \
    void \
    name##_array_array_cpu(Array<T> &result, const Array<T> &array) \
    { \
        std::cout << "Running on CPU!" << std::endl; \
        using namespace std; \
        clock_t begin = clock(); \
        for (size_type<T> i = 0; i < result.size(); ++i) { \
            if (check) { NN_RUNTIME_ERROR(!array.data(i), "array: division by zero") } \
            result.data(i) op##= array.data(i); \
        } \
        clock_t end = clock(); \
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; \
        cout << "TIME: " << elapsed_secs << "s" << endl; \
    }

    MX_ARRAY_OPERATION_AA_CPU(+, add, 0)
    MX_ARRAY_OPERATION_AA_CPU(-, subtract, 0)
    MX_ARRAY_OPERATION_AA_CPU(*, multiply, 0)
    MX_ARRAY_OPERATION_AA_CPU(/, divide, 1)


} // namespace NN
} // namespace MX
} // namespace internal
