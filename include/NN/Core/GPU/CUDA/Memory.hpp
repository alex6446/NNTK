#include <cstddef>

namespace NN
{
namespace internal
{

    void allocate_memory_cuda(void **pointer, size_t bytes);
    void free_memory_cuda(void **pointer);

} // namespace NN
} // namespace internal
