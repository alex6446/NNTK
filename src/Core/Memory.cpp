#include <cstddef>
#include <cstring>

namespace NN::internal
{

void
copy_memory_cpu(void *dst, const void *src, std::size_t count)
{ std::memcpy(dst, src, count); }

} // NN::internal
