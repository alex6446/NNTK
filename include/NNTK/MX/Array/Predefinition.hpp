#pragma once

#include "NNTK/Core/Device.hpp"

namespace NN::MX
{

  template<typename T, Device D = CPU>
    class Array;

  template<typename T, Device D, Device V>
    using ArrayPrefCPU = Array<T, (V == GPU && D == GPU ? GPU : CPU)>;

  template<typename T, Device D, Device V>
    using ArrayPrefGPU = Array<T, (V == CPU && D == CPU ? CPU : GPU)>;

} // namespace NN::MX


