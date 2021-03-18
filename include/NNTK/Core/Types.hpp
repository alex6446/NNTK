#pragma once

#include "NNTK/MX/Array.hpp"

namespace NN
{

using nn_type = float;
using NDArray = MX::Array<nn_type>;
using NDSize = MX::Array<typename MX::Array<nn_type>::size_type>;

} // namespace NN

