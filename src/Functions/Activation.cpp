#include "NNTK/Functions/Activation.hpp"

#include <cmath>
#include <algorithm>

#include "NNTK/Core/Types.hpp"

namespace NN
{

NDArray
AFSigmoid::
function(const NDArray &x) const
{
    NDArray y = x;
    for (auto &i : y)
        i = 1. / (1. + exp(-i));
    return y;
}

NDArray
AFReLU::
function(const NDArray &x) const
{
    NDArray y = x;
    for (auto &i : y)
        i = i > 0 ? i : 0;
    return y;
}

NDArray
AFReLU::
derivative(const NDArray &x) const
{
    NDArray dy = x;
    for (auto &i : dy)
        i = i > 0;
    return dy;
}

NDArray
AFLeakyReLU::
function(const NDArray &x) const
{
    NDArray y = x;
    for (auto &i : y)
        i = i >= 0 ? i : m_alpha * i;
    return y;
}

NDArray
AFLeakyReLU::
derivative(const NDArray &x) const
{
    NDArray dy = x;
    for (auto &i : dy)
        i = i >= 0 ? 1 : m_alpha;
    return dy;
}

namespace Activation
{

ActivationFunction *None = new AFNone;
ActivationFunction *Sigmoid = new AFSigmoid;
ActivationFunction *ReLU = new AFReLU;
ActivationFunction * LeakyReLU(nn_type alpha) { return new AFLeakyReLU(alpha); }

}

} // namespace NN
