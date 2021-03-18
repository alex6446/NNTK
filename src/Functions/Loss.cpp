#include "NNTK/Functions/Loss.hpp"

#include <cmath>

#include "NNTK/Core/Types.hpp"

namespace NN
{

nn_type
LFBinaryCrossEntropy::
function(const NDArray &output, const NDArray &target) const
{
    NDArray loss = NDArray::empty(output.shape());
    for (NDArray::size_type i = 0; i < output.size(); ++i)
        loss.data(i) = target.data(i) * std::log(output.data(i)) +
            (1. - target.data(i)) * std::log(1. - output.data(i));
    return -NDArray::sum(loss) / output.shape(0);
}

NDArray
LFBinaryCrossEntropy::
derivative(const NDArray &output, const NDArray &target) const
{
    NDArray loss = NDArray::empty(output.shape());
    for (NDArray::size_type i = 0; i < output.size(); ++i)
        loss.data(i) = (output.data(i) - target.data(i)) /
            (output.data(i) * (1. - output.data(i)));
    return loss / output.shape(0);
}

namespace Loss
{

LossFunction *MSE = new LFMeanSquareError;
LossFunction *BCE = new LFBinaryCrossEntropy;;

} // namespace Loss

} // namespace NN
