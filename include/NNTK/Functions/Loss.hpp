#pragma once

#include "NNTK/Core/Base.hpp"

namespace NN
{

class LossFunction
{
public:

    virtual ~LossFunction() = default;

    virtual nn_type function(const NDArray &output, const NDArray &target) const = 0;
    virtual NDArray derivative(const NDArray &output, const NDArray &target) const = 0;

};

class LFMeanSquareError : public LossFunction
{
public:

    nn_type function(const NDArray &output, const NDArray &target) const override
    { return NDArray::sum((output - target) * (output - target)) / output.shape(0); }

    NDArray derivative(const NDArray &output, const NDArray &target) const override
    { return 2. * (output - target) / output.shape(0); }

};

class LFBinaryCrossEntropy : public LossFunction
{
public:

    nn_type function(const NDArray &output, const NDArray &target) const override;
    NDArray derivative(const NDArray &output, const NDArray &target) const override;

};

namespace Loss
{

extern LossFunction *MSE;
extern LossFunction *BCE;;

} // namespace Loss

} // namespace NN

