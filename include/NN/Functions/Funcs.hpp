#pragma once

#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include "NN/MX/Array.hpp"

namespace NN
{
namespace Activation
{

enum class Mode { Base, Derivative };

template <class T>
MX::Array<T>
None(const MX::Array<T> &X, Mode mode=Mode::Base, float hp=1)
{
    switch (mode) {
    case Mode::Base: return X;
    case Mode::Derivative: return MX::Ones<T>(X.shape());
    default: throw std::runtime_error("None: wrong mode");
    }
}

template <class T>
MX::Array<T>
Sigmoid(const MX::Array<T> &X, Mode mode=Mode::Base, float hp=1)
{
    MX::Array<T> Y;
    switch (mode) {
    case Mode::Base:
        Y = X;
        for (auto &i : Y)
            i = 1. / (1. + exp(-i));
        return Y;
    case Mode::Derivative:
        Y = Sigmoid(X, Mode::Base);
        return Y * (1. - Y);
    default: throw std::runtime_error("Sigmoid: wrong mode");
    }
}

template <class T>
MX::Array<T>
ReLU(const MX::Array<T> &X, Mode mode=Mode::Base, float hp=1)
{
    using size_type = typename MX::Array<T>::size_type;
    MX::Array<T> Y = X;
    switch (mode) {
    case Mode::Base:
        for (auto &i : Y)
            i = std::max((T)0, i);
        return Y;
    case Mode::Derivative:
        for (auto &i : Y)
            i = i < 0. ? 0. : 1.;
        return Y;
    default: throw std::runtime_error("ReLU: wrong mode");
    }
}

} // namespace Activation

namespace Loss
{

enum class Mode { Base, Derivative };

// Assuming each column is a separate output layer

template <class T>
MX::Array<T>
MSE(const MX::Array<T> &A, const MX::Array<T> &Y, Mode mode=Mode::Base, float hp=1)
{
    switch (mode) {
    case Mode::Base: return MX::Sum((A - Y) * (A - Y), 0) / A.shape(0);
    case Mode::Derivative: return 2. * (A - Y) / A.shape(0);
    default: throw std::runtime_error("mean_squared_error: wrong mode");
    }
}

template <class T>
MX::Array<T>
BCE(const MX::Array<T> &A, const MX::Array<T> &Y, Mode mode=Mode::Base, float hp=1)
{
    using size_type = typename MX::Array<T>::size_type;
    MX::Array<T> loss = MX::Empty<T>(A.shape());
    switch (mode) {
    case Mode::Base:
        for (size_type i = 0; i < A.size(); ++i)
            loss.data(i) = Y.data(i) * std::log(A.data(i)) + (1. - Y.data(i)) * std::log(1. - A.data(i));
        return -MX::Sum(loss, 0) / A.shape(0);
    case Mode::Derivative:
        for (size_type i = 0; i < A.size(); ++i)
            loss.data(i) = (A.data(i) - Y.data(i)) / (A.data(i) * (1. - A.data(i)));
        return loss / A.shape(0);
    default: throw std::runtime_error("binary_crossentropy: wrong mode");
    }
}

} // namespace Loss

} // namespace NN
