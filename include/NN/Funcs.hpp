#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "MX/Array.hpp"

namespace NN {

    namespace Activation {

        extern MX::Array<float> (*None)(const MX::Array<float> &, int, float);

        template <class T>
        MX::Array<T>
        Sigmoid(const MX::Array<T> &X, int mode=0, float hp=1)
        {
            MX::Array<T> Y;
            switch (mode) {
            case 0:
                Y = X;
                for (auto &i : Y)
                    i = 1 / (1 + exp(-i));
                return Y;
            case 1:
                Y = Sigmoid(X, 0);
                return Y * (1 - Y);
            default:
                throw std::invalid_argument(__FUNCTION__);
            }
        }

        template <class T>
        MX::Array<T>
        ReLU(const MX::Array<T> &X, int mode=0, float hp=1)
        {
            using size_type = typename MX::Array<T>::size_type;
            MX::Array<T> Y = X;
            switch (mode) {
            case 0:
                for (auto &i : Y)
                    i = std::max((T)0, i);
                return Y;
            case 1:
                for (auto &i : Y)
                    i = i < 0 ? 0 : 1;
                return Y;
            default:
                throw std::invalid_argument(__FUNCTION__);
            }
        }

    }

    namespace Loss {

        extern const MX::Array<float> (*None)(const MX::Array<float> &, const MX::Array<float> &, int, float);

        // Assuming each column is a separate output layer
        const MX::Array<float> MSE(const MX::Array<float> &A, const MX::Array<float> &Y, int mode, float hp);
        const MX::Array<float> BCE(const MX::Array<float> &A, const MX::Array<float> &Y, int mode, float hp);

    }

}
