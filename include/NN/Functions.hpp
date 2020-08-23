#pragma once

#include <cmath>
#include <algorithm>

#include "Matrix.hpp"

namespace NN {

    namespace Activation {

        #define None nullptr

        template <class T>
        T Sigmoid (T x, int mode = 0, float hp = 1) {
            switch (mode) {
                case 0: return 1 / (1 + exp(-x));
                case 1: return Sigmoid(x, 0) * (1 - Sigmoid(x, 0));
                default: abort();
            }
        }
        
        template <class T>
        T ReLU (T x, int mode = 0, float hp = 1) {
            switch (mode) {
                case 0: return std::max((T)0, x);
                case 1: return x < 0 ? 0 : 1;
                default: abort();
            }
        }

    }

    namespace Loss {

        #define None nullptr

        // Assuming each column is a separate output layer
        const void* MSE (const MX::Matrixf& A, const MX::Matrixf& Y, int mode, float hp);
        
    }

}
