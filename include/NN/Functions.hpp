#pragma once

#include <cmath>
#include <algorithm>

namespace NN {

    namespace Activation {

        template <class T>
        T Sigmoid (T x, int mode = 0) {
            switch (mode) {
                case 0: return 1 / (1 + exp(-x));
                case 1: return Sigmoid(x, 0) * (1 - Sigmoid(x, 0));
                default: abort();
            }
        }
        
        template <class T>
        T ReLU (T x, int mode = 0) {
            switch (mode) {
                case 0: return std::max((T)0, x);
                case 1: return x < 0 ? 0 : 1;
                default: abort();
            }
        }

    }

    namespace Loss {

        
        
    }

}
