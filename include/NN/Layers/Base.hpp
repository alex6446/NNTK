#pragma once

#include <vector>

#include "NN/Matrix.hpp"

namespace NN {

    namespace Layer {

        class Base {
        protected:

            bool bias;
            MX::Matrixf b; // bias vector
            MX::Matrixf db;
            
            float (*g) (float, int, float); // activation function
            float hp; // hyperparameter

            bool bound;

        public:

            virtual void forwardProp (const void* X) = 0;
            virtual void backProp (const void* gradient) = 0;

            virtual void update (float learning_rate) = 0;
            virtual void bind (const std::vector<int>& dimensions) = 0;
            virtual inline void reset () { bound = false; }

            virtual const void* getA () const = 0;
            virtual const void* getGradient () const = 0;
            virtual std::vector<int> getDimensions () const = 0;

        };

    } // namespace Layer

} // namespace NN