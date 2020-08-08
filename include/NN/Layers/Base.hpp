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

        public:

            virtual void forwardProp (const MX::Matrixf& X);
            virtual void backProp (const MX::Matrixf& gradient);

            virtual void update (float learning_rate) = 0;
            virtual void bind (const std::vector<int>& dimensions) = 0;

            virtual MX::Matrixf const& getA () const;
            virtual MX::Matrixf getGradient () const;
            virtual std::vector<int> getDimensions () const = 0;

        };

    } // namespace Layer

} // namespace NN