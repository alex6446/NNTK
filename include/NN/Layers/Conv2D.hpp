#pragma once

#include <vector>

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class Conv2D : public Base {
        private:
            
            std::vector<MX::Filter> W;
            std::vector<MX::Filter> dW;

            // volumes for the whole dataset
            std::vector<MX::Image> Z; 
            std::vector<MX::Image> A;
            std::vector<MX::Image> dZ;
            std::vector<MX::Image> dA;

            const std::vector<MX::Image>* X;

            int size; // number of filters 
            int f; // filter dimensions
            int p; // padding
            int s; // stride

            // random initialization range
            int rand_a;
            int rand_b;

        public:

            Conv2D (
                int filters,
                int filter_size=3,
                int padding=0,
                int stride=1,
                float (*activation) (float, int, float) = Activation::ReLU,
                bool bias = true,
                int rand_from = -1,
                int rand_to = 1,
                float hyperparameter = 1
            );

            void forwardProp (const void* X) override;
            void backProp (const void* gradient) override;
            void update (float learning_rate) override;
            void bind (const std::vector<int>& dimensions) override;

            inline const void* getA () const override { return &A; }
            const void* getGradient () const override;
            inline std::vector<int> getDimensions () const override { return { size }; }
        };

    } // namespace Layer

} // namespace NN
