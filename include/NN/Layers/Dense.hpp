#pragma once

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class Dense : public Base {
        private:
            
            MX::Matrixf W;
            MX::Matrixf dW;

            // neurons for the whole dataset
            MX::Matrixf Z; 
            MX::Matrixf A;
            MX::Matrixf dZ;

            const MX::Matrixf* X;

            // number of neurons
            int size;

            // random initialization range
            float rand_a;
            float rand_b;

        public:

            Dense (
                int neurons=1,
                float (*activation) (float, int, float) = Activation::Sigmoid,
                bool bias = true,
                float rand_from = -1,
                float rand_to = 1,
                float hyperparameter = 1
            );

            inline Dense* sNeurons (int size) { this->size = size; return this; }
            inline Dense* sActivation (float (*g) (float, int, float)) { this->g = g; return this; }
            inline Dense* sBias (bool bias) { this->bias = bias; return this; }
            inline Dense* sRandFrom (float rand_a) { this->rand_a = rand_a; return this; }
            inline Dense* sRandTo (float rand_b) { this->rand_b = rand_b; return this; }
            inline Dense* sHyperparameter (float hp) { this->hp = hp; return this; }

            void forwardProp (const void* X) override;
            void backProp (const void* gradient) override;
            void update (float learning_rate) override;
            void bind (const std::vector<int>& dimensions) override;

            inline const void* getA () const override { return &A; }
            inline const void* getGradient () const override { return new MX::Matrixf(MX::Dot(W.transpose(), dZ)); }
            inline std::vector<int> getDimensions () const override { return { size }; }

            inline void print () const override { std::cout << *this; }
            void save (std::string file) const override;
            void load (std::string file) override;
            inline void output (std::ostream& os) override { os << *this; }
            inline void input (std::istream& is) override { is >> *this; }

            friend std::ostream& operator<< (std::ostream& os, const Dense& l);
            friend std::istream& operator>> (std::istream& is, Dense& l);

        };

    } // namespace Layer

} // namespace NN
