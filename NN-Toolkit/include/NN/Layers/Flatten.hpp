#pragma once

#include "NN/Matrix.hpp"
#include "NN/Error.hpp"
#include "NN/Layers/Base.hpp"
#include "NN/Functions.hpp"

namespace NN {

    namespace Layer {

        class Flatten : public Base {
        private:

            // neurons for the whole dataset
            MX::Matrixf Z; 
            MX::Matrixf A;
            MX::Matrixf dZ;

            const std::vector<MX::Image>* X;

            // total number of neurons
            int size;

            // random initialization range
            float rand_a;
            float rand_b;

        public:

            Flatten (
                float (*activation) (float, int, float) = Activation::None,
                bool bias = false,
                float rand_from = -1,
                float rand_to = 1,
                float hyperparameter = 1
            );

            inline Flatten* sActivation (float (*g) (float, int, float)) { this->g = g; return this; }
            inline Flatten* sBias (bool bias) { this->bias = bias; return this; }
            inline Flatten* sRandFrom (float rand_a) { this->rand_a = rand_a; return this; }
            inline Flatten* sRandTo (float rand_b) { this->rand_b = rand_b; return this; }
            inline Flatten* sHyperparameter (float hp) { this->hp = hp; return this; }

            void forwardProp (const void* X) override;
            void backProp (const void* gradient) override;
            void update (float learning_rate) override;
            void bind (const std::vector<int>& dimensions) override;

            inline const void* getA () const override { return &A; }
            const void* getGradient () const override;
            inline std::vector<int> getDimensions () const override { return { size }; }

            inline void print () const override { std::cout << *this; }
            void save (std::string file) const override;
            void load (std::string file) override;
            inline void output (std::ostream& os) override { os << *this; }
            inline void input (std::istream& is) override { is >> *this; }

            friend std::ostream& operator<< (std::ostream& os, const Flatten& l);
            friend std::istream& operator>> (std::istream& is, Flatten& l);

        };

    } // namespace Layer

} // namespace NN