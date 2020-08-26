#pragma once

#include <vector>

#include<NN/Layers/Base.hpp>

namespace NN {

    class Sequential {
    private:
        
        std::vector<Layer::Base*> L; // stack of layers

    public:

        Sequential ();
        ~Sequential ();
        
        void add (Layer::Base* layer);
        void fit (
            const MX::Matrixf& X, 
            const MX::Matrixf& Y,
            const void* (*l) (const MX::Matrixf&, const MX::Matrixf&, int, float), // loss function
            int batch_size = 1,
            int epochs = 2000,
            float learning_rate = 0.5,
            float hyperparameter = 1 // for loss function if needed
        );

        void fit (
            const std::vector<MX::Image>& X, 
            const MX::Matrixf& Y,
            const void* (*l) (const MX::Matrixf&, const MX::Matrixf&, int, float), // loss function
            int batch_size = 1,
            int epochs = 2000,
            float learning_rate = 0.5,
            float hyperparameter = 1 // for loss function if needed
        );

        void build (const std::vector<int>& dimensions); // sample input dimensions
        void reset () { for (auto& i : L) i->reset(); }

        MX::Matrixf predict (const MX::Matrixf& X);
        MX::Matrixf predict (const std::vector<MX::Image>& X);
        std::vector<MX::Image> predict2D (const std::vector<MX::Image>& X);

    };

}