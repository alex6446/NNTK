#pragma once

#include <vector>

#include<NN/Layers/Base.hpp>
#include<NN/Layers/Dense.hpp>

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
            MX::Matrixf (*l) (const MX::Matrixf&, const MX::Matrixf&, int, float), // activation function
            int batch_size = 1,
            int epochs = 2000,
            float learning_rate = 0.5,
            float hyperparameter = 1 // for loss function if needed
        );

        void build (const std::vector<int>& dimensions); // sample input dimensions

        MX::Matrixf predict (const MX::Matrixf& X);

    };

}