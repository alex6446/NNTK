#pragma once

#include <vector>

#include "NNTK/Layers/Base.hpp"
#include "NNTK/Layers/Dense.hpp"
#include "NNTK/Layers/Conv2D.hpp"
#include "NNTK/Layers/Flatten.hpp"
#include "NNTK/Layers/MaxPooling2D.hpp"
#include "NNTK/Layers/AveragePooling2D.hpp"

namespace NN
{

class Sequential
{
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
    inline const std::vector<Layer::Base*>& layers () { return L; }

    MX::Matrixf predict (const MX::Matrixf& X);
    MX::Matrixf predict (const std::vector<MX::Image>& X);
    std::vector<MX::Image> predict2D (const std::vector<MX::Image>& X);

    inline void print () const { for (auto i : L) i->print(); }
    void save (std::string file) const;
    void load (std::string file);

    friend std::ostream& operator<< (std::ostream& os, const Sequential& l);
    friend std::istream& operator>> (std::istream& is, Sequential& l);

};

} // namespace NN
