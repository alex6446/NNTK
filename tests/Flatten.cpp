#include <NN/Toolkit.hpp>

#include <iostream>

int main () {
    using namespace std;
    using namespace NN;
    //srand(time(NULL));
    std::vector<MX::Image> X = { {
        { 
            { 3, 5, 3, 2, 2 }, 
            { 2, 5, 2, 1, 3 }, 
            { 5, 1, 3, 4, 1 }, 
            { 4, 2, 4, 1, 4 } 
        }, { 
            { 3, 1, 4, 5, 3 }, 
            { 1, 2, 4, 4, 5 }, 
            { 4, 4, 5, 3, 1 }, 
            { 3, 4, 2, 1, 3 } 
        } 
    } };
    MX::Matrixf Y = MX::Matrixf({ 
        { 4, 3, 4, 2, 5, 
          3, 1, 4, 1, 3, 
          2, 3, 3, 3, 3, 
          5, 3, 5, 4, 2, 
          
          5, 2, 5, 4, 1, 
          4, 2, 2, 3, 2, 
          1, 5, 2, 2, 4, 
          5, 3, 5, 5, 4 } 
    }).transpose();

    Layer::Flatten lf(Activation::None, true);
    lf.bind({ (int)X[0].size(), X[0][0].rows(), X[0][0].cols() });
    lf.forwardProp(&X);
    MX::Matrixf A = *(MX::Matrixf*)lf.getA();
    std::cout << "A" << std::endl;
        cout << A << endl;
    lf.backProp(new MX::Matrixf(Y));
    std::vector<MX::Image> dX = *((std::vector<MX::Image>*)lf.getGradient());
    std::cout << "dX" << std::endl;
    for (auto i : dX)
        for (auto j : i)
            std::cout << "{ " << j  << " }" << std::endl;
    lf.update(2.f);
    //cout << MX::Matrixi(1, 40).randomize(1, 6) << endl;
    lf.print();

}


// std::cout << "W" << std::endl;
// for (auto i : W)
//     for (auto j : i)
//         std::cout << "{ " << j  << " }" << std::endl;
// std::cout << "dZ" << std::endl;
// for (auto i : dZ)
//     for (auto j : i)
//         std::cout << "{ " << j  << " }" << std::endl;
// std::cout << "X" << std::endl;
// for (auto i : *(this->X))
//     for (auto j : i)
//         std::cout << "{ " << j  << " }" << std::endl;
// std::cout << "dW" << std::endl;
// for (auto i : dW)
//     for (auto j : i)
//         std::cout << "{ " << j  << " }" << std::endl;
// std::cout << "db" << std::endl;
//     std::cout << db << std::endl;