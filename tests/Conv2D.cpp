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
    std::vector<MX::Image> Y = { {
        { 
            { 5, 2, 2 }, 
            { 5, 1, 2 } 
        }, { 
            { 4, 2, 1 }, 
            { 2, 2, 4 } 
        }
    } };
    // W = {
    //     {
    //         { 
    //             { 3, 3, 4 }, 
    //             { 5, 1, 3 }, 
    //             { 2, 3, 5 } 
    //         }, { 
    //             { 2, 2, 2 }, 
    //             { 5, 4, 4 }, 
    //             { 1, 5, 3 } 
    //         }
    //     }, {
    //         { 
    //             { 1, 1, 2 }, 
    //             { 2, 1, 3 }, 
    //             { 5, 1, 4 } 
    //         }, { 
    //             { 2, 1, 1 }, 
    //             { 1, 5, 2 }, 
    //             { 4, 4, 2 } 
    //         }
    //     }
    // };
    Layer::Conv2D lc(2, 3, 1, 2, Activation::None, true, -2, 2);
    lc.bind({ (int)X[0].size() });
    lc.forwardProp(&X);
    std::vector<MX::Image> A = *((std::vector<MX::Image>*)lc.getA());
    std::cout << "A" << std::endl;
    for (auto i : A)
        for (auto j : i)
            cout << "{ " << j  << " }" << endl;
    lc.backProp(new std::vector<MX::Image>(Y));
    std::vector<MX::Image> dX = *((std::vector<MX::Image>*)lc.getGradient());
    std::cout << "dX" << std::endl;
    for (auto i : dX)
        for (auto j : i)
            std::cout << "{ " << j  << " }" << std::endl;
    lc.update(2.f);
    //cout << MX::Matrixi(3, 3).randomize(1, 6) << endl;
    //cout << MX::Matrixi(3, 3).randomize(1, 6) << endl;

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