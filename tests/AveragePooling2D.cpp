#include <NN/Toolkit.hpp>

int main () {
    using namespace std;
    using namespace NN;
    std::vector<MX::Image> X = { {
    {
        { 1, 3, -1, 4 }, 
        { 3, -4, -4, 5 }, 
        { -6, -2, 5, -3 }, 
        { -4, -4, 1, -1 }, 
        { 0, -1, -5, 1 }, 
        { 3, 3, 2, -4 }, 
        { 5, -1, 1, 2 }, 
        { -4, -2, 3, 2 }
    }, {
        { 2, 4, -5, -6 }, 
        { 0, -4, -4, -3 }, 
        { 1, -6, 2, -3 }, 
        { -4, -6, 2, 1 }, 
        { 4, 0, -3, -4 }, 
        { 5, -1, 3, -2 }, 
        { -5, -4, 3, -2 }, 
        { -2, -3, 3, 3 }
    }
    } };
    std::vector<MX::Image> Y = { {
    {
        { 1, 3, -1, 4, 3 }, 
        { -4, -4, 5, -6, -2 }, 
        { 5, -3, -4, -4, 1 }, 
        { -1, 0, -1, -5, 1 }, 
        { 3, 3, 2, -4, 5 }, 
        { -1, 1, 2, -4, -2 }, 
        { 3, 2, 2, 4, -5 }, 
        { -6, 0, -4, -4, -3 }, 
        { 1, -6, 2, -3, -4 }
    }, {
        { -6, 2, 1, 4, 0 }, 
        { -3, -4, 5, -1, 3 }, 
        { -2, -5, -4, 3, -2 }, 
        { -2, -3, 3, 3, 3 }, 
        { -3, -1, 0, 4, 5 }, 
        { -2, 2, -3, 0, -2 }, 
        { -2, -6, -4, -4, -5 }, 
        { 5, 2, -5, -5, -4 }, 
        { -6, 5, -3, 5, -3 }
    }
    } };
    // X.push_back(X[0]);
    // Y.push_back(Y[0]);

    cout << "X" << endl;
    for (auto& i : X)
        for (auto& j : i)
            cout << j;

    cout << "Y" << endl;
    for (auto& i : Y)
        for (auto& j : i)
            cout << j;

    Layer::AveragePooling2D mp(2, 1, 1);
    mp.bind({ (int)X[0].size(), X[0][0].rows(), X[0][0].cols()});
    mp.forwardProp(&X);
    std::vector<MX::Image> A = *((std::vector<MX::Image>*)mp.getA());
    cout << "A" << endl;
    for (auto& i : A)
        for (auto& j : i)
            cout << j;
    mp.backProp(new std::vector<MX::Image>(Y));
    std::vector<MX::Image> dX = *((std::vector<MX::Image>*)mp.getGradient());
    cout << "dX" << endl;
    for (auto& i : dX)
        for (auto& j : i)
            cout << j;
    // cout << MX::Matrixi(9, 5).randomize(-6, 6) << endl;
    // cout << MX::Matrixi(9, 5).randomize(-6, 6) << endl;

}