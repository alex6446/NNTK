#include <NN/Matrix.hpp>
#include <NN/Functions.hpp>
#include <iostream>

using namespace NN;
using namespace std;

void constructors () {
    cout << endl << "### CONSTRUCTORS TEST" << endl << endl; 
    MX::Matrixf A;
    cout << "A()\n" << A << endl;
    MX::Matrixf B(2, 3);
    cout << "B(2, 3)\n" << B << endl;
    float* arrC = new float[6] {1, 2, 3, 4, 5, 6};
    MX::Matrixf C(2, 3, arrC);
    cout << "C(2, 3, float*)\n" << C << endl;
    float** arrD = new float*[2] {new float[3] {1, 2, 3}, new float[3] {4, 5, 6}};
    MX::Matrixf D(2, 3, arrD);
    cout << "D(2, 3, float**)\n" << D << endl;
    MX::Matrixf E({1, 2, 3, 4, 5, 6});
    cout << "E(init_list<float>)\n" << E << endl;
    MX::Matrixf F({{1, 2, 3}, {4, 5, 6}});
    cout << "F(init_list<init_list<float>>)\n" << F << endl;
    delete[] arrC;
    delete[] arrD[0];
    delete[] arrD[1];
    delete[] arrD;
}

void assignment () {
    cout << endl << "### ASSIGNMENT TEST" << endl << endl; 
    MX::Matrixf A, B, C({{3, 2, 1}, {2, 1, 4}});
    A = B = C;
    cout << "A = B = C\n" << A << endl;
    A = {{2, 3, 1, 4, 8, 7}};
    cout << "A = init_list<float>\n" << A << endl;
    A = {{2, 3}, {1, 4}, {8, 7}};
    cout << "A = init_list<init_list<float>>\n" << A << endl;
}

void functions () {
    cout << endl << "### FUNCTIONS TEST" << endl << endl; 
    MX::Matrixf A(3, 4);
    cout << "A.randomize(-5, 5)\n" << A.randomize(-2, 2) << endl;
    MX::Matrixf B(2, 6);
    cout << "B.fit(A)\n" << B.fit(A) << endl;
    cout << "A.flatten()\n" << A.flatten() << endl;
    cout << "A.flatten(1)\n" << A.flatten(1) << endl;
    cout << "A.reshape(4, 3)\n" << A.reshape(6, 2) << endl;
    cout << "A.transpose()\n" << A.transpose() << endl;
    cout << "A.rotate180()\n" << A.rotate180() << endl;
    cout << "A.apply(ReLU, 0)\n" << A.apply(Activation::ReLU, 0) << endl;
    MX::Matrixi C = MX::Matrixi(3, 4).randomize(-5, 5);
    MX::Matrixi D = MX::Matrixi(4, 2).randomize(-5, 5);
    cout << "C\n" << C << endl;
    cout << "D\n" << D << endl;
    cout << "Dot(C, D)\n" << MX::Dot(C, D) << endl;
    cout << "Sum(D)\n" << MX::Sum(D) << endl;
    cout << "Sum(D, 0)\n" << MX::Sum(D, 0) << endl;
    cout << "Sum(D, 1)\n" << MX::Sum(D, 1) << endl;
    MX::Matrixi E = {
        {2, 3, 7, 4, 6, 2, 9, 1},
        {6, 6, 9, 8, 7, 4, 3, 1},
        {3, 4, 8, 3, 8, 9, 7, 1},
        {7, 8, 3, 6, 6, 3, 4, 1},
        {4, 2, 1, 8, 3, 4, 6, 1},
        {3, 2, 4, 1, 9, 8, 3, 1},
        {0, 1, 3, 9, 2, 1, 4, 1}
    };
    MX::Matrixi F = {
        {3, 4, 4},
        {1, 0, 2},
        {-1, 0, 3}
    };
    cout << "Convolve(E, F, 2, 3)\n" << MX::Convolve(E, F, 2, 3) << endl;
    /* Answer: { 
        { 6, 9, 21, -1 }, 
        { 51, 106, 77, 3 }, 
        { 22, 72, 74, 3 } 
    } */
}

void accessors () {
    cout << endl << "### ACCESSORS TEST" << endl << endl; 
    MX::Matrixf A = MX::Matrixf(4, 6).randomize(-1, 1);
    cout << "A\n" << A << endl;
    cout << "A(2, 3)\n" << A(2, 3) << endl;
    cout << "A.get(2, 8)\n" << A.get(2, 8) << endl;
    cout << "A.rows()\n" << A.rows() << endl;
    cout << "A.cols()\n" << A.cols() << endl;
    cout << "A.elements()\n" << A.elements() << endl;
    float** arr = A.data();
    cout << "A.data() manually displayed\n";
    cout << "[ [ ";
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) 
            cout << A(i, j) << " ";
        if (i != A.rows()-1)
            cout << "] " << std::endl << "  [ ";
    }
    cout << "] ] " << endl;
}

void operators () {
    cout << endl << "### OPERATORS TEST" << endl << endl; 
    MX::Matrixi A = MX::Matrixi(2, 3).randomize(1, 10);
    MX::Matrixi B = MX::Matrixi(2, 3).randomize(1, 10);
    cout << "A\n" << A << endl;
    cout << "A + 3" << endl << A + 3 << endl;
    cout << "A - 3" << endl << A - 3 << endl;
    cout << "A * 3" << endl << A * 3 << endl;
    cout << "A / 3" << endl << A / 3 << endl;
    cout << "A\n" << A << endl;
    cout << "B\n" << B << endl;
    cout << "A + B" << endl << A + B << endl;
    cout << "A - B" << endl << A - B << endl;
    cout << "A * B" << endl << A * B << endl;
    cout << "A / B" << endl << A / B << endl;
    cout << "-A" << endl << -A << endl;
    cout << "A\n" << A << endl;
    cout << "A += 3" << endl << (MX::Matrixi(A) += 3) << endl;
    cout << "A -= 3" << endl << (MX::Matrixi(A) -= 3) << endl;
    cout << "A *= 3" << endl << (MX::Matrixi(A) *= 3) << endl;
    cout << "A /= 3" << endl << (MX::Matrixi(A) /= 3) << endl;
    cout << "A\n" << A << endl;
    cout << "B\n" << B << endl;
    cout << "A += B" << endl << (MX::Matrixi(A) += B) << endl;
    cout << "A -= B" << endl << (MX::Matrixi(A) -= B) << endl;
    cout << "A *= B" << endl << (MX::Matrixi(A) *= B) << endl;
    cout << "A /= B" << endl << (MX::Matrixi(A) /= B) << endl;
    cout << "A\n" << A << endl;
    cout << "3 + A" << endl << 3 + A << endl;
    cout << "3 - A" << endl << 3 - A << endl;
    cout << "3 * A" << endl << 3 * A << endl;
    cout << "3 / A" << endl << 3 / A << endl;
    cout << "3 + A == A + 3" << endl << (3 + A == A + 3) << endl;
    cout << "3 + A != (A += 3)" << endl << (3 + A != (MX::Matrixi(A) += 3)) << endl;
}

int main () {

    cout << endl << "START MATRIX TESTS ......................................." << endl << endl; 

    constructors();
    assignment();
    accessors();
    operators();
    functions();

    cout << endl << "FINISH MATRIX TESTS ......................................" << endl << endl;

}