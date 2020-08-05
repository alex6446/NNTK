#include <NN/Matrix.hpp>
#include <NN/Functions.hpp>
#include <iostream>

using namespace NN;
using namespace std;

void constructors () {
    cout << endl << "### CONSTRUCTORS TEST" << endl << endl; 
    Matrixf A;
    cout << "A()\n" << A << endl;
    Matrixf B(2, 3);
    cout << "B(2, 3)\n" << B << endl;
    float* arrC = new float[6] {1, 2, 3, 4, 5, 6};
    Matrixf C(2, 3, arrC);
    cout << "C(2, 3, float*)\n" << C << endl;
    float** arrD = new float*[2] {new float[3] {1, 2, 3}, new float[3] {4, 5, 6}};
    Matrixf D(2, 3, arrD);
    cout << "D(2, 3, float**)\n" << D << endl;
    Matrixf E({1, 2, 3, 4, 5, 6});
    cout << "E(init_list<float>)\n" << E << endl;
    Matrixf F({{1, 2, 3}, {4, 5, 6}});
    cout << "F(init_list<init_list<float>>)\n" << F << endl;
    delete[] arrC;
    delete[] arrD[0];
    delete[] arrD[1];
    delete[] arrD;
}

void assignment () {
    cout << endl << "### ASSIGNMENT TEST" << endl << endl; 
    Matrixf A, B, C({{3, 2, 1}, {2, 1, 4}});
    A = B = C;
    cout << "A = B = C\n" << A << endl;
    A = {{2, 3, 1, 4, 8, 7}};
    cout << "A = init_list<float>\n" << A << endl;
    A = {{2, 3}, {1, 4}, {8, 7}};
    cout << "A = init_list<init_list<float>>\n" << A << endl;
}

void functions () {
    cout << endl << "### FUNCTIONS TEST" << endl << endl; 
    Matrixf A(3, 4);
    cout << "A.randomize(-5, 5)\n" << A.randomize(-2, 2) << endl;
    Matrixf B(2, 6);
    cout << "B.fit(A)\n" << B.fit(A) << endl;
    cout << "A.flatten()\n" << A.flatten() << endl;
    cout << "A.reshape(4, 3)\n" << A.reshape(6, 2) << endl;
    cout << "A.transpose()\n" << A.transpose() << endl;
    cout << "A.apply(ReLU, 0)\n" << A.apply(Activation::ReLU, 0) << endl;
    Matrixi C = Matrixi(3, 4).randomize(-5, 5);
    Matrixi D = Matrixi(4, 2).randomize(-5, 5);
    cout << "C\n" << C << endl;
    cout << "D\n" << D << endl;
    cout << "C.dot(D)\n" << C.dot(D) << endl;
}

void accessors () {
    cout << endl << "### ACCESSORS TEST" << endl << endl; 
    Matrixf A = Matrixf(4, 6).randomize(-1, 1);
    cout << "A\n" << A << endl;
    cout << "A(2, 3)\n" << A(2, 3) << endl;
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
    Matrixi A = Matrixi(2, 3).randomize(1, 10);
    Matrixi B = Matrixi(2, 3).randomize(1, 10);
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
    cout << "A += 3" << endl << (Matrixi(A) += 3) << endl;
    cout << "A -= 3" << endl << (Matrixi(A) -= 3) << endl;
    cout << "A *= 3" << endl << (Matrixi(A) *= 3) << endl;
    cout << "A /= 3" << endl << (Matrixi(A) /= 3) << endl;
    cout << "A\n" << A << endl;
    cout << "B\n" << B << endl;
    cout << "A += B" << endl << (Matrixi(A) += B) << endl;
    cout << "A -= B" << endl << (Matrixi(A) -= B) << endl;
    cout << "A *= B" << endl << (Matrixi(A) *= B) << endl;
    cout << "A /= B" << endl << (Matrixi(A) /= B) << endl;
    cout << "A\n" << A << endl;
    cout << "3 + A" << endl << 3 + A << endl;
    cout << "3 - A" << endl << 3 - A << endl;
    cout << "3 * A" << endl << 3 * A << endl;
    cout << "3 / A" << endl << 3 / A << endl;
    cout << "3 + A == A + 3" << endl << (3 + A == A + 3) << endl;
    cout << "3 + A != (A += 3)" << endl << (3 + A != (Matrixi(A) += 3)) << endl;
}

int main () {

    cout << endl << "START MATRIX TESTS ......................................." << endl << endl; 

    constructors();
    assignment();
    functions();
    accessors();
    operators();

    cout << endl << "FINISH MATRIX TESTS ......................................" << endl << endl;

}