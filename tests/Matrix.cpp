#include <NN/Matrix.hpp>
#include <NN/Functions.hpp>
#include <iostream>
#include <fstream>

using namespace NN;
using namespace std;

void constructors () {
    cout << endl << "### CONSTRUCTORS TEST" << endl << endl; 
    MX::Matrixf A;
    cout << "A() " << A << endl;
    MX::Matrixf B(2, 3);
    cout << "B(2, 3) " << B << endl;
    float* arrC = new float[6] {1, 2, 3, 4, 5, 6};
    MX::Matrixf C(2, 3, arrC);
    cout << "C(2, 3, float*) " << C << endl;
    float** arrD = new float*[2] {new float[3] {1, 2, 3}, new float[3] {4, 5, 6}};
    MX::Matrixf D(2, 3, arrD);
    cout << "D(2, 3, float**) " << D << endl;
    MX::Matrixf E({1, 2, 3, 4, 5, 6});
    cout << "E(init_list<float>) " << E << endl;
    MX::Matrixf F({{1, 2, 3}, {4, 5, 6}});
    cout << "F(init_list<init_list<float>>) " << F << endl;
    delete[] arrC;
    delete[] arrD[0];
    delete[] arrD[1];
    delete[] arrD;
}

void assignment () {
    cout << endl << "### ASSIGNMENT TEST" << endl << endl; 
    MX::Matrixf A, B, C({{3, 2, 1}, {2, 1, 4}});
    A = B = C;
    cout << "A = B = C " << A << endl;
    A = {{2, 3, 1, 4, 8, 7}};
    cout << "A = init_list<float> " << A << endl;
    A = {{2, 3}, {1, 4}, {8, 7}};
    cout << "A = init_list<init_list<float>> " << A << endl;
}

void functions () {
    cout << endl << "### FUNCTIONS TEST" << endl << endl; 
    MX::Matrixf A(3, 4);
    cout << "A.randomize(-5, 5) " << A.randomize(-2, 2) << endl;
    MX::Matrixf B(2, 6);
    cout << "B.fit(A) " << B.fit(A) << endl;
    cout << "A.flatten() " << A.flatten() << endl;
    cout << "A.flatten(1) " << A.flatten(1) << endl;
    cout << "A.reshape(4, 3) " << A.reshape(6, 2) << endl;
    cout << "A.transpose() " << A.transpose() << endl;
    cout << "A.rotate180() " << A.rotate180() << endl;
    cout << "A.apply(ReLU, 0) " << A.apply(Activation::ReLU, 0) << endl;
    MX::Matrixi C = MX::Matrixi(3, 4).randomize(-5, 5);
    MX::Matrixi D = MX::Matrixi(4, 2).randomize(-5, 5);
    cout << "C " << C << endl;
    cout << "D " << D << endl;
    cout << "Dot(C, D) " << MX::Dot(C, D) << endl;
    cout << "Sum(D) " << MX::Sum(D) << endl;
    cout << "Sum(D, 0) " << MX::Sum(D, 0) << endl;
    cout << "Sum(D, 1) " << MX::Sum(D, 1) << endl;
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
    cout << "Convolve(E, F, 2, 3) " << MX::Convolve(E, F, 2, 3) << endl;
    /* Answer: { 
        { 6, 9, 21, -1 }, 
        { 51, 106, 77, 3 }, 
        { 22, 72, 74, 3 } 
    } */
}

void accessors () {
    cout << endl << "### ACCESSORS TEST" << endl << endl; 
    MX::Matrixf A = MX::Matrixf(4, 6).randomize(-1, 1);
    cout << "A" << A << endl;
    cout << "A(2, 3) " << A(2, 3) << endl;
    cout << "A.get(2, 8) " << A.get(2, 8) << endl;
    A.add(2, 3, 1);
    cout << "A.add(2, 3, 1) " << A << endl;
    cout << "A.rows() " << A.rows() << endl;
    cout << "A.cols() " << A.cols() << endl;
    cout << "A.elements() " << A.elements() << endl;
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
    cout << "A " << A << endl;
    cout << "A + 3 " << A + 3 << endl;
    cout << "A - 3 " << A - 3 << endl;
    cout << "A * 3 " << A * 3 << endl;
    cout << "A / 3 " << A / 3 << endl;
    cout << "A " << A << endl;
    cout << "B " << B << endl;
    cout << "A + B " << A + B << endl;
    cout << "A - B " << A - B << endl;
    cout << "A * B " << A * B << endl;
    cout << "A / B " << A / B << endl;
    cout << "-A" << endl << -A << endl;
    cout << "A " << A << endl;
    cout << "A += 3 " << (MX::Matrixi(A) += 3) << endl;
    cout << "A -= 3 " << (MX::Matrixi(A) -= 3) << endl;
    cout << "A *= 3 " << (MX::Matrixi(A) *= 3) << endl;
    cout << "A /= 3 " << (MX::Matrixi(A) /= 3) << endl;
    cout << "A " << A << endl;
    cout << "B " << B << endl;
    cout << "A += B " << (MX::Matrixi(A) += B) << endl;
    cout << "A -= B " << (MX::Matrixi(A) -= B) << endl;
    cout << "A *= B " << (MX::Matrixi(A) *= B) << endl;
    cout << "A /= B " << (MX::Matrixi(A) /= B) << endl;
    cout << "A " << A << endl;
    cout << "3 + A " << 3 + A << endl;
    cout << "3 - A " << 3 - A << endl;
    cout << "3 * A " << 3 * A << endl;
    cout << "3 / A " << 3 / A << endl;
    cout << "3 + A == A + 3 " << (3 + A == A + 3) << endl;
    cout << "3 + A != (A += 3) " << (3 + A != (MX::Matrixi(A) += 3)) << endl;
}

void filewr () {
    cout << endl << "### FILE WRITE READ TEST" << endl << endl; 
    ofstream fout("tests/files/Matrix.mx");
    MX::Matrixf A = MX::Matrixf(3, 6).randomize(0, 1);
    fout << A;
    fout.close();
    cout << "Write " << A << endl;
    ifstream fin("tests/files/Matrix.mx");
    MX::Matrixf B;
    if (fin.is_open())
    fin >> B;
    cout << "Read " << B << endl;
}

void filewrFilter() {
    cout << endl << "### FILE WRITE READ FILTER TEST" << endl << endl; 
    std::vector<MX::Image> A(2);
    for (int i = 0; i < A.size(); ++i) {
        A[i].push_back(MX::Matrixf(6, 6).randomize(0, 1));             
        A[i].push_back(MX::Matrixf(6, 6).randomize(0, 1));             
        A[i].push_back(MX::Matrixf(6, 6).randomize(0, 1));             
    } 
    ofstream fout("tests/files/Filter.mx");
    fout << A;
    fout.close();
    cout << "Write " << A << endl;
    std::vector<MX::Image> B;
    ifstream fin("tests/files/Filter.mx");
    if (fin.is_open())
    fin >> B;
    cout << "Read " << B << endl;
}

int main () {

    cout << endl << "START MATRIX TESTS ......................................." << endl << endl; 

    constructors();
    assignment();
    accessors();
    operators();
    functions();
    filewr();
    filewrFilter();

    cout << endl << "FINISH MATRIX TESTS ......................................" << endl << endl;

}