#include "Array.hpp"
#include "ArrayBase.hpp"
#include "Matrix.hpp"
#include <ostream>

using namespace NN;
int
main()
{
    std::ofstream fout("Matrix.mx");
    MX::Matrixf A = MX::Matrixf(10000, 1000).randomize(0, 1);
    fout << A;
    fout.close();
    //std::cout << "Write " << A << std::endl;
    std::ifstream fin("Matrix.mx");
    MX::Matrixf B;
    if (fin.is_open())
        fin >> B;
    //std::cout << "Read " << B << std::endl;
    
    MX::Array<float> a = MX::Random<float>({10000, 1000});
    a.save("Array.mxa");
    MX::Array<float> b;
    b.load("Array.mxa");
}
