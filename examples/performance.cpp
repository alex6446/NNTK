
#include "Array.hpp"
#include "ArrayBase.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iterator>
#include <ostream>

NN::MX::Array<float> TEMP_FLOAT;
NN::MX::Array<int> TEMP_INT;
NN::MX::Array<NN::MX::Array<int>::size_type> TEMP_SIZE;

void
operators_test()
{
    using namespace NN;
    MX::Array<float> a = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    MX::Array<float> b = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};

    TEMP_FLOAT = a;
    TEMP_FLOAT = b;

    TEMP_FLOAT = -b;

    //std::cout << std::endl << "Section 1 (a and b)" << std::endl;
    TEMP_FLOAT = a + b;
    TEMP_FLOAT = a - b;
    TEMP_FLOAT = a * b;
    TEMP_FLOAT = a / b;

    //std::cout << std::endl << "Section 2 (a and number)" << std::endl;
    TEMP_FLOAT = a + 2.f;
    TEMP_FLOAT = a - 2.f;
    TEMP_FLOAT = a * 2.f;
    TEMP_FLOAT = a / 2.f;

    //std::cout << std::endl << "Section 3 (number and b)" << std::endl;
    TEMP_FLOAT = 2.f + b;
    TEMP_FLOAT = 2.f - b;
    TEMP_FLOAT = 2.f * b;
    TEMP_FLOAT = 2.f / b;

    TEMP_FLOAT = a;
    TEMP_FLOAT = b;

    //std::cout << std::endl << "Section 4 (a and equal b)" << std::endl;
    TEMP_FLOAT = a += b;
    TEMP_FLOAT = a -= b;
    TEMP_FLOAT = a *= b;
    TEMP_FLOAT = a /= b;

    //std::cout << std::endl << "Section 5 (a and equal number)" << std::endl;
    TEMP_FLOAT = a += 2.f;
    TEMP_FLOAT = a -= 2.f;
    TEMP_FLOAT = a *= 2.f;
    TEMP_FLOAT = a /= 2.f;

}

void
creation_test()
{
    using namespace NN;

    //std::cout << std::endl << "Section 6 (random)" << std::endl;
    MX::Array<float> ind = MX::Random<float>({122000}, -1.5, 1.5);
    //TEMP_FLOAT = ind;
    TEMP_SIZE = ind.shape();
    TEMP_SIZE = ind.dimensions();

    //std::cout << std::endl << "Section 7 (sequence)" << std::endl;
    MX::Array<float> seq = MX::Sequence<float>({15}, 3, -2);
    TEMP_FLOAT = seq;
    TEMP_SIZE = seq.shape();
    TEMP_SIZE = seq.dimensions();

    //std::cout << std::endl << "Section 8 (zeros)" << std::endl;
    MX::Array<int> zs = MX::Fill<int>({2, 2, 3}, 0);
    TEMP_INT = zs;
    TEMP_SIZE = zs.shape();
    TEMP_SIZE = zs.dimensions();

    //std::cout << std::endl << "Section 9 (ones)" << std::endl;
    MX::Array<int> os = MX::Fill<int>({2, 3}, 1);
    TEMP_INT = os;
    TEMP_SIZE = os.shape();
    TEMP_SIZE = os.dimensions();

    //std::cout << std::endl << "Section 10 (empty)" << std::endl;
    MX::Array<int> mp = MX::Empty<int>(os.shape());
    TEMP_INT = mp;
    TEMP_SIZE = mp.shape();
    TEMP_SIZE = mp.dimensions();

}

void
dot_product_test()
{
    using namespace NN;

    //MX::Array<int> l = MX::Random<int>({100, 300}, -10, 10);
    //MX::Array<int> r = MX::Random<int>({300, 500}, -10, 10);

    //std::cout << std::endl << "Section 11 (dot)" << std::endl;
    MX::Array<int> l = MX::Random<int>({2, 3}, 0, 5);
    MX::Array<int> r = MX::Random<int>({3, 1}, 0, 5);

    TEMP_INT = l; TEMP_INT = r;
    TEMP_INT = MX::Dot(l, r);
    TEMP_SIZE = MX::Dot(l, r).shape();
    TEMP_SIZE = MX::Dot(l, r).dimensions();
}

void
save_load_test()
{
    using namespace NN;

    //std::cout << std::endl << "Section 12 (save load)" << std::endl;
    MX::Array<float> a = MX::Random<float>({3, 2, 3}, -2, 2);

    TEMP_FLOAT = a;
    a[2][1].save("save_test");

    MX::Array<float> b = MX::Fill<float>({3, 3}, 0);
    b[0].load("save_test");
    TEMP_FLOAT = b;

}

void
conv_test()
{
    using namespace NN;

    //std::cout << std::endl << "Section 13 (convolve)" << std::endl;
    MX::Array<int> E = {
        {2, 3, 7, 4, 6, 2, 9, 1},
        {6, 6, 9, 8, 7, 4, 3, 1},
        {3, 4, 8, 3, 8, 9, 7, 1},
        {7, 8, 3, 6, 6, 3, 4, 1},
        {4, 2, 1, 8, 3, 4, 6, 1},
        {3, 2, 4, 1, 9, 8, 3, 1},
        {0, 1, 3, 9, 2, 1, 4, 1}
    };
    MX::Array<int> F = {
        {3, 4, 4},
        {1, 0, 2},
        {-1, 0, 3}
    };
    TEMP_INT = MX::Convolve(E, F, 2, 3);
    /* Answer: {
        { 6, 9, 21, -1 },
        { 51, 106, 77, 3 },
        { 22, 72, 74, 3 }
    } */
}

void
sum_test()
{
    using namespace NN;

    //std::cout << std::endl << "Section 14 (sum)" << std::endl;
    MX::Array<int> s = {{{1,2,3}, {4,5,6}}, {{7,8,9}, {10,11,12}}};
    TEMP_INT = s;
    TEMP_INT = MX::Sum(s, 0);
    TEMP_INT = MX::Sum(s, 1);
    TEMP_INT = MX::Sum(s, 2);
    int t = MX::Sum(s);
    TEMP_INT = MX::Sum(s, MX::Array<int>({{7,8,9}, {10,11,12}}));
}

void
access_test()
{
    using namespace NN;
    using std::cout;
    using std::endl;
    //std::cout << std::endl << "Section 15 (value access)" << std::endl;
    MX::Array<float> a = MX::Random<float>({2, 2, 3});
    TEMP_FLOAT = a;
    float t = a(0, 1);

    TEMP_SIZE = a.shape();
    TEMP_SIZE = a.dimensions();

    //std::cout << std::endl << "Section 16 (subarray access)" << std::endl;
    TEMP_FLOAT = a[0];
    TEMP_FLOAT = a[1][0];
    TEMP_SIZE = a[1][0].shape();
    t = a[1](0, 2);
    TEMP_FLOAT = (a[1] = {{1, 2, 3}, {4, 5, 6}});
    TEMP_FLOAT = a;
}

void
iterator_test()
{
    using namespace NN;

    //std::cout << std::endl << "Section 17 (iterators)" << std::endl;
    MX::Array<int> a = MX::Random<int>({2, 2}, 0, 5);
    TEMP_INT = a;
    for (auto &i : a)
        int t = (i *= i);
    TEMP_INT = a;
}

NN::MX::Array<int>
f(NN::MX::Array<int> a)
{
    return a;
}

void
move_test()
{
//TODO: ...
}

void
transpose_test()
{
    using namespace NN;

    //std::cout << std::endl << "Section 18 (transpose)" << std::endl;
    //MX::Array<int> a = MX::Random<int>({2, 1, 3}, 0, 5);
    MX::Array<int> a =
        {{{7, 4, 0, 5},
          {5, 7, 7, 8},
          {1, 7, 3, 2}},

         {{3, 8, 8, 4},
          {3, 9, 2, 1},
          {5, 5, 5, 9}}};
    TEMP_INT = a;
    TEMP_SIZE = a.shape();
    TEMP_INT = MX::Transpose(a, {1, 2, 0});
    TEMP_INT = MX::Transpose(a);
}

int
main()
{
    using namespace std;
    clock_t begin = clock();
    srand(time(NULL));

    for (int i = 0; i < 10000; ++i) {
        operators_test();
        creation_test();
        dot_product_test();
        save_load_test();
        conv_test();
        sum_test();
        access_test();
        iterator_test();
        transpose_test();
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "TIME: " << elapsed_secs << "s" << endl;
}

