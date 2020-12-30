#include "Array.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iterator>
#include <ostream>

void
operators_test()
{
    using namespace NN;
    MX::Array<float> a = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    MX::Array<float> b = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};

    std::cout << "a " << a << std::endl;
    std::cout << "b " << b << std::endl;

    std::cout << -b << std::endl;

    std::cout << std::endl << "Section 1 (a and b)" << std::endl;
    std::cout << a + b << std::endl;
    std::cout << a - b << std::endl;
    std::cout << a * b << std::endl;
    std::cout << a / b << std::endl;

    std::cout << std::endl << "Section 2 (a and number)" << std::endl;
    std::cout << a + 2.f << std::endl;
    std::cout << a - 2.f << std::endl;
    std::cout << a * 2.f << std::endl;
    std::cout << a / 2.f << std::endl;

    std::cout << std::endl << "Section 3 (number and b)" << std::endl;
    std::cout << 2.f + b << std::endl;
    std::cout << 2.f - b << std::endl;
    std::cout << 2.f * b << std::endl;
    std::cout << 2.f / b << std::endl;

    std::cout << "a " << a << std::endl;
    std::cout << "b " << b << std::endl;

    std::cout << std::endl << "Section 4 (a and equal b)" << std::endl;
    std::cout << (a += b) << std::endl;
    std::cout << (a -= b) << std::endl;
    std::cout << (a *= b) << std::endl;
    std::cout << (a /= b) << std::endl;

    std::cout << std::endl << "Section 5 (a and equal number)" << std::endl;
    std::cout << (a += 2.f) << std::endl;
    std::cout << (a -= 2.f) << std::endl;
    std::cout << (a *= 2.f) << std::endl;
    std::cout << (a /= 2.f) << std::endl;

}

void
creation_test()
{
    using namespace NN;

    std::cout << std::endl << "Section 6 (random)" << std::endl;
    MX::Array<float> ind = MX::Random<float>({122}, -1.5, 1.5);
    std::cout << ind << std::endl;
    std::cout << ind.shape() << std::endl;
    std::cout << ind.dimensions() << std::endl;

    std::cout << std::endl << "Section 7 (zeros)" << std::endl;
    MX::Array<int> zs = MX::Zeros<int>({2, 2, 3});
    std::cout << zs << std::endl;
    std::cout << zs.shape() << std::endl;
    std::cout << zs.dimensions() << std::endl;

    std::cout << std::endl << "Section 8 (ones)" << std::endl;
    MX::Array<int> os = MX::Ones<int>({2, 3});
    std::cout << os << std::endl;
    std::cout << os.shape() << std::endl;
    std::cout << os.dimensions() << std::endl;

}

void
dot_product_test()
{
    using namespace NN;

    //MX::Array<int> l = MX::Random<int>({100, 300}, -10, 10);
    //MX::Array<int> r = MX::Random<int>({300, 500}, -10, 10);

    std::cout << std::endl << "Section 9 (dot)" << std::endl;
    MX::Array<int> l = MX::Random<int>({2, 3}, 0, 5);
    MX::Array<int> r = MX::Random<int>({3, 1}, 0, 5);

    std::cout << l << std::endl << r << std::endl;
    std::cout << MX::Dot(l, r) << std::endl;
    std::cout << MX::Dot(l, r).shape() << std::endl;
    std::cout << MX::Dot(l, r).dimensions() << std::endl;
}

void
save_load_test()
{
    using namespace NN;

    std::cout << std::endl << "Section 10 (save load)" << std::endl;
    MX::Array<float> a = MX::Random<float>({3, 2, 3}, -2, 2);

    std::cout << a << std::endl;
    a[2][1].save("save_test");

    MX::Array<float> b = MX::Zeros<float>({3, 3});
    b[0].load("save_test");
    std::cout << b << std::endl;

}

void
conv_test()
{
    using namespace NN;

    std::cout << std::endl << "Section 11 (convolve)" << std::endl;
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
    std::cout << MX::Convolve(E, F, 2, 3) << std::endl;
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

    std::cout << std::endl << "Section 12 (sum)" << std::endl;
    MX::Array<int> s = {{{1,2,3}, {4,5,6}}, {{7,8,9}, {10,11,12}}};
    std::cout << s << std::endl;
    std::cout << MX::Sum(s, 0) << std::endl;
    std::cout << MX::Sum(s, 1) << std::endl;
    std::cout << MX::Sum(s, 2) << std::endl;
    std::cout << MX::Sum(s) << std::endl;
    std::cout << MX::Sum(s, MX::Array<int>({{7,8,9}, {10,11,12}})) << std::endl;
}

void
access_test()
{
    using namespace NN;
    using std::cout;
    using std::endl;
    std::cout << std::endl << "Section 13 (value access)" << std::endl;
    MX::Array<float> a = MX::Random<float>({2, 2, 3});
    cout << a << endl;
    cout << a(0, 1) << endl;

    cout << a.shape() << endl;
    cout << a.dimensions() << endl;

    std::cout << std::endl << "Section 14 (subarray access)" << std::endl;
    cout << a[0] << endl;
    cout << a[1][0] << endl;
    cout << a[1][0].shape() << endl;
    cout << a[1](0, 2) << endl;
    cout << (a[1] = {{1, 2, 3}, {4, 5, 6}}) << endl;
    cout << a << endl;
}

void
iterator_test()
{
    using namespace NN;

    std::cout << std::endl << "Section 15 (iterators)" << std::endl;
    MX::Array<int> a = MX::Random<int>({2, 2}, 0, 5);
    std::cout << a << std::endl;
    for (auto &i : a)
        std::cout << (i *= i) << " ";
    std::cout << std::endl;
    std::cout << a << std::endl;
}

int
main()
{
    using namespace std;
    clock_t begin = clock();
    srand(time(NULL));

    operators_test();
    creation_test();
    dot_product_test();
    save_load_test();
    conv_test();
    sum_test();
    access_test();
    iterator_test();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "TIME: " << elapsed_secs << "s" << endl;
}

