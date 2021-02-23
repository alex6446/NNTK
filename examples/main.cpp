#include "NN/MX/Array.hpp"
#include <ctime>
#include <iostream>


#ifndef _WIN32
#define SECTION_NUMBER __COUNTER__
#else
int COUNTER = 0;
#define SECTION_NUMBER COUNTER++
#endif


#ifndef NO_OUTPUT

#define TESTING_SECTION(x) std::cout << std::endl << "SECTION " << SECTION_NUMBER <<  " (" <<  x << ")" << std::endl;
#define TEST_EXP(x) std::cout << #x << "  -->  " << x << std::endl;

#else

#define TESTING_SECTION(x)
#define TEST_EXP(x) x;

#endif

void
operators_test()
{
    using namespace NN;
    MX::Array<float> a = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    MX::Array<float> b = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};

    TEST_EXP(a)
    TEST_EXP(b)

    TEST_EXP(-b)

    TESTING_SECTION("a and b")
    TEST_EXP(a + b)
    TEST_EXP(a - b)
    TEST_EXP(a * b)
    TEST_EXP(a / b)

    TESTING_SECTION("a and number")
    TEST_EXP(a + 2.f)
    TEST_EXP(a - 2.f)
    TEST_EXP(a * 2.f)
    TEST_EXP(a / 2.f)

    TESTING_SECTION("number and b")
    TEST_EXP(2.f + b)
    TEST_EXP(2.f - b)
    TEST_EXP(2.f * b)
    TEST_EXP(2.f / b)

    TEST_EXP(a)
    TEST_EXP(b)

    TESTING_SECTION("a and equal b")
    TEST_EXP((MX::Array<float>(a) += b))
    TEST_EXP((MX::Array<float>(a) -= b))
    TEST_EXP((MX::Array<float>(a) *= b))
    TEST_EXP((MX::Array<float>(a) /= b))

    TESTING_SECTION("a and equal number")
    TEST_EXP((MX::Array<float>(a) += 2.f))
    TEST_EXP((MX::Array<float>(a) -= 2.f))
    TEST_EXP((MX::Array<float>(a) *= 2.f))
    TEST_EXP((MX::Array<float>(a) /= 2.f))

}

void
creation_test()
{
    using namespace NN;

    TESTING_SECTION("random")
    MX::Array<float> ind;
    TEST_EXP((ind = MX::Random<float>({12}, -1.5, 1.5)))
    TEST_EXP(ind.shape())
    TEST_EXP(ind.dimensions())

    TESTING_SECTION("sequence")
    MX::Array<float> seq;
    TEST_EXP((seq = MX::Sequence<float>({15}, 3, -2)))
    TEST_EXP(seq.shape())
    TEST_EXP(seq.dimensions())

    TESTING_SECTION("zeros")
    MX::Array<int> zs;
    TEST_EXP((zs = MX::Full<int>({2, 2, 3}, 0)))
    TEST_EXP(zs.shape())
    TEST_EXP(zs.dimensions())

    TESTING_SECTION("ones")
    MX::Array<int> os;
    TEST_EXP((os = MX::Full<int>({2, 3}, 1)))
    TEST_EXP(os.shape())
    TEST_EXP(os.dimensions())

    TESTING_SECTION("empty")
    MX::Array<int> mp;
    TEST_EXP((mp = MX::Empty<int>(os.shape())))
    TEST_EXP(mp.shape())
    TEST_EXP(mp.dimensions())
}

void
dot_product_test()
{
    using namespace NN;

    //MX::Array<int> l = MX::Random<int>({100, 300}, -10, 10);
    //MX::Array<int> r = MX::Random<int>({300, 500}, -10, 10);

    TESTING_SECTION("dot")
    MX::Array<int> l;
    MX::Array<int> r;

    TEST_EXP((l = MX::Random<int>({2, 3}, 0, 5)))
    TEST_EXP((r = MX::Random<int>({3, 1}, 0, 5)))
    TEST_EXP(MX::Dot(l, r));
    TEST_EXP(MX::Dot(l, r).shape());
    TEST_EXP(MX::Dot(l, r).dimensions());
}

void
save_load_test()
{
    using namespace NN;

    TESTING_SECTION("save load")

    MX::Array<float> a;
    TEST_EXP((a = MX::Random<float>({3, 2, 3}, -2, 2)))
    TEST_EXP(a[2][1].save_to_file("save_test." + MX::Array<float>::FileExt))

    MX::Array<float> b;
    TEST_EXP((b = MX::Full<float>({3, 3}, 0)))
    TEST_EXP(b[0].load_from_file("save_test." + MX::Array<float>::FileExt))
    TEST_EXP(b)
}

void
save_load_pack_test()
{
    using namespace NN;

    TESTING_SECTION("save load pack")
    MX::Array<float> a;
    MX::Array<float> b;
    MX::Array<float> c;

    TEST_EXP((a = MX::Random<float>({3, 2, 3}, -2, 2)))
    TEST_EXP((b = MX::Random<float>({3}, -3, 1)))
    TEST_EXP((c = MX::Random<float>({2, 6}, -3, 1)))
    MX::SavePack<float>("save_test.mxp", {&a, &b, &c});
    auto arrays = MX::LoadPack<float>("save_test.mxp");
    std::cout << "loaded arrays  -->" << std::endl;
    for (auto i : arrays)
        std::cout << *i << std::endl;
}

void
conv_test()
{
    using namespace NN;

    TESTING_SECTION("convolve")
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
    TEST_EXP(MX::Convolve(E, F, 2, 3))
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

    TESTING_SECTION("sum")
    MX::Array<int> s;
    TEST_EXP((s = {{{1,2,3}, {4,5,6}}, {{7,8,9}, {10,11,12}}}))
    TEST_EXP(MX::Sum(s, 0))
    TEST_EXP(MX::Sum(s, 1))
    TEST_EXP(MX::Sum(s, 2))
    TEST_EXP(MX::Sum(s))
    TEST_EXP(MX::Sum(s, MX::Array<int>({{7,8,9}, {10,11,12}})))
}

void
access_test()
{
    using namespace NN;

    TESTING_SECTION("value access")
    MX::Array<float> a;
    TEST_EXP((a = MX::Random<float>({2, 2, 3})))

    TEST_EXP(a(0, 1))
    TEST_EXP(a.shape())
    TEST_EXP(a.dimensions())

    TESTING_SECTION("subarray access")
    TEST_EXP(a[0])
    TEST_EXP(a[1][0])
    TEST_EXP(a[1][0].shape())
    TEST_EXP(a[1](0, 2))
    TEST_EXP((a[1] = {{1, 2, 3}, {4, 5, 6}}))
    TEST_EXP(a)
}

void
iterator_test()
{
    using namespace NN;

    TESTING_SECTION("iterators")
    MX::Array<int> a;
    TEST_EXP((a = MX::Random<int>({2, 2}, 0, 5)))
    std::cout << "(i *= i)  -->  ";
     for (auto &i : a)
        std::cout << (i *= i) << " ";
    std::cout << std::endl;
    TEST_EXP(a)
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

    TESTING_SECTION("transpose")
    //MX::Array<int> a = MX::Random<int>({2, 1, 3}, 0, 5);
    MX::Array<int> a =
        {{{7, 4, 0, 5},
          {5, 7, 7, 8},
          {1, 7, 3, 2}},

         {{3, 8, 8, 4},
          {3, 9, 2, 1},
          {5, 5, 5, 9}}};
    TEST_EXP(a)
    TEST_EXP(a.shape())
    TEST_EXP(MX::Transpose(a, {1, 2, 0}))
    TEST_EXP(MX::Transpose(a))
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
    save_load_pack_test();
    conv_test();
    sum_test();
    access_test();
    iterator_test();
    transpose_test();

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "TIME: " << elapsed_secs << "s" << endl;
}

