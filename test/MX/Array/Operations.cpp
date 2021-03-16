#include "catch2/catch.hpp"

//#define NN_GPU_ENABLED
#include "NN/MX/Array.hpp"

using namespace NN;

TEST_CASE("MX::Array operations", "[array]")
{
    SECTION("Unary operations")
    {
        MX::Array<int> array = {{-3, 2}, {0, -3}, {16, 2}};

        SECTION("-array")
        {
            MX::Array<int> negative_array = -array;
            MX::Array<int> expected = {{3, -2}, {0, 3}, {-16, -2}};

            CHECK(negative_array == expected);
        }
    }

    SECTION("Arithmetic operations between two arrays")
    {
        MX::Array<int> left_array = {{{7, 8, 9}, {-10, -11, -12}}, {{13, 14, 15}, {-16, -17, -18}}};
        MX::Array<int> right_array = {{{-6, -5, -4}, {-3, -2, -1}}, {{0, 1, 2}, {3, 4, 5}}};

        SECTION("array + array")
        {
            MX::Array<int> result = left_array + right_array;
            MX::Array<int> expected = {{{1, 3, 5}, {-13, -13, -13}}, {{13, 15, 17}, {-13, -13, -13}}};

            CHECK(result == expected);
        }

        SECTION("array - array")
        {
            MX::Array<int> result = left_array - right_array;
            MX::Array<int> expected = {{{13, 13, 13}, {-7, -9, -11}}, {{13, 13, 13}, {-19, -21, -23}}};

            CHECK(result == expected);
        }

        SECTION("array * array")
        {
            MX::Array<int> result = left_array * right_array;
            MX::Array<int> expected = {{{-42, -40, -36}, {30, 22, 12}}, {{0, 14, 30}, {-48, -68, -90}}};

            CHECK(result == expected);
        }

        SECTION("array / array")
        {
            right_array(1, 0, 0) = 13;
            MX::Array<int> result = left_array / right_array;
            MX::Array<int> expected = {{{-1, -1, -2}, {3, 5, 12}}, {{1, 14, 7}, {-5, -4, -3}}};

            CHECK(result == expected);
        }

        SECTION("array / array (zero division)")
        {
            CHECK_THROWS(left_array / right_array);
        }

        SECTION("array += array")
        {
            left_array += right_array;
            MX::Array<int> expected = {{{1, 3, 5}, {-13, -13, -13}}, {{13, 15, 17}, {-13, -13, -13}}};

            CHECK(left_array == expected);
        }

        SECTION("array -= array")
        {
            left_array -= right_array;
            MX::Array<int> expected = {{{13, 13, 13}, {-7, -9, -11}}, {{13, 13, 13}, {-19, -21, -23}}};

            CHECK(left_array == expected);
        }

        SECTION("array *= array")
        {
            left_array *= right_array;
            MX::Array<int> expected = {{{-42, -40, -36}, {30, 22, 12}}, {{0, 14, 30}, {-48, -68, -90}}};

            CHECK(left_array == expected);
        }

        SECTION("array /= array")
        {
            right_array(1, 0, 0) = 13;
            left_array /= right_array;
            MX::Array<int> expected = {{{-1, -1, -2}, {3, 5, 12}}, {{1, 14, 7}, {-5, -4, -3}}};

            CHECK(left_array == expected);
        }

        SECTION("array /= array (zero division)")
        {
            CHECK_THROWS((left_array /= right_array));
        }
    }

    SECTION("Arithmetic operations between array and scalar")
    {
        MX::Array<int> array = {{-6, 12, 0}, {-15, -9, 3}};
        int scalar = 3;
        
        SECTION("array + scalar")
        {
            MX::Array<int> result = array + scalar;
            MX::Array<int> expected = {{-3, 15, 3}, {-12, -6, 6}};

            CHECK(result == expected);
        }
        
        SECTION("array - scalar")
        {
            MX::Array<int> result = array - scalar;
            MX::Array<int> expected = {{-9, 9, -3}, {-18, -12, 0}};

            CHECK(result == expected);
        }
        
        SECTION("array * scalar")
        {
            MX::Array<int> result = array * scalar;
            MX::Array<int> expected = {{-18, 36, 0}, {-45, -27, 9}};

            CHECK(result == expected);
        }

        SECTION("array / scalar")
        {
            MX::Array<int> result = array / scalar;
            MX::Array<int> expected = {{-2, 4, 0}, {-5, -3, 1}};

            CHECK(result == expected);
        }

        SECTION("array / scalar (zero division)")
        {
            CHECK_THROWS((array / 0));
        }

        SECTION("array += scalar")
        {
            array += scalar;
            MX::Array<int> expected = {{-3, 15, 3}, {-12, -6, 6}};

            CHECK(array == expected);
        }
        
        SECTION("array -= scalar")
        {
            array -= scalar;
            MX::Array<int> expected = {{-9, 9, -3}, {-18, -12, 0}};

            CHECK(array == expected);
        }
        
        SECTION("array *= scalar")
        {
            array *= scalar;
            MX::Array<int> expected = {{-18, 36, 0}, {-45, -27, 9}};

            CHECK(array == expected);
        }

        SECTION("array /= scalar")
        {
            array /= scalar;
            MX::Array<int> expected = {{-2, 4, 0}, {-5, -3, 1}};

            CHECK(array == expected);
        }

        SECTION("array /= scalar (zero division)")
        {
            CHECK_THROWS((array /= 0));
        }
    }

    SECTION("Arithmetic operations between scalar and array")
    {
        int scalar = 8;
        MX::Array<int> array = {-4, 16, 0};
        
        SECTION("scalar + array")
        {
            MX::Array<int> result = scalar + array;
            MX::Array<int> expected = {4, 24, 8};

            CHECK(result == expected);
        }
        
        SECTION("scalar - array")
        {
            MX::Array<int> result = scalar - array;
            MX::Array<int> expected = {12, -8, 8};

            CHECK(result == expected);
        }
        
        SECTION("scalar * array")
        {
            MX::Array<int> result = scalar * array;
            MX::Array<int> expected = {-32, 128, 0};

            CHECK(result == expected);
        }

        SECTION("scalar / array")
        {
            array(2) = 1;
            MX::Array<int> result = scalar / array;
            MX::Array<int> expected = {-2, 0, 8};

            CHECK(result == expected);
        }

        SECTION("array / scalar (zero division)")
        {
            CHECK_THROWS((scalar / array));
        }
    }
}
