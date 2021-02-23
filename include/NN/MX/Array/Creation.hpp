#pragma once

#include <bits/c++config.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#include "Array.hpp"

namespace NN
{

namespace MX
{

    template<typename T>
    Array<T>
    Empty(const Array<typename Array<T>::size_type> &shape)
    {
        Array<T> result;
        result.allocate(&(*(shape.begin())), shape.size());
        return result;
    }

    template<typename T>
    Array<T>
    Random(const Array<typename Array<T>::size_type> &shape, double from=0.0, double to=1.0)
    {
        Array<T> result = Empty<T>(shape);
        for (auto &i : result)
            i = (T)((double)std::rand() / (double)RAND_MAX * (to-from) + from);
        return result;
    }

    template<typename T>
    Array<T>
    Sequence(const Array<typename Array<T>::size_type> &shape, double start=1.0, double step=1.0)
    {
        Array<T> result = Empty<T>(shape);
        double current = start;
        for (auto &i : result)
            i = (T)current;
        return result;
    }

    template<typename T>
    Array<T>
    Full(const Array<typename Array<T>::size_type> &shape, typename Array<T>::value_type value)
    {
        Array<T> result = Empty<T>(shape);
        for (auto &i : result)
            i = value;
        return result;
    }

    template<typename T>
    Array<T>
    Zeros(const Array<typename Array<T>::size_type> &shape)
    { return Full<T>(shape, 0); }

    template<typename T>
    Array<T>
    Ones(const Array<typename Array<T>::size_type> &shape)
    { return Full<T>(shape, 1); }

} // namespace MX

} // namespace NN

