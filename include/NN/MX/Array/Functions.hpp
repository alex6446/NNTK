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
#include "Creation.hpp"
#include "NN/Core/Base.hpp"

namespace NN
{
namespace MX
{

    template<typename T>
    Array<T>
    Reshape(const Array<T> &array, const Array<typename Array<T>::size_type> &shape)
    {
        return Array<T>(array).reshape(shape);
    }

    template<typename T>
    Array<T>
    Dot(const Array<T> &left, const Array<T> &right)
    {
        using size_type = typename Array<T>::size_type;
        using value_type = typename Array<T>::value_type;
        assert(left.depth() == 2 && right.depth() == 2);
        assert(left.shape(1) == right.shape(0));
        Array<T> result = Zeros<T>({left.shape(0), right.shape(1)});
        for (value_type *di = result.data(), *li = (value_type *)left.data();
                li != left.data() + left.size();
                di += result.dimensions(0), li += left.dimensions(0))
            for (size_type j = 0; j < right.shape(1); ++j) {
                const value_type *rk = right.data();
                for (size_type k = 0; k < left.shape(1); ++k, rk += right.dimensions(0))
                    *(di+j) += *(li+k) * *(rk+j);
            }
        return result;
    }

    template<typename T>
    Array<T>
    Convolve(const Array<T> &array, const Array<T> &kernel, int padding, int stride)
    {
        using size_type = typename Array<T>::size_type;
        assert(array.depth() == 2 && kernel.depth() == 2);
        assert(stride >= 1);

        size_type cm = (array.shape(0) + 2*padding - kernel.shape(0)) / stride + 1;
        size_type cn = (array.shape(1) + 2*padding - kernel.shape(1)) / stride + 1;
        assert(cm > 0 && cn > 0);

        Array<T> c = Zeros<T>({cm, cn});
        for (size_type ai = -padding, ci = 0; ci < cm; ai+=stride, ++ci)
            for (size_type aj = -padding, cj = 0; cj < cn; aj+=stride, ++cj)
                for (size_type ki = 0; ki < kernel.shape(0); ++ki)
                    for (size_type kj = 0; kj < kernel.shape(1); ++kj)
                        c(ci, cj) += kernel(ki, kj) * array.force_get(ai+ki, aj+kj);
        return c;
    }

    template<typename T>
    Array<T>
    Sum(const Array<T> &array, int axes, bool keepdims=true)
    {
        using size_type = typename Array<T>::size_type;
        using depth_type = typename Array<T>::depth_type;
        depth_type depth = keepdims ? array.depth() : array.depth()-1;
        Array<size_type> shape = Empty<size_type>({depth});
        depth_type index = 0;
        for (depth_type i = 0; i < array.depth(); ++i) {
            if (i == axes) {
                if (keepdims)
                    shape.data(index++) = 1;
            } else {
                shape.data(index++) = array.shape(i);
            }
        }
        Array<T> s = Zeros<T>(shape);
        index = 0;
        size_type ds = array.dimensions(axes)*array.shape(axes);
        // Legacy code
        for (size_type i = 0; i < array.size()/ds; ++i) {
            for (size_type k = i*ds; k < array.dimensions(axes) + i*ds; ++k) {
                for (size_type j = k; j < ds+k; j+=array.dimensions(axes))
                    s.data(index) += array.data(j);
                index++;
            }
        }
        return s;
    }

    template<typename T>
    typename Array<T>::value_type
    Sum(const Array<T> &array)
    {
        typename Array<T>::value_type s = 0;
        for (auto i = array.data(); i != array.data() + array.size(); ++i)
            s += *i;
        return s;
    }


    template<typename T>
    Array<T>
    Sum(const Array<T> &left, const Array<T> &right)
    {
        using depth_type = typename Array<T>::depth_type;
        using size_type = typename Array<T>::size_type;
        assert(left.data() && right.data());
        assert(left.depth() >= right.depth());
        assert(left.size() % right.size() == 0);
        Array<T> s = left;
        for (size_type i = 0, index = 0; i < s.size(); ++i, ++index) {
            if (index >= right.size())
                index = 0;
            s.data(i) += right.data(index);
        }
        return s;
    }

    template<typename T>
    Array<T>
    Transpose(const Array<T> &array, const Array<typename Array<T>::depth_type> &order)
    {
        using depth_type = typename Array<T>::depth_type;
        using size_type = typename Array<T>::size_type;
        using value_type = typename Array<T>::value_type;
        assert(order.size() == array.depth() || order.depth() == 0);
        // Array of depth_type because its not the elements count but the index
        Array<depth_type> ord = order.depth() ? order
            : Sequence<depth_type>({array.depth()}, array.depth()-1, -1);
        std::vector<value_type *> ptrs(array.depth(), (value_type *)array.data());
        std::vector<size_type> inds(array.depth(), 0);
        Array<size_type> shape = array.shape();
        for (depth_type i = 0; i < array.depth(); ++i)
            shape.data(i) = array.shape(ord.data(i));
        Array<T> result = Empty<T>(shape);
        for (auto it = result.begin(); it != result.end(); ++it) {
            *it = *(ptrs.back());
            if (&(*it) != &(*(result.end()))-1) {
                depth_type last_dim = array.depth()-1;
                while (inds[last_dim]+1 >= array.shape(ord.data(last_dim)))
                    last_dim--;
                inds[last_dim]++;
                ptrs[last_dim] += array.dimensions(ord.data(last_dim));
                for (depth_type j = last_dim+1; j < array.depth(); ++j) {
                    inds[j] = 0;
                    ptrs[j] = ptrs[last_dim];
                }
            }
        }
        return result;
    }

    template<typename T>
    void
    SavePack(std::ofstream &file, const std::vector<const Array<T> *> &arrays)
    {
        NN_RUNTIME_ERROR(!file, "caNNot open a file")
        std::size_t arrays_count = arrays.size();
        file.write((char *)&arrays_count, sizeof(arrays.size()));
        for (auto *i : arrays)
            i->write(file);
    }

    template<typename T>
    void
    SavePack(std::string filepath, const std::vector<const Array<T> *> &arrays)
    {
        std::ofstream file(filepath, std::ios::binary);
        SavePack(file, arrays);
    }

    template<typename T>
    std::vector<std::shared_ptr<Array<T>>>
    LoadPack(std::ifstream &file)
    {
        NN_RUNTIME_ERROR(!file, "caNNot open a file")
        std::size_t arrays_count;
        file.read((char *)&arrays_count, sizeof(arrays_count));
        std::vector<std::shared_ptr<Array<T>>> arrays(arrays_count);
        for (std::size_t i = 0; i < arrays_count; ++i) {
            arrays[i] = std::make_shared<Array<T>>();
            arrays[i]->read(file);
        }
        return arrays;
    }

    template<typename T>
    std::vector<std::shared_ptr<Array<T>>>
    LoadPack(std::string filepath)
    {
        std::ifstream file(filepath, std::ios::binary);
        return LoadPack<T>(file);
    }


} // namespace MX

} // namespace NN
