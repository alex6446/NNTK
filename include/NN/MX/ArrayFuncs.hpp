#include <initializer_list>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <cassert>

namespace NN {

namespace MX {

    template<typename T>
    class Array;


    template<typename T>
    Array<T>
    Random(const std::initializer_list<typename Array<T>::size_type> &shape, double from, double to)
    {
        using value_type = typename Array<T>::value_type;
        Array<T> result;
        result.allocate(shape.begin(), shape.size());
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i = (T)((double)std::rand() / (double)RAND_MAX * (to-from) + from);
        return result;
    }

    template<typename T>
    Array<T>
    Zeros(const std::initializer_list<typename Array<T>::size_type> &shape)
    {
        using value_type = typename Array<T>::value_type;
        Array<T> result;
        result.allocate(shape.begin(), shape.size());
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i = 0;
        return result;
    }

    template<typename T>
    Array<T>
    Ones(const std::initializer_list<typename Array<T>::size_type> &shape)
    {
        using value_type = typename Array<T>::value_type;
        Array<T> result;
        result.allocate(shape.begin(), shape.size());
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i = 1;
        return result;
    }

    template<typename T>
    Array<T>
    Dot(const Array<T> &left, const Array<T> &right)
    {
        using size_type = typename Array<T>::size_type;
        using value_type = typename Array<T>::value_type;
        assert(left.m_depth == 2 && right.m_depth == 2);
        assert(left.m_shape[1] == right.m_shape[0]);
        Array<T> result;
        result.allocate(std::initializer_list<size_type>({left.m_shape[0], right.m_shape[1]}).begin(), 2);
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i = 0;
        for (value_type *di = result.m_data, *li = left.m_data, *le = &left.m_data[left.m_size]; li != le; di += result.m_dimensions[0], li += left.m_dimensions[0])
            for (size_type j = 0; j < right.m_shape[1]; ++j) {
                value_type *rk = right.m_data;
                for (size_type k = 0; k < left.m_shape[1]; ++k, rk += right.m_dimensions[0])
                    *(di+j) += *(li+k) * *(rk+j);
            }
        return result;
    }

    template<typename T>
    Array<T>
    Convolve(const Array<T> &array, const Array<T> &kernel, int padding, int stride)
    {
        using size_type = typename Array<T>::size_type;
        assert(array.m_depth == 2 && kernel.m_depth == 2);
        assert(stride >= 1);

        size_type cm = (array.shape(0) + 2*padding - kernel.shape(0)) / stride + 1;
        size_type cn = (array.shape(1) + 2*padding - kernel.shape(1)) / stride + 1;
        assert(cm > 0 && cn > 0);

        Array<T> c = Zeros<T>({cm, cn});
        for (size_type ai = -padding, ci = 0; ci < cm; ai+=stride, ++ci)
            for (size_type aj = -padding, cj = 0; cj < cn; aj+=stride, ++cj)
                for (size_type ki = 0; ki < kernel.shape(0); ++ki)
                    for (size_type kj = 0; kj < kernel.shape(1); ++kj)
                        c(ci, cj) += kernel(ki, kj) * array.get(ai+ki, aj+kj);
        return c;
    }

    template<typename T>
    Array<T>
    Sum(const Array<T> &array, int axes, bool keepdims)
    {
        using size_type = typename Array<T>::size_type;
        using depth_type = typename Array<T>::depth_type;
        depth_type depth = keepdims ? array.depth() : array.depth()-1;
        size_type* shape = new size_type[depth];
        depth_type index = 0;
        for (depth_type i = 0; i < array.depth(); ++i) {
            if (i == axes) {
                if (keepdims)
                    shape[index++] = 1;
            } else {
                shape[index++] = array.shape(i);
            }
        }
        Array<T> s;
        s.allocate(shape, depth);
        for (size_type i = 0; i < s.size(); ++i)
            s.data(i) = 0;
        index = 0;
        size_type ds = array.dimensions(axes)*array.shape(axes);
        // Legacy code
        for (size_type i = 0; i < array.size()/ds; ++i) {
            for (size_type k = i*ds; k < array.dimensions(axes) + i*ds; ++k) {
                for (size_type j = k; j < ds+k; j+=array.dimensions(axes))
                    s.m_data[index] += array.m_data[j];
                index++;
            }
        }
        return s;
    }

    template<typename T>
    typename Array<T>::value_type
    Sum(const Array<T> &array)
    {
        using value_type = typename Array<T>::value_type;
        value_type s = 0;
        for (value_type *i = array.m_data; i != array.m_data + array.m_size; ++i)
            s += *i;
        return s;
    }


    template<typename T>
    Array<T>
    Sum(const Array<T> &left, const Array<T> &right)
    {
        using depth_type = typename Array<T>::depth_type;
        using size_type = typename Array<T>::size_type;
        assert(left.m_data && right.m_data);
        assert(left.depth() >= right.depth());
        for (depth_type i = 1; i <= right.depth(); ++i)
            assert(left.shape(left.depth()-i) == right.shape(right.depth()-i));
        MX::Array<T> s = left;
        for (size_type i = 0, index = 0; i < s.size(); ++i, ++index) {
            if (index >= right.size())
                index = 0;
            s.m_data[i] += right.m_data[index];
        }
        return s;
    }

}

namespace Internal {



}

}
