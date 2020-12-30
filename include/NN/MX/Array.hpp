#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>

#include "ArrayIterator.hpp"

namespace NN
{
namespace MX
{

    template<typename T>
    class Array;

    template<typename T>
    std::ostream & operator<<(std::ostream& os, const Array<T> &array);

    template<typename T>
    Array<T> Random(const std::initializer_list<typename Array<T>::size_type> &shape, double from=0.0, double to=1.0);

    template<typename T>
    Array<T> Zeros(const std::initializer_list<typename Array<T>::size_type> &shape);

    template<typename T>
    Array<T> Ones(const std::initializer_list<typename Array<T>::size_type> &shape);

    template<typename T>
    Array<T> Dot(const Array<T> &left, const Array<T> &right);

    template<typename T>
    Array<T> Convolve(const Array<T> &array, const Array<T> &kernel, int padding, int stride);

    template<typename T>
    Array<T> Sum(const Array<T> &array, int axes, bool keepdims=true);

    template<typename T>
    typename Array<T>::value_type Sum(const Array<T> &array);

    template<typename T>
    Array<T> Sum(const Array<T> &left, const Array<T> &right);

} // namespace MX

namespace MX
{

    template<typename T>
    class Array
    {
    public:

        using size_type  = std::int64_t;
        using depth_type = std::int16_t ;
        using value_type = T;
        using iterator   = ArrayIterator<Array<T>>;

    public:

        Array();
        Array(const value_type *data, size_type size);
        Array(const std::initializer_list<value_type> &il);
        Array(const std::initializer_list<Array> &il);
        Array(const Array &copy);
        Array(Array &&move);
        Array(value_type *subdata, size_type size, size_type *shape, size_type *dimensions, depth_type depth, int array_type);
        ~Array();

        Array & operator=(const Array &array);
        Array & operator=(const std::initializer_list<value_type> &il);
        Array & operator=(const std::initializer_list<Array> &il);

        Array & operator[](size_type index);
        const Array & operator[](size_type index) const;

        template<class ...Types>
        value_type & operator()(Types... indices);

        template<class ... Types>
        const value_type & operator()(Types ... indices) const;

        template<class ...Types>
        value_type get(Types... indices) const;

        const size_type &
        size() const
        { return m_size; }

        const depth_type &
        depth() const
        { return m_depth; }

        const size_type *
        data() const
        { return m_data; }

        size_type *
        data()
        { return m_data; }

        const value_type &
        data(size_type index) const
        {
            assert(index >= 0 && index < m_size);
            return m_data[index];
        }

        value_type &
        data(size_type index)
        {
            assert(index >= 0 && index < m_size);
            return m_data[index];
        }

        // returns shape in form of subarray
        const Array<size_type> & shape() const;

        size_type
        shape(depth_type index) const
        {
            assert(index >= 0 && index < m_depth);
            return m_shape[index];
        }

        // returns dimensions in form of subarray
        const Array<size_type> & dimensions() const;

        size_type
        dimensions(depth_type index) const
        {
            assert(index >= 0 && index < m_depth);
            return m_dimensions[index];
        }

        iterator
        begin()
        { return iterator(m_data); }

        iterator
        end()
        { return iterator(m_data + m_size); }

        const Array & print(std::ostream &os=std::cout) const;

        const Array & save(std::string filename) const;
        const Array & load(std::string filename);

        bool operator==(const Array &array) const;

        bool
        operator!=(const Array &array) const
        { return !operator==(array); }

        Array operator+(value_type value) const;
        Array operator-(value_type value) const;
        Array operator*(value_type value) const;
        Array operator/(value_type value) const;

        Array operator+(const Array &array) const;
        Array operator-(const Array &array) const;
        Array operator*(const Array &array) const;
        Array operator/(const Array &array) const;

        Array operator-() const;

        Array & operator+=(value_type value);
        Array & operator-=(value_type value);
        Array & operator*=(value_type value);
        Array & operator/=(value_type value);

        Array & operator+=(const Array &array);
        Array & operator-=(const Array &array);
        Array & operator*=(const Array &array);
        Array & operator/=(const Array &array);

    private:

        void allocate();
        void clear();

        void allocate(const size_type *shape, depth_type depth);

        template<class ...Types>
        value_type * element_get(Types... indices);

        template<class ...Types>
        const value_type * element_get(Types... indices) const;

        void elements_print(std::ostream &os, depth_type index, value_type *ptr) const;

    public:

        friend std::ostream & operator<< <>(std::ostream &os, const Array &array);

        friend Array Random<>(const std::initializer_list<size_type> &shape, double from, double to);
        friend Array Zeros<>(const std::initializer_list<size_type> &shape);
        friend Array Ones<>(const std::initializer_list<size_type> &shape);
        friend Array Dot<>(const Array &left, const Array &right);
        friend Array Convolve<>(const Array &array, const Array &kernel, int padding, int stride);
        friend Array Sum<>(const Array &array, int axes, bool keepdims);
        friend value_type Sum<>(const Array &array);
        friend Array Sum<>(const Array &left, const Array& right);

        friend Array
        operator+(value_type value, const Array &array)
        {
            Array<value_type> result = array;
            for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
                *i += value;
            return result;
        }

        friend Array
        operator-(value_type value, const Array &array)
        {
            Array<value_type> result = array;
            for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
                *i = value - *i;
            return result;
        }

        friend Array
        operator*(value_type value, const Array &array)
        {
            Array<value_type> result = array;
            for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
                *i *= value;
            return result;
        }

        friend Array
        operator/(value_type value, const Array &array)
        {
            Array<value_type> result = array;
            for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i) {
                if (*i == 0)
                    throw std::runtime_error("operator/: division by zero");
                *i = value / *i;
            }
            return result;
        }

    private:

        size_type m_size;
        value_type *m_data = nullptr;

        depth_type m_depth;
        size_type *m_shape = nullptr;
        size_type *m_dimensions = nullptr;

    private:

        mutable std::shared_ptr<Array<size_type>> m_shape_subarray;
        mutable std::shared_ptr<Array<size_type>> m_dimensions_subarray;
        mutable std::shared_ptr<Array<value_type>> m_data_subarray;

        int m_array_type = 0;

    };

} // namespace MX

namespace MX
{

    template<typename T>
    void
    Array<T>::
    allocate()
    {
        if (m_size)
            m_data = new value_type[m_size];
        if (m_depth) {
            m_shape = new size_type[m_depth];
            m_dimensions = new size_type[m_depth];
        }
    }

    template<typename T>
    void
    Array<T>::
    clear()
    {
        if (m_array_type == 2)
            return;
        if (m_depth) {
            delete[] m_shape;
            m_shape = nullptr;
            delete[] m_dimensions;
            m_dimensions = nullptr;
        }
        if (m_array_type == 1)
            return;
        if (m_data) {
            delete[] m_data;
            m_data = nullptr;
        }
        m_size = 0;
        m_depth = 0;
    }

    template<typename T>
    void
    Array<T>::
    allocate(const size_type *shape, depth_type depth)
    {
        if (!depth) return;
        m_depth = depth;
        m_shape = new size_type[m_depth];
        m_dimensions = new size_type[m_depth];
        m_dimensions[m_depth-1] = 1;
        for (depth_type i = m_depth-1; i > 0; --i) {
            m_shape[i] = shape[i];
            m_dimensions[i-1] = m_dimensions[i] * m_shape[i];
        }
        m_shape[0] = shape[0];
        m_size = m_dimensions[0] * m_shape[0];
        m_data = new value_type[m_size];
    }

    template<typename T>
    template<class ...Types>
    typename Array<T>::value_type *
    Array<T>::
    element_get(Types... indices)
    {
        const std::array<size_type, sizeof...(indices)> inds = {{indices...}};
        value_type *ptr = m_data;
        for (depth_type i = 0; i < inds.size(); ++i) {
            if (inds[i] < 0 || inds[i] >= m_shape[i])
                return nullptr;
            ptr += m_dimensions[i] * inds[i];
        }
        return ptr;
    }

    template<typename T>
    template<class ...Types>
    const typename Array<T>::value_type *
    Array<T>::
    element_get(Types... indices) const
    {
        const std::array<size_type, sizeof...(indices)> inds = {{indices...}};
        value_type *ptr = m_data;
        for (depth_type i = 0; i < inds.size(); ++i) {
            if (inds[i] < 0 || inds[i] >= m_shape[i])
                return nullptr;
            ptr += m_dimensions[i] * inds[i];
        }
        return ptr;
    }

    template<typename T>
    Array<T>::
    Array()
    : m_size(0)
    , m_depth(0)
    {}

    template<typename T>
    Array<T>::
    Array(const value_type *data, size_type size)
    : m_size(size)
    , m_depth(1)
    {
        assert(m_size >= 0);
        allocate();
        m_shape[0] = m_size;
        m_dimensions[0] = 1;
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = data[i];
    }

    template<typename T>
    Array<T>::
    Array(const std::initializer_list<value_type> &il)
    : m_size(il.size())
    , m_depth(1)
    {
        allocate();
        m_shape[0] = il.size();
        m_dimensions[0] = 1;
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = il.begin()[i];
    }

    template<typename T>
    Array<T>::
    Array(const std::initializer_list<Array> &il)
    : m_size(il.begin()->m_size * il.size())
    , m_depth(il.begin()->m_depth + 1)
    {
        allocate();
        m_shape[0] = il.size();
        m_dimensions[0] = il.begin()->m_size;
        for (size_type i = 0, c = 0; i < il.size(); ++i)
            for (size_type j = 0; j < il.begin()->m_size; ++j, ++c)
                m_data[c] = il.begin()[i].m_data[j];
        for (depth_type i = 0; i < il.begin()->m_depth; ++i) {
            m_shape[i+1] = il.begin()->m_shape[i];
            m_dimensions[i+1] = il.begin()->m_dimensions[i];
        }
    }

    template<typename T>
    Array<T>::
    Array(value_type *data, size_type size, size_type *shape, size_type *dimensions, depth_type depth, int array_type)
    {
        m_data = data;
        m_size = size;
        m_shape = shape;
        m_dimensions = dimensions;
        m_depth = depth;
        m_array_type = array_type;
    }

    template<typename T>
    Array<T>::
    Array(const Array &copy)
    : m_size(copy.m_size)
    , m_depth(copy.m_depth)
    {
        allocate();
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = copy.m_data[i];
        for (depth_type i = 0; i < m_depth; ++i) {
            m_shape[i] = copy.m_shape[i];
            m_dimensions[i] = copy.m_dimensions[i];
        }
    }

    template<typename T>
    Array<T>::
    Array(Array &&move)
    : m_size(move.m_size)
    , m_depth(move.m_depth)
    {
        allocate();
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = move.m_data[i];
        for (depth_type i = 0; i < m_depth; ++i) {
            m_shape[i] = move.m_shape[i];
            m_dimensions[i] = move.m_dimensions[i];
        }
    }

    template<typename T>
    Array<T>::
    ~Array()
    {
        clear();
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator=(const Array &array)
    {
        if (m_array_type == 0) {
            clear();
            m_size = array.m_size;
            m_depth = array.m_depth;
            allocate();
            for (depth_type i = 0; i < m_depth; ++i) {
                m_shape[i] = array.m_shape[i];
                m_dimensions[i] = array.m_dimensions[i];
            }
        } else {
            assert(m_depth == array.m_depth);
            for (depth_type i = 0; i < m_depth; ++i)
                assert(m_shape[i] == array.m_shape[i]);
        }
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = array.m_data[i];
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator=(const std::initializer_list<T> &il)
    {
        if (m_array_type == 0) {
            clear();
            m_size = il.size();
            m_depth = 1;
            allocate();
            m_shape[0] = il.size();
            m_dimensions[0] = 1;
        } else {
            assert(m_depth == 1);
            assert(m_shape[0] == il.size());
        }
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] = *(il.begin() + i);
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator=(const std::initializer_list<Array> &il)
    {
        if (m_array_type == 0) {
            m_size = il.begin()->m_size * il.size();
            m_depth = il.begin()->m_depth + 1;
            allocate();
            m_shape[0] = il.size();
            m_dimensions[0] = il.begin()->m_size;
            for (depth_type i = 0; i < il.begin()->m_depth; ++i) {
                m_shape[i+1] = il.begin()->m_shape[i];
                m_dimensions[i+1] = il.begin()->m_dimensions[i];
            }
        } else {
            assert(m_depth == il.begin()->m_depth + 1);
            for (depth_type i = 0; i < il.begin()->m_depth; ++i) {
                assert(m_shape[i+1] == il.begin()->m_shape[i]);
            }
        }
        for (size_type i = 0, c = 0; i < il.size(); ++i)
            for (size_type j = 0; j < il.begin()->m_size; ++j, ++c)
                m_data[c] = (il.begin() + i)->m_data[j];
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator[](size_type index)
    {
        assert(m_depth > 1);
        assert(index >= 0 && index < m_shape[0]);
        m_data_subarray = std::make_shared<Array<value_type>>(
                    m_data + m_dimensions[0] * index,
                    m_dimensions[0],
                    m_shape + 1,
                    m_dimensions + 1,
                    m_depth - 1, 2);
        return *m_data_subarray;
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    operator[](size_type index) const
    {
        assert(m_depth > 1);
        assert(index >= 0 && index < m_shape[0]);
        m_data_subarray = std::make_shared<Array<value_type>>(
                    m_data + m_dimensions[0] * index,
                    m_dimensions[0],
                    m_shape + 1,
                    m_dimensions + 1,
                    m_depth - 1, 2);
        return *m_data_subarray;
    }

    template<typename T>
    template<class ...Types>
    typename Array<T>::value_type &
    Array<T>::
    operator()(Types... indices)
    {
        assert(sizeof...(indices) <= m_depth);
        value_type *p = element_get(indices...);
        assert(p);
        return *p;
    }

    template<typename T>
    template<class ...Types>
    const typename Array<T>::value_type &
    Array<T>::
    operator()(Types... indices) const
    {
        assert(sizeof...(indices) <= m_depth);
        const value_type *p = element_get(indices...);
        assert(p);
        return *p;
    }

    template<typename T>
    template<class ...Types>
    typename Array<T>::value_type
    Array<T>::
    get(Types... indices) const
    {
        assert(sizeof...(indices) <= m_depth);
        const value_type *p = element_get(indices...);
        if (p == nullptr)
            return 0;
        return *p;
    }

    template<typename T>
    const Array<typename Array<T>::size_type> &
    Array<T>::
    shape() const
    {
        m_shape_subarray = std::make_shared<Array<size_type>>(
                    m_shape, m_depth,
                    new size_type {m_depth},
                    new size_type {1}, 1, 1);
        return *m_shape_subarray;
    }

    template<typename T>
    const Array<typename Array<T>::size_type> &
    Array<T>::
    dimensions() const
    {
        m_dimensions_subarray = std::make_shared<Array<size_type>>(
                    m_dimensions, m_depth,
                    new size_type {m_depth},
                    new size_type {1}, 1, 1);
        return *m_dimensions_subarray;
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    print(std::ostream &os) const
    {
        if (!m_data) os << "{}";
        else elements_print(os, 0, m_data);
        return *this;
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    save(std::string filename) const
    {
        std::ofstream file(filename + ".mxa", std::ios::binary);
        if (!file)
            throw std::runtime_error("save: cannot open file");
        file.write((char *)&m_depth, sizeof(m_depth));
        for (size_type *i = m_shape; i != m_shape + m_depth; ++i)
            file.write((char *)i, sizeof(*m_shape));
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            file.write((char *)i, sizeof(*m_data));
        return *this;
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    load(std::string filename)
    {
        std::ifstream file(filename + ".mxa", std::ios::binary);
        if (!file)
            throw std::runtime_error("load: cannot open file");
        if (m_array_type == 0) {
            clear();
            file.read((char *)&m_depth, sizeof(m_depth));
            size_type *s = new size_type[m_depth];
            for (size_type *i = s; i != s + m_depth; ++i)
                file.read((char *)i, sizeof(*s));
            allocate(s, m_depth);
        } else {
            depth_type depth;
            file.read((char *)&depth, sizeof(m_depth));
            if (depth != m_depth)
                throw std::runtime_error("load: subarray depth not match");
            size_type s;
            for (size_type *i = m_shape; i != m_shape + m_depth; ++i) {
                file.read((char *)&s, sizeof(s));
                if (s != *i)
                    throw std::runtime_error("load: subarray shape not match");
            }
        }
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            file.read((char *)i, sizeof(*m_data));
        return *this;
    }

    template<typename T>
    void
    Array<T>::
    elements_print(std::ostream &os, depth_type index, value_type *place) const
    {
        os << "{";
        if (index == m_depth-1) {
            for (size_type i = 0; i < m_shape[index]; ++i) {
                os << place[i];
                if (i != m_shape[index]-1)
                    os << ", ";
            }
        } else {
            for (size_type i = 0; i < m_shape[index]; ++i) {
                elements_print(os, index+1, place+m_dimensions[index]*i);
                if (i != m_shape[index]-1)
                    os << ", ";
            }
        }
        os << "}";
    }

    template<typename T>
    bool
    Array<T>::
    operator==(const Array &array) const
    {
        if (m_size != array.m_size || m_depth != array.m_depth)
            return false;
        for (depth_type i = 0; i < m_depth; ++i)
            if (m_shape[i] != array.m_shape[i])
                return false;
        for (size_type i = 0; i < m_size; ++i)
            if (m_data[i] != array.m_data)
                return false;
        return true;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator+(value_type value) const
    {
        Array<T> result = *this;
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i += value;
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator-(value_type value) const
    {
        Array<T> result = *this;
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i -= value;
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator*(value_type value) const
    {
        Array<T> result = *this;
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i *= value;
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator/(value_type value) const
    {
        if (!value)
            throw std::runtime_error("operator/:number: division by zero");
        Array<T> result = *this;
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i /= value;
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator+(const Array &array) const
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        Array<T> result = *this;
        for (size_type i = 0; i < m_size; ++i)
            result.m_data[i] += array.m_data[i];
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator-(const Array &array) const
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        Array<T> result = *this;
        for (size_type i = 0; i < m_size; ++i)
            result.m_data[i] -= array.m_data[i];
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator*(const Array &array) const
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        Array<T> result = *this;
        for (size_type i = 0; i < m_size; ++i)
            result.m_data[i] *= array.m_data[i];
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator/(const Array &array) const
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        Array<T> result = *this;
        for (size_type i = 0; i < m_size; ++i) {
            if (!array.m_data[i])
                throw std::runtime_error("operator/:array: division by zero");
            result.m_data[i] /= array.m_data[i];
        }
        return result;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator-() const
    {
        Array<T> result = *this;
        for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i)
            *i *= -1;
        return result;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator+=(value_type value)
    {
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            *i += value;
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator-=(value_type value)
    {
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            *i -= value;
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator*=(value_type value)
    {
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            *i *= value;
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator/=(value_type value)
    {
        if (!value)
            throw std::runtime_error("operator/=:number: division by zero");
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            *i /= value;
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator+=(const Array &array)
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] += array.m_data[i];
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator-=(const Array &array)
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] -= array.m_data[i];
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator*=(const Array &array)
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        for (size_type i = 0; i < m_size; ++i)
            m_data[i] *= array.m_data[i];
        return *this;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    operator/=(const Array &array)
    {
        assert(m_size == array.m_size && m_depth == array.m_depth);
        for (depth_type i = 0; i < m_depth; ++i)
            assert(m_shape[i] == array.m_shape[i]);
        for (size_type i = 0; i < m_size; ++i) {
            if (!array.m_data[i])
                throw std::runtime_error("operator/=:array: division by zero");
            m_data[i] /= array.m_data[i];
        }
        return *this;
    }

} // namespace MX

namespace MX
{

    template<typename T>
    std::ostream &
    operator<<(std::ostream &os, const Array<T> &array)
    {
        array.print(os);
        return os;
    }

} // namespace MX

} // namespace NN

#include "ArrayFuncs.hpp"

