#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <cassert>
#include <memory>
#include <vector>

#include "NN/Core/Base.hpp"
#include "NN/Core/CPU/Memory.hpp"
#include "Iterator.hpp"
#include "CPU/ArithmeticOperations.hpp"

#ifdef NN_GPU_ENABLED
#include "NN/Core/GPU/Memory.hpp"
#include "GPU/ArithmeticOperations.hpp"
#endif

namespace NN
{
namespace MX
{

    template<typename T>
    class Array;

    template<typename T>
    Array<T> Empty(const Array<typename Array<T>::size_type> &shape);

    template<typename T>
    Array<T> Transpose(const Array<T> &array, const Array<typename Array<T>::depth_type> &order={});

} // namespace MX

namespace MX
{

    template<typename T>
    class Array
    {
    public:

        using size_type      = std::int64_t;
        using depth_type     = std::int16_t ;
        using value_type     = T;
        using iterator       = ArrayIterator<Array<T>>;
        using const_iterator = ArrayIterator<const Array<T>>;

        inline static const std::string FileExt = "mxa";
        inline static DeviceType Device = NN_USE_GPU ? DeviceType::GPU : DeviceType::CPU;

        enum class Type { Array, Subarray, Shape };

    public:

        Array();
        Array(const value_type *data, size_type size);
        Array(const std::initializer_list<value_type> &il);
        Array(const std::initializer_list<Array> &il);
        Array(const Array &copy);
        Array(Array &&move);
        Array(value_type *subdata, size_type size, size_type *shape, size_type *dimensions, depth_type depth, Type array_type);
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
        value_type force_get(Types... indices) const;

        template<class ...Types>
        Array & force_add(T value, Types... indices);

        const size_type &
        size() const
        { return m_size; }

        const depth_type &
        depth() const
        { return m_depth; }

        const value_type *
        data() const
        { return m_data; }

        value_type *
        data()
        { return m_data; }

        const value_type &
        data(size_type index) const
        { assert(index >= 0 && index < m_size); return m_data[index]; }

        value_type &
        data(size_type index)
        { assert(index >= 0 && index < m_size); return m_data[index]; }

        // returns shape in form of subarray
        const Array<size_type> & shape() const;

        size_type
        shape(depth_type index) const
        { assert(index >= 0 && index < m_depth); return m_shape[index]; }

        // returns dimensions in form of subarray
        const Array<size_type> & dimensions() const;

        size_type
        dimensions(depth_type index) const
        { assert(index >= 0 && index < m_depth); return m_dimensions[index]; }

        Array & reshape(const Array<size_type> &shape);

        Array &
        transpose(const Array<typename Array<T>::depth_type> &order={})
        { *this = Transpose(*this, order); return *this; }

        iterator
        begin()
        { return iterator(m_data); }

        iterator
        end()
        { return iterator(m_data + m_size); }

        const_iterator
        begin() const
        { return const_iterator(m_data); }

        const_iterator
        end() const
        { return const_iterator(m_data + m_size); }

        const Array & print(std::ostream &os=std::cout) const;
        const Array & parse(std::istream &is=std::cin);

        const Array & write(std::ostream &stream) const;
        const Array & read(std::istream &stream);

        const Array & save_to_file(std::string filepath) const;
        const Array & load_from_file(std::string filepath);

        bool operator==(const Array &array) const;

        bool
        operator!=(const Array &array) const
        { return !operator==(array); }

        Array operator-() const;

        Array operator+(const Array &array) const;
        Array operator-(const Array &array) const;
        Array operator*(const Array &array) const;
        Array operator/(const Array &array) const;

        Array & operator+=(const Array &array);
        Array & operator-=(const Array &array);
        Array & operator*=(const Array &array);
        Array & operator/=(const Array &array);

        Array operator+(value_type value) const;
        Array operator-(value_type value) const;
        Array operator*(value_type value) const;
        Array operator/(value_type value) const;

        Array & operator+=(value_type value);
        Array & operator-=(value_type value);
        Array & operator*=(value_type value);
        Array & operator/=(value_type value);

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

        friend Array Empty<>(const Array<size_type> &shape);

#define MX_ARRAY_OPERATION_VA(op, mode) \
        friend Array \
        operator op (value_type value, const Array &array) \
        { \
            Array<value_type> result = array; \
            for (value_type *i = result.m_data; i != result.m_data + result.m_size; ++i) { \
                if (mode) \
                    NN_RUNTIME_ERROR(*i == 0, "division by zero") \
                *i = value op *i; \
            } \
            return result; \
        }

        MX_ARRAY_OPERATION_VA(+, 0)
        MX_ARRAY_OPERATION_VA(-, 0)
        MX_ARRAY_OPERATION_VA(*, 0)
        MX_ARRAY_OPERATION_VA(/, 1)

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

        Type m_array_type = Type::Array;

        DeviceType m_device = Device;

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
            NN_GPU_FUNCTION_CALL(true, NN::internal::allocate_memory, (&m_data, m_size))
        if (m_depth) {
            NN_GPU_FUNCTION_CALL(true, NN::internal::allocate_memory, (&m_shape, m_depth))
            NN_GPU_FUNCTION_CALL(true, NN::internal::allocate_memory, (&m_dimensions, m_depth))
        }
    }

    template<typename T>
    void
    Array<T>::
    clear()
    {
        if (m_array_type == Type::Subarray)
            return;
        if (m_shape) {
            NN_GPU_FUNCTION_CALL(true, NN::internal::free_memory, (&m_shape))
            m_shape = nullptr;
        }
        if (m_dimensions) {
            NN_GPU_FUNCTION_CALL(true, NN::internal::free_memory, (&m_dimensions))
            m_dimensions = nullptr;
        }
        if (m_array_type == Type::Shape)
            return;
        if (m_data) {
            NN_GPU_FUNCTION_CALL(true, NN::internal::free_memory, (&m_data))
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
        NN_GPU_FUNCTION_CALL(true, NN::internal::allocate_memory, (&m_shape, m_depth))
        NN_GPU_FUNCTION_CALL(true, NN::internal::allocate_memory, (&m_dimensions, m_depth))
        m_dimensions[m_depth-1] = 1;
        for (depth_type i = m_depth-1; i > 0; --i) {
            m_shape[i] = shape[i];
            m_dimensions[i-1] = m_dimensions[i] * m_shape[i];
        }
        m_shape[0] = shape[0];
        m_size = m_dimensions[0] * m_shape[0];
        NN_GPU_FUNCTION_CALL(true, NN::internal::allocate_memory, (&m_data, m_size))
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
    Array(value_type *data, size_type size, size_type *shape, size_type *dimensions, depth_type depth, Type array_type)
    : m_size(size)
    , m_data(data)
    , m_depth(depth)
    , m_shape(shape)
    , m_dimensions(dimensions)
    , m_array_type(array_type)
    { }

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
    , m_data(move.m_data)
    , m_depth(move.m_depth)
    , m_shape(move.m_shape)
    , m_dimensions(move.m_dimensions)
    {
        move.m_shape = nullptr;
        move.m_dimensions = nullptr;
        move.m_data = nullptr;
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
        if (m_array_type == Type::Array) {
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
        if (m_array_type == Type::Array) {
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
        if (m_array_type == Type::Array) {
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
                    m_depth - 1,
                    Type::Subarray);
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
                    m_depth - 1,
                    Type::Subarray);
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
    force_get(Types... indices) const
    {
        assert(sizeof...(indices) <= m_depth);
        const value_type *p = element_get(indices...);
        if (p == nullptr)
            return 0;
        return *p;
    }

    template<typename T>
    template<class ...Types>
    Array<T> &
    Array<T>::
    force_add(T value, Types... indices)
    {
        assert(sizeof...(indices) <= m_depth);
        value_type *p = element_get(indices...);
        if (p != nullptr)
            *p += value;
        return *this;
    }

    template<typename T>
    const Array<typename Array<T>::size_type> &
    Array<T>::
    shape() const
    {
        m_shape_subarray = std::make_shared<Array<size_type>>(
                    m_shape, m_depth,
                    new size_type {m_depth},
                    new size_type {1}, 1,
                    Type::Shape);
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
                    new size_type {1}, 1,
                    Type::Shape);
        return *m_dimensions_subarray;
    }

    template<typename T>
    Array<T> &
    Array<T>::
    reshape(const Array<size_type> &shape)
    {
        assert(m_array_type == Type::Array);
        size_type size = 1;
        for (auto &i : shape)
            size *= i;
        NN_RUNTIME_ERROR(size != m_size, "total shape size not match")
        delete [] m_shape;
        delete [] m_dimensions;
        m_depth = shape.size();
        m_shape = new size_type[m_depth];
        m_dimensions = new size_type[m_depth];
        m_dimensions[m_depth-1] = 1;
        for (depth_type i = m_depth-1; i > 0; --i) {
            m_shape[i] = shape.data(i);
            m_dimensions[i-1] = m_dimensions[i] * m_shape[i];
        }
        m_shape[0] = shape.data(0);
        return *this;
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
    parse(std::istream &os)
    {
        // TODO
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    write(std::ostream &stream) const
    {
        stream.write((char *)&m_depth, sizeof(m_depth));
        for (size_type *i = m_shape; i != m_shape + m_depth; ++i)
            stream.write((char *)i, sizeof(*m_shape));
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            stream.write((char *)i, sizeof(*m_data));
        return *this;
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    read(std::istream &stream)
    {
        if (m_array_type == Type::Array) {
            clear();
            stream.read((char *)&m_depth, sizeof(m_depth));
            size_type *s = new size_type[m_depth];
            for (size_type *i = s; i != s + m_depth; ++i)
                stream.read((char *)i, sizeof(*s));
            allocate(s, m_depth);
        } else {
            depth_type depth;
            stream.read((char *)&depth, sizeof(m_depth));
            NN_RUNTIME_ERROR(depth != m_depth, "subarray depth not match")
            size_type s;
            for (size_type *i = m_shape; i != m_shape + m_depth; ++i) {
                stream.read((char *)&s, sizeof(s));
                NN_RUNTIME_ERROR(s != *i, "subarray shape not match")
            }
        }
        for (value_type *i = m_data; i != m_data + m_size; ++i)
            stream.read((char *)i, sizeof(*m_data));
        return *this;
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    save_to_file(std::string filepath) const
    {
        std::ofstream file(filepath, std::ios::binary);
        NN_RUNTIME_ERROR(!file, "cannot open file")
        write(file);
        return *this;
    }

    template<typename T>
    const Array<T> &
    Array<T>::
    load_from_file(std::string filepath)
    {
        std::ifstream file(filepath, std::ios::binary);
        NN_RUNTIME_ERROR(!file, "cannot open file")
        read(file);
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
            if (m_data[i] != array.m_data[i])
                return false;
        return true;
    }

    template<typename T>
    Array<T>
    Array<T>::
    operator-() const
    {
        Array<T> result = *this;
        for (auto &i : result)
            i *= -1;
        return result;
    }

#define MX_ARRAY_OPERATION_AA(op, name) \
    template<typename T> \
    Array<T> \
    Array<T>:: \
    operator op (const Array &array) const \
    { \
        assert(m_size == array.m_size && m_depth == array.m_depth); \
        for (depth_type i = 0; i < m_depth; ++i) \
            assert(m_shape[i] == array.m_shape[i]); \
        Array result = *this; \
        NN_GPU_FUNCTION_CALL( \
            (m_device == DeviceType::GPU && array.m_device == DeviceType::GPU), \
            internal::name##_array_array, \
            (result, array) \
        ) \
        return result; \
    }

    MX_ARRAY_OPERATION_AA(+, add)
    MX_ARRAY_OPERATION_AA(-, subtract)
    MX_ARRAY_OPERATION_AA(*, multiply)
    MX_ARRAY_OPERATION_AA(/, divide)

#define MX_ARRAY_OPERATION_AA_EQ(op, mode) \
    template<typename T> \
    Array<T> & \
    Array<T>:: \
    operator op (const Array &array) \
    { \
        assert(m_size == array.m_size && m_depth == array.m_depth); \
        for (depth_type i = 0; i < m_depth; ++i) \
            assert(m_shape[i] == array.m_shape[i]); \
        for (size_type i = 0; i < m_size; ++i) { \
            if (mode) NN_RUNTIME_ERROR(!array.m_data[i], "array: division by zero") \
            m_data[i] op array.m_data[i]; \
        } \
        return *this; \
    }

    MX_ARRAY_OPERATION_AA_EQ(+=, 0)
    MX_ARRAY_OPERATION_AA_EQ(-=, 0)
    MX_ARRAY_OPERATION_AA_EQ(*=, 0)
    MX_ARRAY_OPERATION_AA_EQ(/=, 1)

#define MX_ARRAY_OPERATION_AV(op, mode) \
    template<typename T> \
    Array<T> \
    Array<T>:: \
    operator op (value_type value) const \
    { \
        if (mode) NN_RUNTIME_ERROR(!value, "number: division by zero") \
        Array result = *this; \
        for (auto &i : result) \
            i op##= value; \
        return result; \
    }

    MX_ARRAY_OPERATION_AV(+, 0)
    MX_ARRAY_OPERATION_AV(-, 0)
    MX_ARRAY_OPERATION_AV(*, 0)
    MX_ARRAY_OPERATION_AV(/, 1)

#define MX_ARRAY_OPERATION_AV_EQ(op, mode) \
    template<typename T> \
    Array<T> & \
    Array<T>:: \
    operator op (value_type value) \
    { \
        if (mode) NN_RUNTIME_ERROR(!value, "number: division by zero") \
        for (auto &i : *this) \
            i op value; \
        return *this; \
    }

    MX_ARRAY_OPERATION_AV_EQ(+=, 0)
    MX_ARRAY_OPERATION_AV_EQ(-=, 0)
    MX_ARRAY_OPERATION_AV_EQ(*=, 0)
    MX_ARRAY_OPERATION_AV_EQ(/=, 1)

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

