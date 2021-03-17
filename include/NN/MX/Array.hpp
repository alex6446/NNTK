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

#include "NN/Core/Device.hpp"
#include "NN/Core/Error.hpp"
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
class Array
{
public:

    using size_type      = std::int64_t;
    using depth_type     = std::int16_t;
    using value_type     = T;
    using iterator       = ArrayIterator<Array>;
    using const_iterator = ArrayIterator<const Array>;

    inline static const std::string FileExt = "mxa";
    inline static DeviceType Device = NN_USE_GPU ? DeviceType::GPU : DeviceType::CPU;

    enum class Type { Array, Subarray, Shape };

public:

    static Array empty(const Array<size_type> &shape);
    static Array full(const Array<size_type> &shape, value_type value);
    static Array ones(const Array<size_type> &shape) { return full(shape, 1); }
    static Array zeros(const Array<size_type> &shape) { return full(shape, 0); }
    static Array random(const Array<size_type> &shape, double from=0.0, double to=1.0);
    static Array sequence(const Array<size_type> &shape, double start=1.0, double step=1.0);

    static Array reshape(const Array &array, const Array<size_type> &shape) { return Array(array).reshape(shape); }
    static Array transpose(const Array &array, const Array<depth_type> &order={});
    static Array convolve(const Array &array, const Array &kernel, int padding, int stride);
    static Array dot(const Array &left, const Array &right);
    static Array sum(const Array &array, int axes, bool keepdims=true);
    static Array sum(const Array &left, const Array &right);
    static auto  sum(const Array &array) -> value_type;
    static auto  load_pack(std::ifstream &file) -> std::vector<std::shared_ptr<Array>>;
    static auto  load_pack(std::string filepath) -> std::vector<std::shared_ptr<Array>>;
    static void  save_pack(std::ofstream &file, const std::vector<const Array *> &arrays);
    static void  save_pack(std::string filepath, const std::vector<const Array *> &arrays);
    
public:

    Array();
    Array(const value_type *data, size_type size);
    Array(const std::initializer_list<value_type> &il);
    Array(const std::initializer_list<Array> &il);
    Array(const Array &copy);
    Array(Array &&move);
    Array(value_type *subdata, size_type size, size_type *shape, size_type *strides, depth_type depth, Type array_type);
    ~Array();

    Array & operator=(const Array &array);
    Array & operator=(const std::initializer_list<value_type> &il);
    Array & operator=(const std::initializer_list<Array> &il);

    Array & operator[](size_type index);
    const Array & operator[](size_type index) const;

    template<class ...Types> auto operator()(Types... indices) -> value_type &;
    template<class ...Types> auto operator()(Types... indices) const -> const value_type &;
    template<class ...Types> auto force_get(Types... indices) const -> value_type;
    template<class ...Types> Array & force_add(T value, Types... indices);

    auto data() -> value_type * { return m_data; }
    auto data() const -> const value_type * { return m_data; }
    auto size() const -> const size_type & { return m_size; }
    auto depth() const -> const depth_type & { return m_depth; }

    auto data(size_type index) -> value_type & { assert(index >= 0 && index < m_size); return m_data[index]; }
    auto data(size_type index) const -> const value_type & { assert(index >= 0 && index < m_size); return m_data[index]; }
    auto shape(depth_type index) const -> size_type { assert(index >= 0 && index < m_depth); return m_shape[index]; }
    auto strides(depth_type index) const -> size_type { assert(index >= 0 && index < m_depth); return m_strides[index]; }

    // returns in form of a subarray
    auto shape() const -> const Array<size_type> &;
    auto strides() const -> const Array<size_type> &;

    Array & gpu_sync();

    Array & reshape(const Array<size_type> &shape);
    Array & transpose(const Array<depth_type> &order={}) { return *this = transpose(*this, order); }
    Array & t() const { return transpose(*this); }

    iterator begin() { return iterator(m_data); }
    iterator end() { return iterator(m_data + m_size); }
    const_iterator begin() const { return const_iterator(m_data); }
    const_iterator end() const { return const_iterator(m_data + m_size); }

    const Array & print(std::ostream &os=std::cout) const;
    const Array & parse(std::istream &is=std::cin);

    const Array & write(std::ostream &stream) const;
    const Array & read(std::istream &stream);

    const Array & save_to_file(std::string filepath) const;
    const Array & load_from_file(std::string filepath);

    bool operator==(const Array &array) const;
    bool operator!=(const Array &array) const { return !operator==(array); }

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

    template<class ...Types> auto element_get(Types... indices) -> value_type *;
    template<class ...Types> auto element_get(Types... indices) const -> const value_type * ;

    void elements_print(std::ostream &os, depth_type index, value_type *ptr) const;

public:

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

    mutable std::shared_ptr<Array<size_type>> m_shape_subarray;
    mutable std::shared_ptr<Array<size_type>> m_strides_subarray;
    mutable std::shared_ptr<Array<value_type>> m_data_subarray;

    Type m_array_type = Type::Array;

    DeviceType m_device = Device;

private:

    size_type m_size;
    value_type *m_data = nullptr;

    depth_type m_depth;
    size_type *m_shape = nullptr;
    size_type *m_strides = nullptr;

}; // class Array

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
        m_shape = new size_type[m_depth];
        m_strides = new size_type[m_depth];
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
        delete[] m_shape;
        m_shape = nullptr;
    }
    if (m_strides) {
        delete[] m_strides;
        m_strides = nullptr;
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
    m_shape = new size_type[m_depth];
    m_strides = new size_type[m_depth];
    m_strides[m_depth-1] = 1;
    for (depth_type i = m_depth-1; i > 0; --i) {
        m_shape[i] = shape[i];
        m_strides[i-1] = m_strides[i] * m_shape[i];
    }
    m_shape[0] = shape[0];
    m_size = m_strides[0] * m_shape[0];
    NN_GPU_FUNCTION_CALL(true, NN::internal::allocate_memory, (&m_data, m_size))
}

template<typename T>
template<class ...Types>
auto
Array<T>::
element_get(Types... indices) -> value_type *
{
    const std::array<size_type, sizeof...(indices)> inds = {{indices...}};
    value_type *ptr = m_data;
    for (depth_type i = 0; i < inds.size(); ++i) {
        if (inds[i] < 0 || inds[i] >= m_shape[i])
            return nullptr;
        ptr += m_strides[i] * inds[i];
    }
    return ptr;
}

template<typename T>
template<class ...Types>
auto
Array<T>::
element_get(Types... indices) const -> const value_type *
{
    const std::array<size_type, sizeof...(indices)> inds = {{indices...}};
    value_type *ptr = m_data;
    for (depth_type i = 0; i < inds.size(); ++i) {
        if (inds[i] < 0 || inds[i] >= m_shape[i])
            return nullptr;
        ptr += m_strides[i] * inds[i];
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
    m_strides[0] = 1;
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
    m_strides[0] = 1;
    m_data[0] = 0;
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
    m_strides[0] = il.begin()->m_size;
    for (size_type i = 0, c = 0; i < il.size(); ++i)
        for (size_type j = 0; j < il.begin()->m_size; ++j, ++c)
            m_data[c] = il.begin()[i].m_data[j];
    for (depth_type i = 0; i < il.begin()->m_depth; ++i) {
        m_shape[i+1] = il.begin()->m_shape[i];
        m_strides[i+1] = il.begin()->m_strides[i];
    }
}

template<typename T>
Array<T>::
Array(value_type *data, size_type size, size_type *shape, size_type *strides, depth_type depth, Type array_type)
: m_size(size)
, m_data(data)
, m_depth(depth)
, m_shape(shape)
, m_strides(strides)
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
        m_strides[i] = copy.m_strides[i];
    }
}

template<typename T>
Array<T>::
Array(Array &&move)
: m_size(move.m_size)
, m_data(move.m_data)
, m_depth(move.m_depth)
, m_shape(move.m_shape)
, m_strides(move.m_strides)
{
    move.m_shape = nullptr;
    move.m_strides = nullptr;
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
            m_strides[i] = array.m_strides[i];
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
        m_strides[0] = 1;
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
        m_strides[0] = il.begin()->m_size;
        for (depth_type i = 0; i < il.begin()->m_depth; ++i) {
            m_shape[i+1] = il.begin()->m_shape[i];
            m_strides[i+1] = il.begin()->m_strides[i];
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
                m_data + m_strides[0] * index,
                m_strides[0],
                m_shape + 1,
                m_strides + 1,
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
                m_data + m_strides[0] * index,
                m_strides[0],
                m_shape + 1,
                m_strides + 1,
                m_depth - 1,
                Type::Subarray);
    return *m_data_subarray;
}

template<typename T>
template<class ...Types>
auto
Array<T>::
operator()(Types... indices) -> value_type &
{
    assert(sizeof...(indices) <= m_depth);
    value_type *p = element_get(indices...);
    assert(p);
    return *p;
}

template<typename T>
template<class ...Types>
auto
Array<T>::
operator()(Types... indices) const -> const value_type &
{
    assert(sizeof...(indices) <= m_depth);
    const value_type *p = element_get(indices...);
    assert(p);
    return *p;
}

template<typename T>
template<class ...Types>
auto
Array<T>::
force_get(Types... indices) const -> value_type
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
auto
Array<T>::
shape() const -> const Array<size_type> &
{
    m_shape_subarray = std::make_shared<Array<size_type>>(
                m_shape, m_depth,
                new size_type {m_depth},
                new size_type {1}, 1,
                Array<size_type>::Type::Shape);
    return *m_shape_subarray;
}

template<typename T>
auto
Array<T>::
strides() const -> const Array<size_type> &
{
    m_strides_subarray = std::make_shared<Array<size_type>>(
                m_strides, m_depth,
                new size_type {m_depth},
                new size_type {1}, 1,
                Array<size_type>::Type::Shape);
    return *m_strides_subarray;
}

//template<typename T>
//Array<T> &
//Array<T>::
//gpu_sync()
//{
    //NN_GPU_FUNCTION_CALL(true, NN::internal::sync, ())
//}

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
    delete [] m_strides;
    m_depth = shape.size();
    m_shape = new size_type[m_depth];
    m_strides = new size_type[m_depth];
    m_strides[m_depth-1] = 1;
    for (depth_type i = m_depth-1; i > 0; --i) {
        m_shape[i] = shape.data(i);
        m_strides[i-1] = m_strides[i] * m_shape[i];
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
            elements_print(os, index+1, place+m_strides[index]*i);
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
    Array result = *this;
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

template<typename T>
Array<T>
Array<T>::
empty(const Array<size_type> &shape)
{
    Array result;
    result.allocate(&(*(shape.begin())), shape.size());
    return result;
}

template<typename T>
Array<T>
Array<T>::
random(const Array<size_type> &shape, double from, double to)
{
    Array result = empty(shape);
    for (auto &i : result)
        i = (T)((double)std::rand() / (double)RAND_MAX * (to-from) + from);
    return result;
}

template<typename T>
Array<T>
Array<T>::
sequence(const Array<size_type> &shape, double start, double step)
{
    Array result = empty(shape);
    double current = start;
    for (auto &i : result)
        i = (T)current;
    return result;
}

template<typename T>
Array<T>
Array<T>::
full(const Array<size_type> &shape, value_type value)
{
    Array result = empty(shape);
    for (auto &i : result)
        i = value;
    return result;
}

template<typename T>
Array<T>
Array<T>::
dot(const Array &left, const Array &right)
{
    assert(left.depth() == 2 && right.depth() == 2);
    assert(left.shape(1) == right.shape(0));
    Array result = zeros({left.shape(0), right.shape(1)});
    for (value_type *di = result.data(), *li = (value_type *)left.data();
            li != left.data() + left.size();
            di += result.strides(0), li += left.strides(0))
        for (size_type j = 0; j < right.shape(1); ++j) {
            const value_type *rk = right.data();
            for (size_type k = 0; k < left.shape(1); ++k, rk += right.strides(0))
                *(di+j) += *(li+k) * *(rk+j);
        }
    return result;
}

template<typename T>
Array<T>
Array<T>::
convolve(const Array &array, const Array &kernel, int padding, int stride)
{
    assert(array.depth() == 2 && kernel.depth() == 2);
    assert(stride >= 1);

    size_type cm = (array.shape(0) + 2*padding - kernel.shape(0)) / stride + 1;
    size_type cn = (array.shape(1) + 2*padding - kernel.shape(1)) / stride + 1;
    assert(cm > 0 && cn > 0);

    Array c = zeros({cm, cn});
    for (size_type ai = -padding, ci = 0; ci < cm; ai+=stride, ++ci)
        for (size_type aj = -padding, cj = 0; cj < cn; aj+=stride, ++cj)
            for (size_type ki = 0; ki < kernel.shape(0); ++ki)
                for (size_type kj = 0; kj < kernel.shape(1); ++kj)
                    c(ci, cj) += kernel(ki, kj) * array.force_get(ai+ki, aj+kj);
    return c;
}

template<typename T>
Array<T>
Array<T>::
sum(const Array &array, int axes, bool keepdims)
{
    depth_type depth = keepdims ? array.depth() : array.depth()-1;
    auto shape = Array<size_type>::empty({depth});
    depth_type index = 0;
    for (depth_type i = 0; i < array.depth(); ++i) {
        if (i == axes) {
            if (keepdims)
                shape.data(index++) = 1;
        } else {
            shape.data(index++) = array.shape(i);
        }
    }
    Array s = zeros(shape);
    index = 0;
    size_type ds = array.strides(axes)*array.shape(axes);
    // Legacy code
    for (size_type i = 0; i < array.size()/ds; ++i) {
        for (size_type k = i*ds; k < array.strides(axes) + i*ds; ++k) {
            for (size_type j = k; j < ds+k; j+=array.strides(axes))
                s.data(index) += array.data(j);
            index++;
        }
    }
    return s;
}

template<typename T>
auto
Array<T>::
sum(const Array &array) -> value_type
{
    value_type s = 0;
    for (auto i = array.data(); i != array.data() + array.size(); ++i)
        s += *i;
    return s;
}

template<typename T>
Array<T>
Array<T>::
sum(const Array &left, const Array &right)
{
    assert(left.data() && right.data());
    assert(left.depth() >= right.depth());
    assert(left.size() % right.size() == 0);
    Array s = left;
    for (size_type i = 0, index = 0; i < s.size(); ++i, ++index) {
        if (index >= right.size())
            index = 0;
        s.data(i) += right.data(index);
    }
    return s;
}

template<typename T>
Array<T>
Array<T>::
transpose(const Array &array, const Array<depth_type> &order)
{
    assert(order.size() == array.depth() || order.depth() == 0);
    // Array of depth_type because its not the elements count but the index
    auto ord = order.depth() ? order: Array<depth_type>::sequence({array.depth()}, array.depth()-1, -1);
    std::vector<value_type *> ptrs(array.depth(), (value_type *)array.data());
    std::vector<size_type> inds(array.depth(), 0);
    Array<size_type> shape = array.shape();
    for (depth_type i = 0; i < array.depth(); ++i)
        shape.data(i) = array.shape(ord.data(i));
    Array result = empty(shape);
    for (auto it = result.begin(); it != result.end(); ++it) {
        *it = *(ptrs.back());
        if (&(*it) != &(*(result.end()))-1) {
            depth_type last_dim = array.depth()-1;
            while (inds[last_dim]+1 >= array.shape(ord.data(last_dim)))
                last_dim--;
            inds[last_dim]++;
            ptrs[last_dim] += array.strides(ord.data(last_dim));
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
Array<T>::
save_pack(std::ofstream &file, const std::vector<const Array *> &arrays)
{
    NN_RUNTIME_ERROR(!file, "caNNot open a file")
    std::size_t arrays_count = arrays.size();
    file.write((char *)&arrays_count, sizeof(arrays.size()));
    for (auto *i : arrays)
        i->write(file);
}

template<typename T>
void
Array<T>::
save_pack(std::string filepath, const std::vector<const Array *> &arrays)
{
    std::ofstream file(filepath, std::ios::binary);
    save_pack(file, arrays);
}

template<typename T>
auto
Array<T>::
load_pack(std::ifstream &file) -> std::vector<std::shared_ptr<Array>>
{
    NN_RUNTIME_ERROR(!file, "caNNot open a file")
    std::size_t arrays_count;
    file.read((char *)&arrays_count, sizeof(arrays_count));
    std::vector<std::shared_ptr<Array>> arrays(arrays_count);
    for (std::size_t i = 0; i < arrays_count; ++i) {
        arrays[i] = std::make_shared<Array>();
        arrays[i]->read(file);
    }
    return arrays;
}

template<typename T>
auto
Array<T>::
load_pack(std::string filepath) -> std::vector<std::shared_ptr<Array>>
{
    std::ifstream file(filepath, std::ios::binary);
    return load_pack(file);
}

} // namespace MX

} // namespace NN

