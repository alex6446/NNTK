#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include "NNTK/Core/Device.hpp"
#include "NNTK/Core/Error.hpp"
#include "NNTK/Core/Memory.hpp"

#include "../Array.hpp"

namespace NN::MX
{

  template<typename T, Device D>
    void
    Array<T, D>::
    allocate()
    {
      if (m_size)
        NN_GPU_FUNCTION_CALL(D == GPU, NN::internal::allocate_memory,
                             (&m_data, m_size))
      if (m_depth) {
        m_shape = new size_type[m_depth];
        m_strides = new size_type[m_depth];
      }
    }

  template<typename T, Device D>
    void
    Array<T, D>::
    free()
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
      if (m_array_type == Type::Size)
        return;
      if (m_data) {
        NN_GPU_FUNCTION_CALL(D == GPU, NN::internal::free_memory, (&m_data))
        m_data = nullptr;
      }
      m_size = 0;
      m_depth = 0;
    }

  template<typename T, Device D>
    void
    Array<T, D>::
    allocate(const size_type* shape, depth_type depth)
    {
      if (depth == 0) return;
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
      NN_GPU_FUNCTION_CALL(D == GPU, NN::internal::allocate_memory,
                           (&m_data, m_size))
    }

  template<typename T, Device D>
  template<class ...Types>
    auto
    Array<T, D>::
    element_get(Types... indices) const -> const value_type*
    {
      assert(sizeof...(indices) <= m_depth);
      const std::array<size_type, sizeof...(indices)> inds = {{indices...}};
      value_type* ptr = m_data;
      for (depth_type i = 0; i < inds.size(); ++i) {
        if (inds[i] < 0 || inds[i] >= m_shape[i])
          return nullptr;
        ptr += m_strides[i] * inds[i];
      }
      return ptr;
    }

  template<typename T, Device D>
    Array<T, D>::
    Array() : m_size(0), m_depth(0)
    { }

  template<typename T, Device D>
    Array<T, D>::
    Array(const value_type* data, size_type size)
    : m_size(size), m_depth(1)
    {
      assert(m_size >= 0);
      allocate();
      m_shape[0] = m_size;
      m_strides[0] = 1;
      ::NN::internal::copy_memory_cpu(m_data, data,
                                      m_size * sizeof(value_type));
    }

  template<typename T, Device D>
    Array<T, D>::
    Array(const std::initializer_list<value_type>& il)
    : m_size(il.size()), m_depth(1)
    {
      allocate();
      m_shape[0] = il.size();
      m_strides[0] = 1;
      ::NN::internal::copy_memory_cpu(m_data, il.begin(),
                                      m_size * sizeof(value_type));
    }

  template<typename T, Device D>
    Array<T, D>::
    Array(const std::initializer_list<Array>& il)
    : m_size(il.begin()->m_size * il.size()),
      m_depth(il.begin()->m_depth + 1)
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

  template<typename T, Device D>
    Array<T, D>::
    Array(value_type* data, size_type size,
          size_type* shape, size_type* strides,
          depth_type depth, Type array_type, Device device)
    : m_size(size), m_data(data), m_depth(depth), m_shape(shape),
      m_strides(strides), m_array_type(array_type), m_device(device)
    { }

  template<typename T, Device D>
    Array<T, D>::
    Array(const Array<T, D>& copy)
    : m_size(copy.m_size), m_depth(copy.m_depth), m_device(copy.m_device)
    {
      std::cout << D << " same copy" << std::endl;
      allocate();
      ::NN::internal::copy_memory_cpu(m_shape, copy.m_shape,
                                      m_depth * sizeof(size_type));
      ::NN::internal::copy_memory_cpu(m_strides, copy.m_strides,
                                      m_depth * sizeof(size_type));
      NN_GPU_FUNCTION_CALL(D == GPU, NN::internal::copy_memory,
                           (m_data, copy.m_data, m_size * sizeof(value_type)))
    }

  template<typename T, Device D>
    Array<T, D>::
    Array(const Array<T, nD>& copy)
    : m_size(copy.m_size), m_depth(copy.m_depth), m_device(D)
    {
      std::cout << D << " diff copy" << std::endl;
      allocate();
      ::NN::internal::copy_memory_cpu(m_shape, copy.m_shape,
                                      m_depth * sizeof(size_type));
      ::NN::internal::copy_memory_cpu(m_strides, copy.m_strides,
                                      m_depth * sizeof(size_type));
      NN_GPU_FUNCTION_CALL(true, NN::internal::copy_memory,
                           (m_data, copy.m_data, 
                            m_size * sizeof(value_type)))
    }

  template<typename T, Device D>
    Array<T, D>::
    Array(Array<T, D>&& move)
    : m_size(move.m_size), m_data(move.m_data), m_depth(move.m_depth),
      m_shape(move.m_shape), m_strides(move.m_strides),
      m_device(move.m_device)
    {
      std::cout << "same move " << D << std::endl;
      move.m_shape = nullptr;
      move.m_strides = nullptr;
      move.m_data = nullptr;
    }

  template<typename T, Device D>
    Array<T, D>::
    Array(Array<T, nD>&& move)
    : m_size(move.m_size), m_data(move.m_data), m_depth(0),
      m_shape(move.m_shape), m_strides(move.m_strides), m_device(D)
    {
      std::cout << "diff move " << D << this
                << " from " << nD << &move << std::endl;
      allocate();
      m_depth = move.m_depth;
      NN_GPU_FUNCTION_CALL(true, NN::internal::copy_memory,
                           (m_data, move.m_data, m_size * sizeof(value_type)))
      move.m_shape = nullptr;
      move.m_strides = nullptr;
    }

  template<typename T, Device D>
    Array<T, D>&
    Array<T, D>::
    operator=(const Array<T, D>& array)
    {
      if (m_array_type == Type::Array) {
        free();
        m_size = array.m_size;
        m_depth = array.m_depth;
        allocate();
        ::NN::internal::copy_memory_cpu(m_shape, array.m_shape,
                                        m_depth * sizeof(size_type));
        ::NN::internal::copy_memory_cpu(m_strides, array.m_strides,
                                        m_depth * sizeof(size_type));
      } else {
        assert(m_depth == array.m_depth);
        assert(std::equal(m_shape, m_shape + m_depth, array.m_shape));
      }
      NN_GPU_FUNCTION_CALL(D == GPU, NN::internal::copy_memory,
                           (m_data, array.m_data, m_size * sizeof(value_type)))
      return *this;
    }

  template<typename T, Device D>
    Array<T, D>&
    Array<T, D>::
    operator=(const Array<T, nD>& array)
    {
      if (m_array_type == Type::Array) {
        free();
        m_size = array.m_size;
        m_depth = array.m_depth;
        allocate();
        ::NN::internal::copy_memory_cpu(m_shape, array.m_shape,
                                        m_depth * sizeof(size_type));
        ::NN::internal::copy_memory_cpu(m_strides, array.m_strides,
                                        m_depth * sizeof(size_type));
      } else {
        assert(m_depth == array.m_depth);
        assert(std::equal(m_shape, m_shape + m_depth, array.m_shape));
      }
      NN_GPU_FUNCTION_CALL(true, NN::internal::copy_memory,
                           (m_data, array.m_data, m_size * sizeof(value_type)))
      return *this;
    }

  template<typename T, Device D>
    Array<T, D>&
    Array<T, D>::
    operator=(const std::initializer_list<T>& il)
    {
      if (m_array_type == Type::Array) {
        free();
        m_size = il.size();
        m_depth = 1;
        allocate();
        m_shape[0] = il.size();
        m_strides[0] = 1;
      } else {
        assert(m_depth == 1);
        assert(m_shape[0] == il.size());
      }
      ::NN::internal::copy_memory_cpu(m_data, il.begin(),
                                      m_size * sizeof(value_type));
      return *this;
    }

  template<typename T, Device D>
    Array<T, D>&
    Array<T, D>::
    operator=(const std::initializer_list<Array>& il)
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
        assert(std::equal(m_shape + 1, m_shape + 1 + m_depth,
                          il.begin()->m_shape));
      }
      for (size_type i = 0, c = 0; i < il.size(); ++i)
        for (size_type j = 0; j < il.begin()->m_size; ++j, ++c)
          m_data[c] = (il.begin() + i)->m_data[j];
      return *this;
    }

  template<typename T, Device D>
    Array<T, D>&
    Array<T, D>::
    operator[](size_type index)
    {
      assert(m_depth > 1);
      assert(index >= 0);
      assert(index < m_shape[0]);
      return *(m_data_subarray =
          std::make_shared<Array<value_type>>(
                               m_data + m_strides[0] * index,
                               m_strides[0],
                               m_shape + 1,
                               m_strides + 1,
                               m_depth - 1,
                               Type::Subarray,
                               m_device));
    }

  template<typename T, Device D>
    const Array<T, D>&
    Array<T, D>::
    operator[](size_type index) const
    {
      assert(m_depth > 1);
      assert(index >= 0);
      assert(index < m_shape[0]);
      return *(m_data_subarray =
          std::make_shared<Array<value_type>>(
                               m_data + m_strides[0] * index,
                               m_strides[0],
                               m_shape + 1,
                               m_strides + 1,
                               m_depth - 1,
                               Type::Subarray));
    }

  template<typename T, Device D>
    auto
    Array<T, D>::
    shape() const -> const Array<size_type, CPU>&
    {
      return *(m_shape_subarray =
          std::make_shared<Array<size_type, CPU>>(
                               m_shape, m_depth,
                               new size_type {m_depth},
                               new size_type {1}, 1,
                               Array<size_type, CPU>::Type::Size,
                               CPU));
    }

  template<typename T, Device D>
    auto
    Array<T, D>::
    strides() const -> const Array<size_type, CPU>&
    {
      return * (m_strides_subarray =
          std::make_shared<Array<size_type, CPU>>(
                               m_strides, m_depth,
                               new size_type {m_depth},
                               new size_type {1}, 1,
                               Array<size_type, CPU>::Type::Size,
                               CPU));
    }

  //template<typename T, Device D>
  //Array<T> &
  //Array<T>::
  //gpu_sync()
  //{
      //NN_GPU_FUNCTION_CALL(true, NN::internal::sync, ())
  //}

  template<typename T, Device D>
    Array<T, D>&
    Array<T, D>::
    reshape(const Array<size_type>& shape)
    {
      assert(m_array_type == Type::Array);
      size_type size = std::accumulate(shape.begin(), shape.end(), 1,
                                       std::multiplies<>());
      assert(size == m_size);
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

  template<typename T, Device D>
    const Array<T, D>&
    Array<T, D>::
    print(std::ostream &os) const
    {
      if (!m_data) os << "{}";
      else elements_print(os, 0, m_data);
      return *this;
    }

  template<typename T, Device D>
    const Array<T, D>&
    Array<T, D>::
    parse(std::istream &os)
    {
        // TODO
    }

  template<typename T, Device D>
    const Array<T, D>&
    Array<T, D>::
    write(std::ostream& stream) const
    {
      stream.write((char*) &m_depth, sizeof(m_depth));
      for (size_type* i = m_shape; i != m_shape + m_depth; ++i)
        stream.write((char*) i, sizeof(*m_shape));
      for (value_type* i = m_data; i != m_data + m_size; ++i)
        stream.write((char*) i, sizeof(*m_data));
      return *this;
    }

  template<typename T, Device D>
    const Array<T, D>&
    Array<T, D>::
    read(std::istream& stream)
    {
      if (m_array_type == Type::Array) {
        free();
        stream.read((char*) &m_depth, sizeof(m_depth));
        size_type* shape = new size_type[m_depth];

        std::for_each(shape, shape + m_depth, [&](size_type& e)
            { stream.read((char*) &e, sizeof(*shape)); });
        allocate(shape, m_depth);
      } else {
        depth_type depth;
        stream.read((char*) &depth, sizeof(m_depth));
        NN_RUNTIME_ERROR(depth != m_depth, "subarray depth not match")
        size_type dim;

        std::for_each(m_shape, m_shape + m_depth, [&](size_type& e) {
            stream.read((char*) &dim, sizeof(dim));
            NN_RUNTIME_ERROR(dim != e, "subarray shape not match")
        });
      }
      this->for_each([&](value_type& e)
          { stream.read((char*) &e, sizeof(*m_data)); });
      return *this;
    }

  template<typename T, Device D>
    const Array<T, D>&
    Array<T, D>::
    save_to_file(const std::string& filepath) const
    {
      std::ofstream file(filepath, std::ios::binary);
      NN_RUNTIME_ERROR(!file, "cannot open file")
      write(file);
      return *this;
    }

  template<typename T, Device D>
    const Array<T, D>&
    Array<T, D>::
    load_from_file(const std::string& filepath)
    {
      std::ifstream file(filepath, std::ios::binary);
      NN_RUNTIME_ERROR(!file, "cannot open file")
      read(file);
      return *this;
    }

  template<typename T, Device D>
    void
    Array<T, D>::
    elements_print(std::ostream& os, depth_type index, value_type* place) const
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

  template<typename T, Device D>
    bool
    Array<T, D>::
    operator==(const Array& array) const
    {
        return (m_size == array.m_size || m_depth == array.m_depth) &&
               std::equal(m_shape, m_shape + m_depth, array.m_shape) &&
               std::equal(m_data, m_data + m_size, array.m_data);
    }

#define MX_ARRAY_ROPERATION_AA_EQ(op, name)                                   \
  template<typename T, Device D>                                              \
  template<Device V>                                                          \
    Array<T, D>&                                                              \
    Array<T, D>::                                                             \
    operator op (const Array<T, V>& array)                                    \
    {                                                                         \
      assert(m_size == array.m_size);                                         \
      assert(m_depth == array.m_depth);                                       \
      assert(std::equal(m_shape, m_shape + m_depth, array.m_shape));          \
      NN_GPU_FUNCTION_CALL((m_device == GPU && array.m_device == GPU),        \
                           internal::name##_array_array, (*this, array))      \
      return *this;                                                           \
    }

  MX_ARRAY_ROPERATION_AA_EQ(+=, add)
  MX_ARRAY_ROPERATION_AA_EQ(*=, multiply)
  MX_ARRAY_ROPERATION_AA_EQ(-=, subtract)
  MX_ARRAY_ROPERATION_AA_EQ(/=, divide)

#define MX_ARRAY_OPERATION_AV_EQ(op, mode, name)                              \
  template<typename T, Device D>                                              \
    Array<T, D>&                                                              \
    Array<T, D>::                                                             \
    operator op (const value_type &value)                                     \
    {                                                                         \
      if (mode)                                                               \
        NN_RUNTIME_ERROR(!value, "number: division by zero")                  \
      NN_GPU_FUNCTION_CALL((m_device == GPU),                                 \
                           internal::name##_array_value, (*this, value))      \
      return *this;                                                           \
    }

  MX_ARRAY_OPERATION_AV_EQ(+=, 0, add)
  MX_ARRAY_OPERATION_AV_EQ(*=, 0, multiply)
  MX_ARRAY_OPERATION_AV_EQ(-=, 0, subtract)
  MX_ARRAY_OPERATION_AV_EQ(/=, 1, divide)

  // static implementation

  template<typename T, Device D>
    std::ostream&
    operator<<(std::ostream& os, const Array<T, D>& array)
    {
      array.print(os);
      return os;
    }

  template<typename T, Device D>
    Array<T, D>
    Array<T, D>::
    dot(const Array& left, const Array& right)
    {
      assert(left.depth() == 2);
      assert(right.depth() == 2);
      assert(left.shape(1) == right.shape(0));
      Array result = zeros({left.shape(0), right.shape(1)});

      for (value_type *di = result.data(), *li = (value_type*) left.data();
           li != left.data() + left.size();
           di += result.strides(0), li += left.strides(0)) {
        for (size_type j = 0; j < right.shape(1); ++j) {
          const value_type* rk = right.data();
          for (size_type k = 0; k < left.shape(1); ++k, rk += right.strides(0))
            *(di+j) += *(li+k) * *(rk+j);
        }
      }
      return result;
    }

  template<typename T, Device D>
    Array<T, D>
    Array<T, D>::
    convolve(const Array& array, const Array& kernel, int padding, int stride)
    {
      assert(array.depth() == 2);
      assert(kernel.depth() == 2);
      assert(stride >= 1);
      size_type cm =
          (array.shape(0) + 2*padding - kernel.shape(0)) / stride + 1;
      size_type cn =
          (array.shape(1) + 2*padding - kernel.shape(1)) / stride + 1;
      assert(cm > 0 && cn > 0);

      Array c = zeros({cm, cn});
      for (size_type ai = -padding, ci = 0; ci < cm; ai+=stride, ++ci)
        for (size_type aj = -padding, cj = 0; cj < cn; aj+=stride, ++cj)
          for (size_type ki = 0; ki < kernel.shape(0); ++ki)
            for (size_type kj = 0; kj < kernel.shape(1); ++kj)
              c(ci, cj) += kernel(ki, kj) * array.force_get(ai+ki, aj+kj);
      return c;
    }

  template<typename T, Device D>
    Array<T, D>
    Array<T, D>::
    sum(const Array& array, int axes, bool keepdims)
    {
      depth_type depth = keepdims ? array.depth() : array.depth()-1;
      auto shape = Array<size_type>::empty({depth});
      depth_type index = 0;
      for (depth_type i = 0; i < array.depth(); ++i) {
        if (i != axes) shape.data(index++) = array.shape(i);
        else if (keepdims) shape.data(index++) = 1;
      }
      Array s = zeros(shape);
      index = 0;
      size_type ds = array.strides(axes) * array.shape(axes);
      // Legacy code
      for (size_type i = 0; i < array.size()/ds; ++i) {
        for (size_type k = i*ds; k < array.strides(axes) + i*ds; ++k) {
          for (size_type j = k; j < ds+k; j+=array.strides(axes))
            s.data(index) += array.data(j);
          ++index;
        }
      }
      return s;
    }

  template<typename T, Device D>
    Array<T, D>
    Array<T, D>::
    sum(const Array& left, const Array& right)
    {
      assert(left.data());
      assert(right.data());
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

  template<typename T, Device D>
    Array<T, D>
    Array<T, D>::
    transpose(const Array& array, const Array<depth_type>& order)
    {
      assert(order.size() == array.depth() || order.depth() == 0);
      // Array of depth_type because its not the elements count but the index
      auto ord = order.depth() ? order : Array<depth_type>::sequence(
                                             {array.depth()},
                                             array.depth()-1, -1);
      std::vector<value_type*> ptrs(array.depth(),
                                    (value_type*) array.data());
      std::vector<size_type> inds(array.depth(), 0);

      Array<size_type, CPU> shape = array.shape();
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

  template<typename T, Device D>
    void
    Array<T, D>::
    save_pack(std::ofstream& file, const std::vector<const Array*>& arrays)
    {
      NN_RUNTIME_ERROR(!file, "caNNot open a file")
      std::size_t arrays_count = arrays.size();
      file.write((char*) &arrays_count, sizeof(arrays.size()));
      for (auto* i : arrays)
        i->write(file);
    }

  template<typename T, Device D>
    std::vector<std::shared_ptr<Array<T, D>>>
    Array<T, D>::
    load_pack(std::ifstream& file)
    {
      NN_RUNTIME_ERROR(!file, "caNNot open a file")
      std::size_t arrays_count;
      file.read((char*) &arrays_count, sizeof(arrays_count));
      std::vector<std::shared_ptr<Array>> arrays(arrays_count);
      for (std::size_t i = 0; i < arrays_count; ++i) {
        arrays[i] = std::make_shared<Array>();
        arrays[i]->read(file);
      }
      return arrays;
    }

} // namespace NN::MX


