#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
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

#include "Array/Predefinition.hpp"
#include "Array/Iterator.hpp"

#include "Array/ArithmeticOperations.hpp"

namespace NN::MX
{

  template<typename T, Device D>
    class Array
    {
    public: // Types

      using size_type      = std::int64_t;
      using depth_type     = std::int16_t;
      using value_type     = T;
      using iterator       = Iterator<value_type>;
      using const_iterator = Iterator<const value_type>;

      inline static const Device nD = (Device) !D;
      inline static const std::string FileExt = "mxa";

      friend class Array<T, nD>;

      enum class Type { Array, Subarray, Size };

    public: // Static functions

      static Array
      empty(const Array<size_type>& shape)
      {
        Array result;
        result.allocate(&(*(shape.begin())), shape.size());
        return result;
      }

      static Array
      full(const Array<size_type>& shape, value_type value)
      { return empty(shape).for_each([&](value_type& e) { e = value; }); }

      static Array
      ones(const Array<size_type>& shape)
      { return full(shape, 1); }

      static Array
      zeros(const Array<size_type>& shape)
      { return full(shape, 0); }

      static Array
      random(const Array<size_type>& shape, double from=0.0, double to=1.0)
      {
        return empty(shape).for_each([&](value_type& e)
            { e = (value_type) ((double) std::rand() /
                                (double) RAND_MAX * (to-from) + from); });
      }

      static Array
      sequence(const Array<size_type>& shape,
               double start=1.0, double step=1.0)
      {
        double current = start - step;
        return empty(shape).for_each([&](value_type& e)
            { e = (value_type) (current += step); });
      }


      static Array
      reshape(Array&& array, const Array<size_type> &shape)
      { return std::move(array.reshape(shape)); }

      static Array
      reshape(const Array& array, const Array<size_type> &shape)
      { return reshape(Array(array), shape); }

      static Array
      transpose(const Array &array, const Array<depth_type> &order={});

      template<typename Function>
      static Array
      for_each(Array&& array, Function function)
      { return std::move(array.for_each(function)); }

      template<typename Function>
      static Array
      for_each(const Array& array, Function function)
      { return for_each(Array(array), function); }

      static Array
      convolve(const Array& array, const Array& kernel,
               int padding, int stride);

      static Array
      dot(const Array& left, const Array& right);

      static Array
      sum(const Array& array, int axes, bool keepdims=true);

      static Array
      sum(const Array& left, const Array& right);

      static value_type 
      sum(const Array& array)
      { return std::accumulate(array.begin(), array.end(), 0); }


      static std::vector<std::shared_ptr<Array>> 
      load_pack(std::ifstream& file);

      static std::vector<std::shared_ptr<Array>> 
      load_pack(const std::string& filepath)
      {
        std::ifstream file(filepath, std::ios::binary);
        return load_pack(file);
      }

      static void 
      save_pack(std::ofstream& file,
                const std::vector<const Array*>& arrays);

      static void 
      save_pack(const std::string& filepath,
                const std::vector<const Array*>& arrays)
      {
        std::ofstream file(filepath, std::ios::binary);
        save_pack(file, arrays);
      }

    public: // Public functions

      /////////////////////////////////////////////////////////////////////////
      // Construction
      /////////////////////////////////////////////////////////////////////////

      Array();
      Array(const value_type* data, size_type size);
      Array(const std::initializer_list<value_type>& il);
      Array(const std::initializer_list<Array>& il);
      Array(const Array<T, D>& copy);
      Array(const Array<T, nD>& copy);
      Array(Array<T, D>&& move);
      Array(Array<T, nD>&& move);
      Array(value_type* subdata, size_type size,
            size_type* shape, size_type* strides,
            depth_type depth, Type array_type, Device device_type);
      virtual ~Array() { free(); }

      Array&
      operator=(const Array<T, D>& array);

      Array&
      operator=(const Array<T, nD>& array);

      Array&
      operator=(const std::initializer_list<value_type>& il);

      Array&
      operator=(const std::initializer_list<Array>& il);

      /////////////////////////////////////////////////////////////////////////
      // Accessors & Modifiers
      /////////////////////////////////////////////////////////////////////////

      Array&
      operator[](size_type index);

      const Array&
      operator[](size_type index) const;

      template<class ...Types>
        value_type&
        operator()(Types... indices)
        {
          const value_type* p = element_get(indices...);
          assert(p);
          return *(value_type*) p;
        }

      template<class ...Types>
        const value_type&
        operator()(Types... indices) const
        {
          const value_type* p = element_get(indices...);
          assert(p);
          return * p;
        }

      template<class ...Types>
        value_type
        force_get(Types... indices) const
        {
          const value_type* p = element_get(indices...);
          return p ? *p : 0;
        }

      template<class ...Types>
        Array&
        force_add(T value, Types... indices)
        {
          const value_type* p = element_get(indices...);
          if (p != nullptr)
            *p += value;
          return *this;
        }

      /////////////////////////////////////////////////////////////////////////
      // Capacity & Data
      /////////////////////////////////////////////////////////////////////////

      const size_type&
      size() const
      { return m_size; }

      const depth_type&
      depth() const
      { return m_depth; }


      value_type*
      data()
      { return m_data; }

      const value_type*
      data() const
      { return m_data; }

      value_type&
      data(size_type index)
      {
        assert(index >= 0);
        assert(index < m_size);
        return m_data[index];
      }

      const value_type&
      data(size_type index) const
      {
        assert(index >= 0);
        assert(index < m_size);
        return m_data[index];
      }


      /// returns in form of a subarray
      const Array<size_type, CPU>&
      shape() const;

      size_type
      shape(depth_type index) const
      {
        assert(index >= 0);
        assert(index < m_depth);
        return m_shape[index];
      }


      /// returns in form of a subarray
      const Array<size_type, CPU>&
      strides() const;

      size_type
      strides(depth_type index) const
      {
        assert(index >= 0);
        assert(index < m_depth);
        return m_strides[index];
      }


      constexpr Device
      device() const
      { return m_device; }

      Array&
      gpu_sync();

      /////////////////////////////////////////////////////////////////////////
      // Transform
      /////////////////////////////////////////////////////////////////////////

      Array&
      reshape(const Array<size_type>& shape);

      Array&
      transpose(const Array<depth_type> &order={})
      { return *this = transpose(*this, order); }

      Array&
      t() const
      { return transpose(*this); }

      template<typename Function>
      Array&
      for_each(Function function)
      {
        std::for_each(m_data, m_data + m_size, function);
        return *this;
      }

      /////////////////////////////////////////////////////////////////////////
      // Iterators
      /////////////////////////////////////////////////////////////////////////

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

      /////////////////////////////////////////////////////////////////////////
      // Streams
      /////////////////////////////////////////////////////////////////////////

      const Array&
      print(std::ostream& os=std::cout) const;

      const Array&
      parse(std::istream& is=std::cin);

      const Array&
      write(std::ostream& stream) const;

      const Array&
      read(std::istream& stream);

      const Array&
      save_to_file(const std::string& filepath) const;

      const Array&
      load_from_file(const std::string& filepath);

      /////////////////////////////////////////////////////////////////////////
      // Arithmetic operators
      /////////////////////////////////////////////////////////////////////////

      bool
      operator==(const Array& array) const;

      bool
      operator!=(const Array& array) const
      { return !operator==(array); }

      /// -rvalue
      Array
      operator-() &&
      { return std::move(this->for_each([&](value_type& e) { e = -e; })); }

      /// -lvalue (creates a copy)
      Array
      operator-() const &
      { return Array(*this).operator-(); }

      /**
       * @brief Overload of binary operator +=
       *
       * @param array Array to add of the same type (rvalue/lvalue)
       * @return Read/write reference of current object.
       *
       * Adds \p array to current object. Does not depend on device type.
       */
      template<Device V>
        Array&
        operator+=(const Array<T, V>& array);

      /// rvalue + lvalue (works as rvalue += lvalue)
      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator+(const Array<T, V>& array) &&
        { return std::move(operator+=(array)); }

      /// rvalue + rvalue (works as rvalue += lvalue)
      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator+(Array<T, V>&& array) &&
        { return std::move(operator+=(array)); }

      /// lvalue + rvalue (works as rvalue += lvalue)
      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator+(Array<T, V>&& array) const &
        { return std::move(array.operator+=(*this)); }

      /// lvalue + lvalue (creates a copy)
      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator+(const Array<T, V>& array) const &
        { return ArrayPrefCPU<T, D, V>(*this).operator+(array); }


      template<Device V>
        Array&
        operator*=(const Array<T, V>& array);

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator*(const Array<T, V>& array) &&
        { return std::move(operator*=(array)); }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator*(Array<T, V>&& array) &&
        { return std::move(operator*=(array)); }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator*(Array<T, V>&& array) const &
        { return std::move(array.operator*=(*this)); }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator*(const Array<T, V>& array) const &
        { return ArrayPrefCPU<T, D, V>(*this).operator*(array); }


      template<Device V>
        Array&
        operator-=(const Array<T, V>& array);

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator-(const Array<T, V>& array) &&
        { return std::move(operator-=(array)); }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator-(Array<T, V>&& array) &&
        { return std::move(operator-=(array)); }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator-(Array<T, V>&& array) const &
        {
          assert(m_size == array.m_size);
          assert(m_depth == array.m_depth);
          assert(std::equal(m_shape, m_shape + m_depth, array.m_shape));
          NN_GPU_FUNCTION_CALL((m_device == GPU && array.m_device == GPU),
                               internal::subtract_array_array_reverse,
                               (array, *this))
          return std::move(array);
        }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator-(const Array<T, V>& array) const &
        { return ArrayPrefCPU<T, D, V>(*this).operator-(array); }


      template<Device V>
        Array&
        operator/=(const Array<T, V>& array);

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator/(const Array<T, V>& array) &&
        { return std::move(operator/=(array)); }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator/(Array<T, V>&& array) &&
        { return std::move(operator/=(array)); }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator/(Array<T, V>&& array) const &
        {
          assert(m_size == array.m_size);
          assert(m_depth == array.m_depth);
          assert(std::equal(m_shape, m_shape + m_depth, array.m_shape));
          NN_GPU_FUNCTION_CALL((m_device == GPU && array.m_device == GPU),
                               internal::divide_array_array_reverse,
                               (array, *this))
          return std::move(array);
        }

      template<Device V>
        ArrayPrefCPU<T, D, V>
        operator/(const Array<T, V>& array) const &
        { return ArrayPrefCPU<T, D, V>(*this).operator/(array); }


      Array &
      operator+=(const value_type& value);

      Array
      operator+(const value_type& value) &&
      { return std::move(operator+=(value)); }

      Array
      operator+(const value_type& value) const &
      { return Array(*this).operator+(value); }

      friend Array
      operator+(const value_type& value, Array &&array)
      { return std::move(array.operator+=(value)); }

      friend Array
      operator+(const value_type& value, const Array &array)
      { return value + Array(array); }


      Array &
      operator*=(const value_type& value);

      Array
      operator*(const value_type& value) &&
      { return std::move(operator*=(value)); }

      Array
      operator*(const value_type& value) const &
      { return Array(*this).operator*(value); }

      friend Array
      operator*(const value_type& value, Array&& array)
      { return std::move(array.operator*=(value)); }

      friend Array
      operator*(const value_type& value, const Array& array)
      { return value * Array(array); }


      Array &
      operator-=(const value_type& value);

      Array
      operator-(const value_type& value) &&
      { return std::move(operator-=(value)); }

      Array
      operator-(const value_type& value) const &
      { return Array(*this).operator-(value); }

      friend Array
      operator-(const value_type& value, Array &&array)
      {
        NN_GPU_FUNCTION_CALL((array.m_device == GPU),
                             internal::subtract_value_array, (value, array))
        return std::move(array);
      }

      friend Array
      operator-(const value_type& value, const Array& array)
      { return value - Array(array); }


      Array&
      operator/=(const value_type& value);

      Array
      operator/(const value_type& value) &&
      { return std::move(operator/=(value)); }

      Array
      operator/(const value_type& value) const &
      { return Array(*this).operator/(value); }

      friend Array
      operator/(const value_type& value, Array&& array)
      {
        NN_GPU_FUNCTION_CALL((array.m_device == GPU),
                             internal::divide_value_array, (value, array))
        return std::move(array);
      }

      friend Array
      operator/(const value_type& value, const Array& array)
      { return value / Array(array); }

    protected: // Private functions

      void free();
      void allocate();
      void allocate(const size_type* shape, depth_type depth);

      template<class ...Types>
        const value_type*
        element_get(Types... indices) const;

      void
      elements_print(std::ostream& os,
          depth_type index, value_type* ptr) const;

    protected: // Private helper variables

      mutable std::shared_ptr<Array<size_type, CPU>> m_shape_subarray;
      mutable std::shared_ptr<Array<size_type, CPU>> m_strides_subarray;
      mutable std::shared_ptr<Array<value_type, D>> m_data_subarray;

      Type m_array_type = Type::Array;

      Device m_device = D;

    protected: // Private variables

      size_type m_size;
      value_type* m_data = nullptr;

      depth_type m_depth;
      size_type* m_shape = nullptr;
      size_type* m_strides = nullptr;

    }; // class Array

} // namespace NN::MX

#include "Array/Array.tpp"

