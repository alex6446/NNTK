#pragma once

#include <cstddef>
#include <iterator>

namespace NN::MX
{

  template<typename T>
    class Iterator
    {
    public:

      using iterator_category = std::forward_iterator_tag;
      using size_type       = std::size_t;
      using value_type      = T;
      using difference_type = T;
      using pointer         = value_type*;
      using reference       = value_type&;

    public:

      Iterator(pointer ptr) : m_ptr(ptr) { }

      Iterator &
      operator++()
      {
        ++m_ptr;
        return *this;
      }

      Iterator &
      operator++(int)
      { return Iterator(m_ptr++); }

      Iterator &
      operator--()
      {
        --m_ptr;
        return *this;
      }

      Iterator &
      operator--(int)
      { return Iterator(m_ptr--); }

      reference
      operator[](size_type index)
      { return *(m_ptr + index); }

      pointer
      operator->()
      { return m_ptr; }

      reference
      operator*()
      { return *m_ptr; }

      bool
      operator==(const Iterator &iterator)
      { return m_ptr == iterator.m_ptr; }

      bool
      operator!=(const Iterator &iterator)
      { return !operator==(iterator); }

    private:

      pointer m_ptr;

    }; // class Iterator

} // namespace NN::MX

