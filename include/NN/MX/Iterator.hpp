#pragma once

namespace NN
{
namespace MX
{

    template<typename Array>
    class ArrayIterator
    {
    public:

        using size_type       = typename Array::size_type;
        using value_type      = typename Array::value_type;
        using pointer_type    = value_type *;
        using reference_type  = value_type &;

    public:

        ArrayIterator(pointer_type ptr)
        : m_ptr(ptr) {}

        ArrayIterator &
        operator++()
        {
            ++m_ptr;
            return *this;
        }

        ArrayIterator &
        operator++(int)
        {
            ArrayIterator iterator = *this;
            ++(*this);
            return iterator;
        }

        ArrayIterator &
        operator--()
        {
            --m_ptr;
            return *this;
        }

        ArrayIterator &
        operator--(int)
        {
            ArrayIterator iterator = *this;
            --(*this);
            return iterator;
        }

        reference_type
        operator[](size_type index)
        { return *(m_ptr + index); }

        pointer_type
        operator->()
        { return m_ptr; }

        reference_type
        operator*()
        { return *m_ptr; }

        bool
        operator==(const ArrayIterator &iterator)
        { return m_ptr == iterator.m_ptr; }

        bool
        operator!=(const ArrayIterator &iterator)
        { return !operator==(iterator); }

    private:

        pointer_type m_ptr;

    };

} // namespace MX

} // namespace NN

