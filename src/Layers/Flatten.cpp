#include "NN/Layers/Flatten.hpp"

#include <ostream>

namespace NN
{
namespace Layer
{

    Flatten::
    Flatten()
    { }

    Base *
    Flatten::
    forwardprop(const MX::Array<nn_type> &input)
    {
        m_output = MX::Array<nn_type>(input).reshape(this->output_shape());
        return this;
    }

    Base *
    Flatten::
    bind(const MX::Array<size_type> &shape)
    {
        if (m_is_bound)
            return this;
        m_input_shape = shape;
        m_is_bound = true;
        return this;
    }

    const Base *
    Flatten::
    save(std::string file) const
    {
        std::ofstream fout(file);
        fout << *this;
        fout.close();
        return this;
    }

    Base *
    Flatten::
    load(std::string file)
    {
        std::ifstream fin(file);
        fin >> *this;
        fin.close();
        return this;
    }

    std::ostream &
    operator<<(std::ostream &os, const Flatten &layer)
    {
        os << "Layer Flatten {" << std::endl
           << "is_bound: " << layer.m_is_bound << std::endl
           << "input_shape: " << layer.m_input_shape << std::endl
           << "}" << std::endl;
        return os;
    }

    std::istream &
    operator>>(std::istream &is, Flatten &layer)
    {
        std::string buffer;
        is >> buffer // Layer
           >> buffer // Flatten
           >> buffer; // {
        while (buffer != "}") {
            is >> buffer;
            if (buffer == "is_bound:") is >> layer.m_is_bound;
            else if (buffer == "input_shape:") is >> layer.m_input_shape;
        }
        if (layer.m_input_shape.size() == 0)
            layer.m_is_bound = false;
        return is;
    }

} // namespace Layer

} // namespace NN
