#pragma once

#include <vector>
#include <iostream>
#include <fstream>

#include "NN/Array.hpp"

#define FUNC_DEF(func) #func

namespace NN {

    namespace Layer {

        class Base {
        protected:

            bool bias;
            MX::Array<float> b; // bias vector
            MX::Array<float> db;
            
            float (*g) (float, int, float); // activation function
            float hp; // hyperparameter

            bool bound;

        public:

            // obviously void* to Base* overload looks like crazy
            // but as a research shows that should work in all cases
            virtual void forwardProp (const void* X) = 0;
            virtual void forwardProp (const Base* layer) { forwardProp(layer->getA()); }
            virtual void backProp (const void* gradient) = 0;
            virtual void backProp (const Base* layer) { backProp(layer->getGradient()); }

            virtual void update (float learning_rate) = 0;
            virtual void bind (const std::vector<int>& dimensions) = 0;
            virtual void bind (const Base* layer) { bind(layer->getDimensions()); }
            virtual inline void reset () { bound = false; }

            virtual const void* getA () const = 0;
            virtual const void* getGradient () const = 0;
            virtual std::vector<int> getDimensions () const = 0;

            virtual void print () const = 0;
            virtual void save (std::string file) const = 0;
            virtual void load (std::string file) = 0;
            virtual void output (std::ostream& os) = 0;
            virtual void input (std::istream& is) = 0;

        };

        template <class T>
        std::ostream& operator<< (std::ostream& os, const std::vector<T>& A) {
            os << "{ ";
            for (int i = 0; i < A.size(); ++i)
                os << A[i] << (i == A.size() - 1 ? " " : ", ");
            os << "}";
            return os;
        }

        template <class T>
        std::istream& operator>> (std::istream& is, std::vector<T>& A) {
            A.clear();
            while (is.peek() != '{' && !is.eof()) is.ignore();
            is.ignore();
            while (is.peek() != '}') {
                if (is.peek() == EOF)
                    throw Error::Base(":vector:operator>>: closing brace is missing");
                switch (is.peek()) {
                    case '\t':
                    case '\n':
                    case ' ':
                    case ',': 
                        is.ignore(); break;
                    default:
                        if ((is.peek() < 48 || is.peek() > 57) && is.peek() != '.')
                            throw Error::Base(":vector:operator>>: unknown symbol");
                        T number;
                        is >> number;
                        A.push_back(number);
                }
            }
            is.ignore();
            return is;
        }

    } // namespace Layer

} // namespace NN
