#pragma once

#include <iostream>
#include <random>
#include <initializer_list>
#include <vector>

#include "NN/Error.hpp"

namespace NN {

    // Forward declaration
    namespace MX {
 
        template <class T>
        class Matrix;

        template <class T>
        Matrix<T> Dot (const Matrix<T>& A, const Matrix<T>& B);
        template <class T>
        T Sum (const Matrix<T>& A);
        template <class T>
        Matrix<T> Sum (const Matrix<T>& A, int axis);
        template <class T>
        Matrix<T> Convolve (const Matrix<T>& A, const Matrix<T>& kernel, int padding=0, int stride=0);
        
    }

    // Matrix interface
    namespace MX {

        template <class T>
        class Matrix {
        private:

            int m; // rows
            int n; // columns
            T** arr; // row-wise

        public:

            // Initialization
            Matrix ();
            Matrix (int r, int c);
            Matrix (int r, int c, T* a);
            Matrix (int r, int c, T** a);
            Matrix (std::initializer_list<T> l);
            Matrix (std::initializer_list<std::initializer_list<T>> l);
            Matrix (const Matrix& copy);
            Matrix (Matrix&& move);
            ~Matrix ();

            // Assignment
            Matrix& operator= (const Matrix& B);
            Matrix& operator= (Matrix&& B);
            Matrix& operator= (std::initializer_list<T> l);
            Matrix& operator= (std::initializer_list<std::initializer_list<T>> l);

            // Functions (affect current object)
            Matrix& fit (const Matrix& B);
            Matrix& randomize (int a, int b);

            // Functions (create new object)
            Matrix flatten (int axis=0) const;
            Matrix reshape (int r, int c) const;
            Matrix transpose () const;
            Matrix rotate180 () const;
            Matrix apply (T (*func) (T)) const;
            Matrix apply (T (*func) (T, int, float), int mode, float hp = 1) const;

            // Accessors
            T& operator() (int i, int j);
            T const& operator() (int i, int j) const;
            T get (int i, int j, T a=0) const;
            void set (int i, int j, T a);
            void add (int i, int j, T a);
            inline int rows() const { return m; }
            inline int cols() const { return n; }
            inline int elements() const { return m * n; }
            inline T** data() { return arr; }
            inline const T** data() const { return (const T**)arr; }
            
            // Comparison operators
            bool operator== (const Matrix& B) const;
            bool operator!= (const Matrix& B) const;
            
            // Arithmetic operators
            Matrix operator+ (const T b) const;
            Matrix operator- (const T b) const;
            Matrix operator* (const T b) const;
            Matrix operator/ (const T b) const;

            Matrix operator+ (const Matrix& B) const;
            Matrix operator- (const Matrix& B) const;
            Matrix operator* (const Matrix& B) const;
            Matrix operator/ (const Matrix& B) const;

            Matrix operator- () const;

            Matrix& operator+= (const T b);
            Matrix& operator-= (const T b);
            Matrix& operator*= (const T b);
            Matrix& operator/= (const T b);

            Matrix& operator+= (const Matrix& B);
            Matrix& operator-= (const Matrix& B);
            Matrix& operator*= (const Matrix& B);
            Matrix& operator/= (const Matrix& B);

        private:

            void Create ();
            void Create (T* a);
            void Create (const T* a);
            void Create (T** a);
            void Create (const T** a);
            void Delete ();

        public: // friends go here

            friend Matrix Dot <> (const Matrix& A, const Matrix& B);
            friend T Sum <> (const Matrix& A);
            friend Matrix Sum <> (const Matrix& A, int axis);
            friend Matrix Convolve <> (const Matrix& A, const Matrix& kernel, int padding, int stride);

            friend std::ostream& operator<< (std::ostream& os, const Matrix& A) {
                os << "{\n    { ";
                for (int i = 0; i < A.m; ++i) {
                    for (int j = 0; j < A.n; ++j) {
                        os << A.arr[i][j];
                        if (j != A.n-1) os << ",";
                        os << " ";
                    }
                    if (i != A.m-1)
                        os << "}, " << std::endl << "    { ";
                }
                os << "}\n}";
                return os;
            }

            friend Matrix operator+ (T a, const Matrix& B) {
                if (!B.arr)
                    throw Error::Matrix(":operator+:1: matrix not initialized");
                Matrix C = B;
                for (int i = 0; i < C.m; ++i)
                    for (int j = 0; j < C.n; ++j)
                        C.arr[i][j] += a;
                return C;
            }

            friend Matrix operator- (T a, const Matrix& B) {
                if (!B.arr)
                    throw Error::Matrix(":operator-:1: matrix not initialized");
                Matrix C = B;
                for (int i = 0; i < C.m; ++i)
                    for (int j = 0; j < C.n; ++j)
                        C.arr[i][j] = a - C.arr[i][j];
                return C;
            }

            friend Matrix operator* (T a, const Matrix& B) {
                if (!B.arr)
                    throw Error::Matrix(":operator*:1: matrix not initialized");
                Matrix C = B;
                for (int i = 0; i < C.m; ++i)
                    for (int j = 0; j < C.n; ++j)
                        C.arr[i][j] *= a;
                return C;
            }

            friend Matrix operator/ (T a, const Matrix& B) {
                if (!B.arr)
                    throw Error::Matrix(":operator/:1: matrix not initialized");
                Matrix C = B;
                for (int i = 0; i < C.m; ++i)
                    for (int j = 0; j < C.n; ++j) {
                        if (!C.arr[i][j])
                            throw Error::Matrix(":operator/:1: division by zero");
                        C.arr[i][j] = a / C.arr[i][j];
                    }
                return C;
            }

        }; // class Matrix

        typedef Matrix<float> Matrixf;
        typedef Matrix<double> Matrixd;
        typedef Matrix<int> Matrixi;

        typedef std::vector<Matrix<float>> Image;
        typedef std::vector<Matrix<float>> Filter;
    
    } // namespace MX

    // Matrix implementation
    namespace MX {

        template <class T>
        Matrix<T>::Matrix()
        : m(0), n(0), arr(nullptr) {}

        template <class T>
        Matrix<T>::Matrix(int r, int c)
        : m(r), n(c), arr(nullptr) {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Matrix:1: incorrect dimensions");
            Create();
        }

        template <class T>
        Matrix<T>::Matrix (int r, int c, T* a) 
        : m(r), n(c), arr(nullptr) {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Matrix:2: incorrect dimensions");
            if (!a)
                throw Error::Matrix(":Matrix:2: can't copy from a null array");
            Create(a);
        }
        
        template <class T>
        Matrix<T>::Matrix (int r, int c, T** a) 
        : m(r), n(c), arr(nullptr) {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Matrix:3: incorrect dimensions");
            if (!a)
                throw Error::Matrix(":Matrix:3: can't copy from a null array");
            Create(a);
        }

        template <class T>
        Matrix<T>::Matrix (std::initializer_list<T> l)
        : m(1), n(l.size()), arr(nullptr) {
            Create(l.begin());
        }

        template <class T>
        Matrix<T>::Matrix (std::initializer_list<std::initializer_list<T>> l)
        : m(l.size()), n(l.begin()->size()), arr(nullptr) {
            Create();
            for (int i = 0; i < m; ++i) {
                if ((l.begin()+i)->size() != n)
                    throw Error::Matrix(":Matrix:4: incorrect initializer list");
                for (int j = 0; j < n; ++j)
                    arr[i][j] = *((l.begin()+i)->begin()+j);
            }
        }
        
        template <class T>
        Matrix<T>::Matrix (const Matrix& copy)
        : m(copy.m), n(copy.n), arr(nullptr) {
            if (copy.arr)
                Create(copy.arr);
        }

        template <class T>
        Matrix<T>::Matrix (Matrix&& move)
        : m(move.m), n(move.n), arr(nullptr) {
            if (move.arr)
                Create(move.arr);
        }

        template <class T>
        Matrix<T>::~Matrix () {
            Delete();
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator= (const Matrix& B) {
            if (B.m == m && B.n == n) { 
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j)
                        arr[i][j] = B.arr[i][j];
                return *this;
            }
            Delete();
            m = B.m;
            n = B.n;
            Create(B.arr);
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator= (Matrix&& B) {
            if (B.m == m && B.n == n) { 
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j)
                        arr[i][j] = B.arr[i][j];
                return *this;
            }
            Delete();
            m = B.m;
            n = B.n;
            Create(B.arr);
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator= (std::initializer_list<T> l) {
            if (m == 1 && n == l.size()) {
                T* a = l.begin();
                for (int j = 0; j < n; ++j)
                    arr[0][j] = a[j];
                return *this;
            }
            Delete();
            m = 1;
            n = l.size();
            Create(l.begin());
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator= (std::initializer_list<std::initializer_list<T>> l) {
            if (m != l.size() || n != l.begin()->size()) {
                Delete();
                m = l.size(); 
                n = l.begin()->size();
                Create();
            }
            for (int i = 0; i < m; ++i) {
                if ((l.begin()+i)->size() != n)
                    throw Error::Matrix(":operator=:1: incorrect initializer list");
                for (int j = 0; j < n; ++j)
                    arr[i][j] = *((l.begin()+i)->begin()+j);
            }
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::fit (const Matrix& B) {
            if (!arr)
                throw Error::Matrix(":fit:1: matrix not initialized");
            if (B.m * B.n != m * n)
                throw Error::Matrix(":fit:1: sizes not match");
            if (B.m == m && B.n == n) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j)
                        arr[i][j] = B.arr[i][j];
                return *this;
            }
            int Bi = 0, Bj = 0;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    arr[i][j] = B.arr[Bi][Bj];
                    ++Bj;
                    if (Bj == B.n) {
                        Bj = 0;
                        ++Bi;
                    }
                }
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::randomize (int a, int b) {
            if (!arr)
                throw Error::Matrix(":randomize:1: matrix not initialized");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] = (rand()%(b-a)) + a + (T)rand()/RAND_MAX;
            return *this;
        }

        template <class T>
        Matrix<T> Matrix<T>::flatten (int axis) const {
            if (!arr)
                throw Error::Matrix(":flatten:1: matrix not initialized");
            if ((axis == 0 && m == 1) || (axis == 1 && n == 1))
                return *this;
            Matrix C;
            if (axis == 0) {
                C = Matrix(1, m * n);
                for (int i = 0, index = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j, ++index)
                        C.arr[0][index] = arr[i][j];
            }
            if (axis == 1) {
                C = Matrix(m * n, 1);
                for (int i = 0, index = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j, ++index)
                        C.arr[index][0] = arr[i][j];
            }
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::reshape (int r, int c) const {
            if (!arr)
                throw Error::Matrix(":reshape:1: matrix not initialized");
            if (r <= 0 || c <= 0)
                throw Error::Matrix(":reshape:1: incorrect dimensions");
            if (m * n != r * c)
                throw Error::Matrix(":reshape:1: sizes not match");
            if (r == m && c == n) 
                return *this;
            Matrix C(r, c);
            int Ci = 0, Cj = 0;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    C.arr[Ci][Cj] = arr[i][j];
                    ++Cj;
                    if (Cj == C.n) {
                        Cj = 0;
                        ++Ci;
                    }
                }
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::transpose () const {
            if (!arr)
                throw Error::Matrix(":transpose:1: matrix not initialized");
            Matrix C(n, m);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[j][i] = arr[i][j];
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::rotate180 () const {
            if (!arr)
                throw Error::Matrix(":rotate180:1: matrix not initialized");
            Matrix C(m, n);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] = arr[m-i-1][n-j-1];
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::apply (T (*func) (T)) const {
            if (!arr)
                throw Error::Matrix(":apply:1: matrix not initialized");
            if (!func)
                throw Error::Matrix(":apply:1: cannot apply null function");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] = func(C.arr[i][j]);
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::apply (T (*func) (T, int, float), int mode, float hp) const {
            if (!arr)
                throw Error::Matrix(":apply:2: matrix not initialized");
            if (!func)
                throw Error::Matrix(":apply:2: cannot apply null function");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] = func(C.arr[i][j], mode, hp);
            return C;
        }

        template <class T>
        T& Matrix<T>::operator() (int i, int j) {
            if (!arr)
                throw Error::Matrix(":operator():1: matrix not initialized");
            if (i < 0 || j < 0)
                throw Error::Matrix(":operator():1: negative index");
            if (i >= m || j >= n)
                throw Error::Matrix(":operator():1: index out of bounds");
            return arr[i][j];
        }

        template <class T>
        T const& Matrix<T>::operator() (int i, int j) const {
            if (!arr)
                throw Error::Matrix(":operator():2: matrix not initialized");
            if (i < 0 || j < 0)
                throw Error::Matrix(":operator():2: negative index");
            if (i >= m || j >= n)
                throw Error::Matrix(":operator():2: index out of bounds");
            return arr[i][j];
        }

        template <class T>
        T Matrix<T>::get (int i, int j, T a) const {
            if (!arr) return a;
            if (i < 0 || j < 0) return a;
            if (i >= m || j >= n) return a;
            return arr[i][j];
        }

        template <class T>
        void Matrix<T>::set (int i, int j, T a) {
            if (!arr) return;
            if (i < 0 || j < 0) return;
            if (i >= m || j >= n) return;
            arr[i][j] = a;
        }

        template <class T>
        void Matrix<T>::add (int i, int j, T a) {
            if (!arr) return;
            if (i < 0 || j < 0) return;
            if (i >= m || j >= n) return;
            arr[i][j] += a;
        }

        template <class T>
        bool Matrix<T>::operator== (const Matrix<T>& B) const {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator==:1: matrix not initialized");
            if (m != B.m || n != B.n)
                return false;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    if (arr[i][j] != B.arr[i][j])
                        return false;
            return true;
        }

        template <class T>
        bool Matrix<T>::operator!= (const Matrix<T>& B) const {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator!=:1: matrix not initialized");
            return !operator==(B);
        }

        template <class T>
        Matrix<T> Matrix<T>::operator+ (const T b) const {
            if (!arr)
                throw Error::Matrix(":operator+:2: matrix not initialized");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] += b;
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator- (const T b) const {
            if (!arr)
                throw Error::Matrix(":operator-:2: matrix not initialized");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] -= b;
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator* (const T b) const {
            if (!arr)
                throw Error::Matrix(":operator*:2: matrix not initialized");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] *= b;
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator/ (const T b) const {
            if (!arr)
                throw Error::Matrix(":operator/:2: matrix not initialized");
            if (!b)
                throw Error::Matrix(":operator/:2: division by zero");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] /= b;
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator+ (const Matrix& B) const {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator+:3: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator+:3: dimentions not match");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] += B.arr[i][j];
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator- (const Matrix& B) const {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator-:3: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator-:3: dimentions not match");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] -= B.arr[i][j];
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator* (const Matrix& B) const {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator*:3: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator*:3: dimentions not match");
            Matrix C = *this;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C.arr[i][j] *= B.arr[i][j];
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator/ (const Matrix& B) const {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator/:3: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator/:3: dimentions not match");
            Matrix C = *this;
            for (int i = 0; i < m; ++i) 
                for (int j = 0; j < n; ++j) {
                    if (!B.arr[i][j])
                        throw Error::Matrix(":operator/:3: division by zero"); 
                    C.arr[i][j] /= B.arr[i][j];
                }
            return C;
        }

        template <class T>
        Matrix<T> Matrix<T>::operator- () const {
            if (!arr)
                throw Error::Matrix(":operator-:4: matrix not initialized");
            return (*this) * -1;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator+= (const T b) {
            if (!arr)
                throw Error::Matrix(":operator+=:1: matrix not initialized");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] += b;
            return *this;
        }
        
        template <class T>
        Matrix<T>& Matrix<T>::operator-= (const T b) {
            if (!arr)
                throw Error::Matrix(":operator-=:1: matrix not initialized");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] -= b;
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator*= (const T b) {
            if (!arr)
                throw Error::Matrix(":operator*=:1: matrix not initialized");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] *= b;
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator/= (const T b) {
            if (!arr)
                throw Error::Matrix(":operator/=:1: matrix not initialized");
            if (!b)
                throw Error::Matrix(":operator/=:1: division by zero");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] /= b;
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator+= (const Matrix& B) {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator+=:2: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator+=:2: dimentions not match");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] += B.arr[i][j];
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator-= (const Matrix& B) {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator-=:2: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator-=:2: dimentions not match");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] -= B.arr[i][j];
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator*= (const Matrix& B) {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator*=:2: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator*=:2: dimentions not match");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    arr[i][j] *= B.arr[i][j];
            return *this;
        }

        template <class T>
        Matrix<T>& Matrix<T>::operator/= (const Matrix& B) {
            if (!arr || !B.arr)
                throw Error::Matrix(":operator/=:2: matrix not initialized");
            if (m != B.m || n != B.n)
                throw Error::Matrix(":operator/=:2: dimentions not match");
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    if (!B.arr[i][j])
                        throw Error::Matrix(":operator/=:2: division by zero");
                    arr[i][j] /= B.arr[i][j];
                }
            return *this;
        }

        template <class T>
        void Matrix<T>::Create () {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Create:1: incorrect dimensions");
            arr = new T*[m];
            for (int i = 0; i < m; ++i)
                arr[i] = new T[n] {0};
        }

        template <class T>
        void Matrix<T>::Create (T* a) {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Create:2: incorrect dimensions");
            if (!a)
                throw Error::Matrix(":Create:2: can't copy from a null array");
            arr = new T*[m];
            for (int i = 0, index = 0; i < m; ++i) {
                arr[i] = new T[n];
                for (int j = 0; j < n; ++j, ++index)
                    arr[i][j] = a[index];
            }
        }

        template <class T>
        void Matrix<T>::Create (const T* a) {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Create:3: incorrect dimensions");
            if (!a)
                throw Error::Matrix(":Create:3: can't copy from a null array");
            arr = new T*[m];
            for (int i = 0, index = 0; i < m; ++i) {
                arr[i] = new T[n];
                for (int j = 0; j < n; ++j, ++index)
                    arr[i][j] = a[index];
            }
        }

        template <class T>
        void Matrix<T>::Create (T** a) {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Create:4: incorrect dimensions");
            if (!a)
                throw Error::Matrix(":Create:4: can't copy from a null array");
            arr = new T*[m];
            for (int i = 0; i < m; ++i) {
                arr[i] = new T[n];
                for (int j = 0; j < n; ++j)
                    arr[i][j] = a[i][j];
            }
        }

        template <class T>
        void Matrix<T>::Create (const T** a) {
            if (m <= 0 || n <= 0)
                throw Error::Matrix(":Create:5: incorrect dimensions");
            if (!a)
                throw Error::Matrix(":Create:5: can't copy from a null array");
            arr = new T*[m];
            for (int i = 0; i < m; ++i) {
                arr[i] = new T[n];
                for (int j = 0; j < n; ++j)
                    arr[i][j] = a[i][j];
            }
        }

        template <class T>
        void Matrix<T>::Delete () {
            if (!arr) return;
            for (int i = 0; i < m; ++i)
                delete[] arr[i];
            delete[] arr;
            arr = nullptr;
        }

    } // namespace MX

    // Friends implementation
    namespace MX {

        template <class T>
        Matrix<T> Dot (const Matrix<T>& A, const Matrix<T>& B) {
            if (!A.arr || !B.arr)
                throw Error::Matrix(":Dot:1: matrix not initialized");
            if (A.n != B.m)
                throw Error::Matrix(":Dot:1: multiplication conditions not met");
            Matrix<T> C(A.m, B.n);
            for (int i = 0; i < A.m; ++i)
                for (int j = 0; j < B.n; ++j)
                    for (int k = 0; k < A.n; ++k)
                        C.arr[i][j] += A.arr[i][k] * B.arr[k][j];
            return C;
        }

        template <class T>
        T Sum (const Matrix<T>& A) {
            if (!A.arr)
                throw Error::Matrix(":Sum:1: matrix not initialized");
            T sum = A.arr[0][0];
            for (int i = 0; i < A.m; ++i)
                for (int j = 0; j < A.n; ++j)
                    sum += A.arr[i][j];
            return sum - A.arr[0][0];
        }

        template <class T>
        Matrix<T> Sum (const Matrix<T>& A, int axis) {
            if (!A.arr)
                throw Error::Matrix(":Sum:2: matrix not initialized");
            Matrix<T> sum;
            if (axis == 0) {
                sum = Matrix<T>(1, A.n);
                for (int i = 0; i < A.m; ++i)
                    for (int j = 0; j < A.n; ++j)
                        sum.arr[0][j] += A.arr[i][j];
            }
            if (axis == 1) {
                sum = Matrix<T>(A.m, 1);
                for (int i = 0; i < A.m; ++i)
                    for (int j = 0; j < A.n; ++j)
                        sum.arr[i][0] += A.arr[i][j];
            }
            return sum;
        }

        template <class T>
        Matrix<T> Convolve (const Matrix<T>& A, const Matrix<T>& kernel, int padding, int stride) {
            if (!A.arr || !kernel.arr)
                throw Error::Matrix(":Convolve:1: matrix not initialized");
            if (stride < 1)
                throw Error::Matrix(":Convolve:1: stride is less than one");
            int cm = (A.m + 2*padding - kernel.m) / stride + 1;
            int cn = (A.n + 2*padding - kernel.n) / stride + 1;
            if (cm <= 0 || cn <= 0)
                throw Error::Matrix(":Convolve:1: kernel too large");
            Matrix<T> C(cm, cn);
            for (int ai = -padding, ci = 0; ci < C.m; ai+=stride, ++ci)
                for (int aj = -padding, cj = 0; cj < C.n; aj+=stride, ++cj) 
                    for (int ki = 0; ki < kernel.m; ++ki)
                        for (int kj = 0; kj < kernel.n; ++kj)
                            C.arr[ci][cj] += kernel.arr[ki][kj] * A.get(ai+ki, aj+kj);
            return C;
        }

    } // namespace MX

} // namespace NN
