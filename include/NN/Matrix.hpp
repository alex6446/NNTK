#pragma once

#include <iostream>
#include <random>
#include <initializer_list>

#include "NN/Error.hpp"


namespace NN {

    template <class T>
    class Matrix {
    private:

        int m; // rows
        int n; // cols
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

        Matrix& operator= (const Matrix& B);
        Matrix& operator= (Matrix&& B);
        Matrix& operator= (std::initializer_list<T> l);
        Matrix& operator= (std::initializer_list<std::initializer_list<T>> l);

        // Functions applied to current object
        Matrix& fit (const Matrix& B);
        Matrix& randomize (int a, int b);

        // Functions returning a result
        Matrix flatten () const;
        Matrix reshape (int r, int c) const;
        Matrix transpose () const;
        Matrix dot (const Matrix& B) const;
        Matrix apply (T (*func) (T)) const;
        Matrix apply (T (*func) (T, int), int a) const;

        // Accessors
        T& operator() (int i, int j);
        T const& operator() (int i, int j) const;
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

    public:

        friend std::ostream& operator<< (std::ostream& os, const Matrix& A) {
            os << "[ [ ";
            for (int i = 0; i < A.m; ++i) {
                for (int j = 0; j < A.n; ++j) 
                    os << A.arr[i][j] << " ";
                if (i != A.m-1)
                    os << "] " << std::endl << "  [ ";
            }
            os << "] ] ";
            return os;
        }

        friend Matrix operator+ (T a, const Matrix& B) {
            if (!B.arr)
                throw MatrixError(":operator+: matrix not initialized");
            Matrix C = B;
            for (int i = 0; i < C.m; ++i)
                for (int j = 0; j < C.n; ++j)
                    C.arr[i][j] += a;
            return C;
        }

        friend Matrix operator- (T a, const Matrix& B) {
            if (!B.arr)
                throw MatrixError(":operator-: matrix not initialized");
            Matrix C = B;
            for (int i = 0; i < C.m; ++i)
                for (int j = 0; j < C.n; ++j)
                    C.arr[i][j] = a - C.arr[i][j];
            return C;
        }

        friend Matrix operator* (T a, const Matrix& B) {
            if (!B.arr)
                throw MatrixError(":operator*: matrix not initialized");
            Matrix C = B;
            for (int i = 0; i < C.m; ++i)
                for (int j = 0; j < C.n; ++j)
                    C.arr[i][j] *= a;
            return C;
        }

        friend Matrix operator/ (T a, const Matrix& B) {
            if (!B.arr)
                throw MatrixError(":operator/: matrix not initialized");
            Matrix C = B;
            for (int i = 0; i < C.m; ++i)
                for (int j = 0; j < C.n; ++j) {
                    if (!C.arr[i][j])
                        throw MatrixError(":operator/: division by zero");
                    C.arr[i][j] = a / C.arr[i][j];
                }
            return C;
        }

    }; // class Matrix

    typedef Matrix<float> Matrixf;
    typedef Matrix<double> Matrixd;
    typedef Matrix<int> Matrixi;

} // namespace NN

namespace NN {

    template <class T>
    Matrix<T>::Matrix()
    : m(0), n(0), arr(nullptr) {}

    template <class T>
    Matrix<T>::Matrix(int r, int c)
    : m(r), n(c), arr(nullptr) {
        if (m <= 0 || n <= 0)
            throw MatrixError(":Matrix: incorrect dimensions");
        Create();
    }

    template <class T>
    Matrix<T>::Matrix (int r, int c, T* a) 
    : m(r), n(c), arr(nullptr) {
        if (m <= 0 || n <= 0)
            throw MatrixError(":Matrix: incorrect dimensions");
        if (!a)
            throw MatrixError(":Matrix: can't copy from a null array");
        Create(a);
    }
    
    template <class T>
    Matrix<T>::Matrix (int r, int c, T** a) 
    : m(r), n(c), arr(nullptr) {
        if (m <= 0 || n <= 0)
            throw MatrixError(":Matrix: incorrect dimensions");
        if (!a)
            throw MatrixError(":Matrix: can't copy from a null array");
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
                throw MatrixError(":Matrix: incorrect initializer list");
            for (int j = 0; j < n; ++j)
                arr[i][j] = *((l.begin()+i)->begin()+j);
        }
    }
    
    template <class T>
    Matrix<T>::Matrix (const Matrix& copy)
    : m(copy.m), n(copy.n), arr(nullptr) {
        Create(copy.arr);
    }

    template <class T>
    Matrix<T>::Matrix (Matrix&& move)
    : m(move.m), n(move.n), arr(nullptr) {
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
                throw MatrixError(":Matrix: incorrect initializer list");
            for (int j = 0; j < n; ++j)
                arr[i][j] = *((l.begin()+i)->begin()+j);
        }
        return *this;
    }

    template <class T>
    Matrix<T>& Matrix<T>::fit (const Matrix& B) {
        if (!arr)
            throw MatrixError(":fit: matrix not initialized");
        if (B.m * B.n != m * n)
            throw MatrixError(":fit: sizes not match");
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
            throw MatrixError(":randomize: matrix not initialized");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] = (rand()%(b-a)) + a + (T)rand()/RAND_MAX;
        return *this;
    }

    template <class T>
    Matrix<T> Matrix<T>::flatten () const {
        if (!arr)
            throw MatrixError(":flatten: matrix not initialized");
        Matrix C(1, m * n);
        for (int i = 0, index = 0; i < m; ++i)
            for (int j = 0; j < n; ++j, ++index)
                C.arr[0][index] = arr[i][j];
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::reshape (int r, int c) const {
        if (!arr)
            throw MatrixError(":reshape: matrix not initialized");
        if (r <= 0 || c <= 0)
            throw MatrixError(":reshape: incorrect dimensions");
        if (m * n != r * c)
            throw MatrixError(":reshape: sizes not match");
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
            throw MatrixError(":transpose: matrix not initialized");
        Matrix C(n, m);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[j][i] = arr[i][j];
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::dot (const Matrix& B) const {
        if (!arr)
            throw MatrixError(":dot: matrix not initialized");
        if (n != B.m)
            throw MatrixError(":dot: multiplication conditions not met");
        Matrix C(m, B.n);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < B.n; ++j)
                for (int k = 0; k < n; ++k)
                    C.arr[i][j] += arr[i][k] * B.arr[k][j];
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::apply (T (*func) (T)) const {
        if (!arr)
            throw MatrixError(":apply: matrix not initialized");
        if (!func)
            throw MatrixError(":apply: cannot apply null function");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] = func(C.arr[i][j]);
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::apply (T (*func) (T, int), int a) const {
        if (!arr)
            throw MatrixError(":apply: matrix not initialized");
        if (!func)
            throw MatrixError(":apply: cannot apply null function");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] = func(C.arr[i][j], a);
        return C;
    }

    template <class T>
    T& Matrix<T>::operator() (int i, int j) {
        if (!arr)
            throw MatrixError(":operator(): matrix not initialized");
        if (i < 0 || j < 0)
            throw MatrixError(":operator(): negative index");
        if (i >= m || j >= n)
            throw MatrixError(":operator(): index out of bounds");
        return arr[i][j];
    }

    template <class T>
    T const& Matrix<T>::operator() (int i, int j) const {
        if (!arr)
            throw MatrixError(":operator(): matrix not initialized");
        if (i < 0 || j < 0)
            throw MatrixError(":operator(): negative index");
        if (i >= m || j >= n)
            throw MatrixError(":operator(): index out of bounds");
        return arr[i][j];
    }

    template <class T>
    bool Matrix<T>::operator== (const Matrix<T>& B) const {
        if (!arr || !B.arr)
            throw MatrixError(":operator==: matrix not initialized");
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
            throw MatrixError(":operator!=: matrix not initialized");
        return !operator==(B);
    }

    template <class T>
    Matrix<T> Matrix<T>::operator+ (const T b) const {
        if (!arr)
            throw MatrixError(":operator+: matrix not initialized");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] += b;
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator- (const T b) const {
        if (!arr)
            throw MatrixError(":operator-: matrix not initialized");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] -= b;
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator* (const T b) const {
        if (!arr)
            throw MatrixError(":operator*: matrix not initialized");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] *= b;
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator/ (const T b) const {
        if (!arr)
            throw MatrixError(":operator/: matrix not initialized");
        if (!b)
            throw MatrixError(":operator/: division by zero");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] /= b;
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator+ (const Matrix& B) const {
        if (!arr || !B.arr)
            throw MatrixError(":operator+: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator+: dimentions not match");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] += B.arr[i][j];
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator- (const Matrix& B) const {
        if (!arr || !B.arr)
            throw MatrixError(":operator-: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator-: dimentions not match");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] -= B.arr[i][j];
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator* (const Matrix& B) const {
        if (!arr || !B.arr)
            throw MatrixError(":operator*: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator*: dimentions not match");
        Matrix C = *this;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                C.arr[i][j] *= B.arr[i][j];
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator/ (const Matrix& B) const {
        if (!arr || !B.arr)
            throw MatrixError(":operator/: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator/: dimentions not match");
        Matrix C = *this;
        for (int i = 0; i < m; ++i) 
            for (int j = 0; j < n; ++j) {
                if (!B.arr[i][j])
                    throw MatrixError(":operator/: division by zero"); 
                C.arr[i][j] /= B.arr[i][j];
            }
        return C;
    }

    template <class T>
    Matrix<T> Matrix<T>::operator- () const {
        if (!arr)
            throw MatrixError(":operator-: matrix not initialized");
        return (*this) * -1;
    }

    template <class T>
    Matrix<T>& Matrix<T>::operator+= (const T b) {
        if (!arr)
            throw MatrixError(":operator+=: matrix not initialized");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] += b;
        return *this;
    }
    
    template <class T>
    Matrix<T>& Matrix<T>::operator-= (const T b) {
        if (!arr)
            throw MatrixError(":operator-=: matrix not initialized");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] -= b;
        return *this;
    }

    template <class T>
    Matrix<T>& Matrix<T>::operator*= (const T b) {
        if (!arr)
            throw MatrixError(":operator*=: matrix not initialized");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] *= b;
        return *this;
    }

    template <class T>
    Matrix<T>& Matrix<T>::operator/= (const T b) {
        if (!arr)
            throw MatrixError(":operator/=: matrix not initialized");
        if (!b)
            throw MatrixError(":operator/=: division by zero");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] /= b;
        return *this;
    }

    template <class T>
    Matrix<T>& Matrix<T>::operator+= (const Matrix& B) {
        if (!arr || !B.arr)
            throw MatrixError(":operator+=: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator+=: dimentions not match");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] += B.arr[i][j];
        return *this;
    }

    template <class T>
    Matrix<T>& Matrix<T>::operator-= (const Matrix& B) {
        if (!arr || !B.arr)
            throw MatrixError(":operator-=: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator-=: dimentions not match");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] -= B.arr[i][j];
        return *this;
    }

    template <class T>
    Matrix<T>& Matrix<T>::operator*= (const Matrix& B) {
        if (!arr || !B.arr)
            throw MatrixError(":operator*=: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator*=: dimentions not match");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                arr[i][j] *= B.arr[i][j];
        return *this;
    }

    template <class T>
    Matrix<T>& Matrix<T>::operator/= (const Matrix& B) {
        if (!arr || !B.arr)
            throw MatrixError(":operator/=: matrix not initialized");
        if (m != B.m || n != B.n)
            throw MatrixError(":operator/=: dimentions not match");
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                if (!B.arr[i][j])
                    throw MatrixError(":operator/=: division by zero");
                arr[i][j] /= B.arr[i][j];
            }
        return *this;
    }

    template <class T>
    void Matrix<T>::Create () {
        if (m <= 0 || n <= 0)
            throw MatrixError(":Create: incorrect dimensions");
        arr = new T*[m];
        for (int i = 0; i < m; ++i)
            arr[i] = new T[n] {0};
    }

    template <class T>
    void Matrix<T>::Create (T* a) {
        if (m <= 0 || n <= 0)
            throw MatrixError(":Create: incorrect dimensions");
        if (!a)
            throw MatrixError(":Create: can't copy from a null array");
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
            throw MatrixError(":Create: incorrect dimensions");
        if (!a)
            throw MatrixError(":Create: can't copy from a null array");
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
            throw MatrixError(":Create: incorrect dimensions");
        if (!a)
            throw MatrixError(":Create: can't copy from a null array");
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
            throw MatrixError(":Create: incorrect dimensions");
        if (!a)
            throw MatrixError(":Create: can't copy from a null array");
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

} // namespace NN