// src/core/matrix.hpp
// Phase 1 — Linear Algebra: generic Matrix<T> implementation.
// Row-major storage. No external ML dependencies. C++20.

#pragma once

#include "vector.hpp"

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace mlp {

template<typename T>
class Matrix {
public:
    // --- Construction ---------------------------------------------------

    Matrix() : rows_(0), cols_(0) {}

    Matrix(std::size_t rows, std::size_t cols, T value = T{})
        : data_(rows * cols, value), rows_(rows), cols_(cols) {}

    // Row-major initializer list: values listed row by row
    Matrix(std::size_t rows, std::size_t cols, std::initializer_list<T> init)
        : data_(init), rows_(rows), cols_(cols) {
        if (data_.size() != rows * cols) {
            throw std::invalid_argument(
                "Matrix: initializer_list size (" +
                std::to_string(data_.size()) + ") does not match " +
                std::to_string(rows) + "x" + std::to_string(cols)
            );
        }
    }

    // --- Accessors -------------------------------------------------------

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool        empty() const noexcept { return data_.empty(); }

    // Element access (row, col) — row-major indexing
    T& operator()(std::size_t r, std::size_t c) {
        check_bounds(r, c, "operator()");
        return data_[r * cols_ + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const {
        check_bounds(r, c, "operator()");
        return data_[r * cols_ + c];
    }

    T*       data()       noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // --- Linear Algebra --------------------------------------------------

    // Matrix–vector multiplication: result[i] = sum_j(A[i,j] * v[j])
    [[nodiscard]] Vector<T> operator*(const Vector<T>& v) const {
        if (cols_ != v.size()) {
            throw std::invalid_argument(
                "Matrix*Vector: cols (" + std::to_string(cols_) +
                ") != vector size (" + std::to_string(v.size()) + ")"
            );
        }
        Vector<T> result(rows_, T{});
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result[i] += data_[i * cols_ + j] * v[j];
            }
        }
        return result;
    }

    // Matrix–matrix multiplication: C[i,j] = sum_k(A[i,k] * B[k,j])
    [[nodiscard]] Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument(
                "Matrix*Matrix: cols (" + std::to_string(cols_) +
                ") != other.rows (" + std::to_string(other.rows_) + ")"
            );
        }
        Matrix result(rows_, other.cols_, T{});
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t k = 0; k < cols_; ++k) {
                for (std::size_t j = 0; j < other.cols_; ++j) {
                    result.data_[i * other.cols_ + j] +=
                        data_[i * cols_ + k] * other.data_[k * other.cols_ + j];
                }
            }
        }
        return result;
    }

    // Transpose: result[i,j] = A[j,i]
    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                result.data_[j * rows_ + i] = data_[i * cols_ + j];
            }
        }
        return result;
    }

    // Hadamard (element-wise) product
    [[nodiscard]] Matrix hadamard(const Matrix& other) const {
        check_same_shape(other, "hadamard");
        Matrix result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }

    // Element-wise sum
    [[nodiscard]] Matrix operator+(const Matrix& other) const {
        check_same_shape(other, "operator+");
        Matrix result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        check_same_shape(other, "operator+=");
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    // Element-wise subtraction
    [[nodiscard]] Matrix operator-(const Matrix& other) const {
        check_same_shape(other, "operator-");
        Matrix result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Matrix& operator-=(const Matrix& other) {
        check_same_shape(other, "operator-=");
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    // Scalar multiplication
    [[nodiscard]] Matrix operator*(T scalar) const {
        Matrix result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    Matrix& operator*=(T scalar) {
        for (auto& val : data_) { val *= scalar; }
        return *this;
    }

    // Equality
    bool operator==(const Matrix& other) const noexcept {
        return rows_ == other.rows_ &&
               cols_ == other.cols_ &&
               data_ == other.data_;
    }

    bool operator!=(const Matrix& other) const noexcept {
        return !(*this == other);
    }

private:
    std::vector<T> data_;
    std::size_t    rows_;
    std::size_t    cols_;

    void check_bounds(std::size_t r, std::size_t c, const char* op) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range(
                std::string("Matrix::") + op +
                ": index (" + std::to_string(r) + "," + std::to_string(c) +
                ") out of range for " + std::to_string(rows_) + "x" +
                std::to_string(cols_) + " matrix"
            );
        }
    }

    void check_same_shape(const Matrix& other, const char* op) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument(
                std::string("Matrix::") + op + ": shape mismatch (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) +
                " vs " + std::to_string(other.rows_) + "x" +
                std::to_string(other.cols_) + ")"
            );
        }
    }
};

// Scalar multiplication (scalar * matrix)
template<typename T>
[[nodiscard]] Matrix<T> operator*(T scalar, const Matrix<T>& m) {
    return m * scalar;
}

} // namespace mlp
