// src/core/vector.hpp
// Phase 1 — Linear Algebra: generic Vector<T> implementation.
// No external ML dependencies. C++20.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace mlp {

template<typename T>
class Vector {
public:
    // --- Construction ---------------------------------------------------

    Vector() = default;

    explicit Vector(std::size_t size, T value = T{})
        : data_(size, value) {}

    Vector(std::initializer_list<T> init)
        : data_(init) {}

    // --- Accessors -------------------------------------------------------

    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    T& operator[](std::size_t i) {
        if (i >= data_.size()) {
            throw std::out_of_range("Vector: index out of range");
        }
        return data_[i];
    }

    const T& operator[](std::size_t i) const {
        if (i >= data_.size()) {
            throw std::out_of_range("Vector: index out of range");
        }
        return data_[i];
    }

    // Raw pointer access (needed for efficient algorithms)
    T*       data()       noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Iterators
    auto begin()       noexcept { return data_.begin(); }
    auto end()         noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end()   const noexcept { return data_.end(); }

    // --- Arithmetic -------------------------------------------------------

    // Element-wise sum
    [[nodiscard]] Vector operator+(const Vector& other) const {
        check_same_size(other, "operator+");
        Vector result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Vector& operator+=(const Vector& other) {
        check_same_size(other, "operator+=");
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    // Element-wise subtraction
    [[nodiscard]] Vector operator-(const Vector& other) const {
        check_same_size(other, "operator-");
        Vector result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Vector& operator-=(const Vector& other) {
        check_same_size(other, "operator-=");
        for (std::size_t i = 0; i < data_.size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    // Dot product
    [[nodiscard]] T dot(const Vector& other) const {
        check_same_size(other, "dot");
        T result = T{};
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result += data_[i] * other.data_[i];
        }
        return result;
    }

    // Scalar multiplication (vector * scalar)
    [[nodiscard]] Vector operator*(T scalar) const {
        Vector result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    Vector& operator*=(T scalar) {
        for (auto& val : data_) { val *= scalar; }
        return *this;
    }

    // Element-wise product (Hadamard for vectors)
    [[nodiscard]] Vector hadamard(const Vector& other) const {
        check_same_size(other, "hadamard");
        Vector result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }

    // Unary negation
    [[nodiscard]] Vector operator-() const {
        Vector result(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = -data_[i];
        }
        return result;
    }

    // Equality
    bool operator==(const Vector& other) const noexcept {
        return data_ == other.data_;
    }

    bool operator!=(const Vector& other) const noexcept {
        return !(*this == other);
    }

private:
    std::vector<T> data_;

    void check_same_size(const Vector& other, const char* op) const {
        if (data_.size() != other.data_.size()) {
            throw std::invalid_argument(
                std::string("Vector::") + op +
                ": size mismatch (" +
                std::to_string(data_.size()) + " vs " +
                std::to_string(other.data_.size()) + ")"
            );
        }
    }
};

// Scalar multiplication (scalar * vector)
template<typename T>
[[nodiscard]] Vector<T> operator*(T scalar, const Vector<T>& v) {
    return v * scalar;
}

} // namespace mlp
