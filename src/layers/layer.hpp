// src/layers/layer.hpp
// Phase 3 — Layer: generic fully-connected layer with forward and backward.
//
// Mathematical formulation (from prd.md):
//   Forward:   z(l) = W(l) * a(l-1) + b(l)
//              a(l) = φ(z(l))
//
//   Backward:  δ(l) = (W(l+1)^T * δ(l+1)) ⊙ φ'(z(l))
//
//   Gradients: ∂L/∂W(l) = δ(l) * a(l-1)^T
//              ∂L/∂b(l) = δ(l)
//
// Initialisation: Xavier uniform
//   limit = sqrt(6 / (fan_in + fan_out))
//   W ~ Uniform(-limit, +limit),  b = 0

#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace mlp {

template<typename T, typename Activation>
class Layer {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    // Creates a layer with Xavier-initialised weights.
    // fan_in  = number of inputs  (columns of W)
    // fan_out = number of neurons (rows of W)
    // seed    = fixed seed for reproducibility
    Layer(std::size_t fan_in,
          std::size_t fan_out,
          std::uint32_t seed = 42)
        : W_(fan_out, fan_in)
        , b_(fan_out, T{0})
        , z_(fan_out)
        , a_(fan_out)
        , delta_(fan_out)
        , grad_W_(fan_out, fan_in, T{0})
        , grad_b_(fan_out, T{0})
        , fan_in_(fan_in)
        , fan_out_(fan_out)
    {
        xavier_init(seed);
    }

    // Creates a layer with pre-set weights (for testing / reproducibility).
    Layer(Matrix<T> W, Vector<T> b)
        : W_(std::move(W))
        , b_(std::move(b))
        , z_(W_.rows())
        , a_(W_.rows())
        , delta_(W_.rows())
        , grad_W_(W_.rows(), W_.cols(), T{0})
        , grad_b_(W_.rows(), T{0})
        , fan_in_(W_.cols())
        , fan_out_(W_.rows())
    {
        if (b_.size() != W_.rows()) {
            throw std::invalid_argument(
                "Layer: bias size must equal number of rows in W");
        }
    }

    // ------------------------------------------------------------------
    // Forward pass:  z = W * input + b,   a = φ(z)
    // Returns reference to the activation vector a.
    // ------------------------------------------------------------------
    const Vector<T>& forward(const Vector<T>& input) {
        if (input.size() != fan_in_) {
            throw std::invalid_argument(
                "Layer::forward: input size (" +
                std::to_string(input.size()) + ") != fan_in (" +
                std::to_string(fan_in_) + ")");
        }

        // z = W * input + b
        z_ = W_ * input + b_;

        // a = φ(z)  — element-wise
        for (std::size_t i = 0; i < fan_out_; ++i) {
            a_[i] = activation_(z_[i]);
        }

        // cache input for gradient computation
        last_input_ = input;

        return a_;
    }

    // ------------------------------------------------------------------
    // Backward pass (hidden layer):
    //   δ(l) = (W(l+1)^T * δ(l+1)) ⊙ φ'(z(l))
    //
    // delta_next : δ of the next layer  (l+1)
    // W_next     : weight matrix of the next layer  W(l+1)
    //
    // Returns δ(l), which is passed to the layer before this one.
    // Also accumulates gradients ∂L/∂W and ∂L/∂b.
    // ------------------------------------------------------------------
    const Vector<T>& backward(const Vector<T>& delta_next,
                               const Matrix<T>& W_next) {
        // Propagate error: W_next^T * delta_next
        Vector<T> propagated = W_next.transpose() * delta_next;

        // Element-wise multiply by φ'(z)
        for (std::size_t i = 0; i < fan_out_; ++i) {
            delta_[i] = propagated[i] * activation_.derivative(z_[i]);
        }

        accumulate_gradients();
        return delta_;
    }

    // ------------------------------------------------------------------
    // Backward pass (output layer — general loss):
    //   Receives ∂L/∂a (upstream gradient w.r.t. activations).
    //   Computes δ(L) = ∂L/∂a ⊙ φ'(z(L))
    //
    // NOTE: For Softmax + CrossEntropy the combined gradient at z is
    //   δ(L) = ŷ - y  (derivative already absorbed).
    //   The MLP class handles that case by calling backward_output_z().
    // ------------------------------------------------------------------
    const Vector<T>& backward_output(const Vector<T>& grad_a) {
        if (grad_a.size() != fan_out_) {
            throw std::invalid_argument(
                "Layer::backward_output: delta size mismatch");
        }
        for (std::size_t i = 0; i < fan_out_; ++i) {
            delta_[i] = grad_a[i] * activation_.derivative(z_[i]);
        }
        accumulate_gradients();
        return delta_;
    }

    // ------------------------------------------------------------------
    // Backward pass (output layer — when δ(L) is provided at z level).
    //   Used for Softmax+CrossEntropy where δ(L) = ŷ - y is already
    //   the gradient w.r.t. z (no φ'(z) multiplication needed).
    // ------------------------------------------------------------------
    const Vector<T>& backward_output_z(const Vector<T>& delta_z) {
        if (delta_z.size() != fan_out_) {
            throw std::invalid_argument(
                "Layer::backward_output_z: delta size mismatch");
        }
        delta_ = delta_z;
        accumulate_gradients();
        return delta_;
    }

    // ------------------------------------------------------------------
    // SGD weight update:  W -= lr * grad_W,   b -= lr * grad_b
    // ------------------------------------------------------------------
    void update(T learning_rate) {
        for (std::size_t i = 0; i < fan_out_; ++i) {
            for (std::size_t j = 0; j < fan_in_; ++j) {
                W_(i, j) -= learning_rate * grad_W_(i, j);
            }
            b_[i] -= learning_rate * grad_b_[i];
        }
        zero_gradients();
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    const Matrix<T>& W()       const noexcept { return W_; }
    const Vector<T>& b()       const noexcept { return b_; }
    const Vector<T>& z()       const noexcept { return z_; }
    const Vector<T>& a()       const noexcept { return a_; }
    const Vector<T>& delta()   const noexcept { return delta_; }
    const Matrix<T>& grad_W()  const noexcept { return grad_W_; }
    const Vector<T>& grad_b()  const noexcept { return grad_b_; }

    std::size_t fan_in()  const noexcept { return fan_in_; }
    std::size_t fan_out() const noexcept { return fan_out_; }

    // Mutable access — used by MLP to set weights in tests
    Matrix<T>& W() noexcept { return W_; }
    Vector<T>& b() noexcept { return b_; }

    void zero_gradients() {
        grad_W_ = Matrix<T>(fan_out_, fan_in_, T{0});
        grad_b_ = Vector<T>(fan_out_, T{0});
    }

private:
    Matrix<T>   W_;           // weight matrix  [fan_out × fan_in]
    Vector<T>   b_;           // bias vector    [fan_out]
    Vector<T>   z_;           // pre-activation cache
    Vector<T>   a_;           // activation cache
    Vector<T>   delta_;       // error signal for this layer
    Matrix<T>   grad_W_;      // accumulated ∂L/∂W
    Vector<T>   grad_b_;      // accumulated ∂L/∂b
    Vector<T>   last_input_;  // input cache (for gradient computation)
    Activation  activation_;

    std::size_t fan_in_;
    std::size_t fan_out_;

    // ------------------------------------------------------------------
    // Xavier uniform initialisation
    // limit = sqrt(6 / (fan_in + fan_out))
    // W[i,j] ~ Uniform(-limit, +limit),  b = 0
    // ------------------------------------------------------------------
    void xavier_init(std::uint32_t seed) {
        const T limit = std::sqrt(T{6} / static_cast<T>(fan_in_ + fan_out_));
        std::mt19937 rng(seed);
        std::uniform_real_distribution<T> dist(-limit, limit);
        for (std::size_t i = 0; i < fan_out_; ++i) {
            for (std::size_t j = 0; j < fan_in_; ++j) {
                W_(i, j) = dist(rng);
            }
        }
    }

    // Accumulate gradients from current delta and last_input_:
    //   grad_W += δ * a_prev^T   (outer product)
    //   grad_b += δ
    void accumulate_gradients() {
        for (std::size_t i = 0; i < fan_out_; ++i) {
            for (std::size_t j = 0; j < fan_in_; ++j) {
                grad_W_(i, j) += delta_[i] * last_input_[j];
            }
            grad_b_[i] += delta_[i];
        }
    }
};

} // namespace mlp
