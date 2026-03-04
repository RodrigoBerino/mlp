// src/mlp/mlp.hpp
// Phase 5 — MLP: fully-connected multi-layer perceptron with Softmax output.
//
// Architecture:
//   layers_[0]       : input → hidden_1    (HiddenActivation)
//   layers_[1..L-2]  : hidden → hidden     (HiddenActivation)
//   layers_[L-1]     : hidden → output     (activation output ignored*)
//   y_hat_           : Softmax(z(L))
//
// (*) The output layer's built-in activation is bypassed.
//     The MLP applies Softmax directly to z(L), then uses the combined
//     Softmax + CrossEntropy gradient:  δ(L) = ŷ − y
//     This avoids materialising the full Softmax Jacobian.
//
// Forward:
//   z(l) = W(l) * a(l-1) + b(l)
//   a(l) = φ(z(l))                 (hidden layers)
//   ŷ    = Softmax(z(L))            (output)
//
// Backward:
//   δ(L) = ŷ − y                   (Softmax+CE combined gradient at z level)
//   δ(l) = W(l+1)^T * δ(l+1) ⊙ φ'(z(l))
//
// Gradients:
//   ∂L/∂W(l) = δ(l) * a(l-1)^T
//   ∂L/∂b(l) = δ(l)
//
// Update:
//   W(l) ← W(l) − η * ∂L/∂W(l)
//   b(l) ← b(l) − η * ∂L/∂b(l)

#pragma once

#include "activations/softmax.hpp"
#include "core/vector.hpp"
#include "layers/layer.hpp"
#include "loss/cross_entropy.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mlp {

template<typename T, typename Activation>
class MLP {
public:
    // ------------------------------------------------------------------
    // Construction
    // layer_sizes : {input_size, hidden1, hidden2, ..., output_size}
    // Example: {4, 8, 3} = 4-input, 8-neuron hidden, 3-class output
    // Each layer gets a different seed derived from the base seed.
    // ------------------------------------------------------------------
    MLP(const std::vector<std::size_t>& layer_sizes,
        std::uint32_t seed = 42)
    {
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument(
                "MLP: layer_sizes must have at least 2 elements "
                "(input_size and output_size)");
        }
        layers_.reserve(layer_sizes.size() - 1);
        for (std::size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
            layers_.emplace_back(layer_sizes[i],
                                 layer_sizes[i + 1],
                                 seed + static_cast<std::uint32_t>(i));
        }
    }

    // ------------------------------------------------------------------
    // Forward pass
    // Returns y_hat = Softmax(z(L)).
    // Intermediate activations are cached inside each Layer for backward.
    // ------------------------------------------------------------------
    const Vector<T>& forward(const Vector<T>& input) {
        const Vector<T>* a = &input;

        // Hidden layers: all except the last
        for (std::size_t i = 0; i + 1 < layers_.size(); ++i) {
            a = &layers_[i].forward(*a);
        }

        // Output layer: compute z(L); activation output a(L) is not used
        layers_.back().forward(*a);

        // Apply Softmax to z(L) to obtain the probability vector
        y_hat_ = softmax_(layers_.back().z());
        return y_hat_;
    }

    // ------------------------------------------------------------------
    // Backward pass
    // Requires a prior call to forward().
    // Computes and accumulates ∂L/∂W and ∂L/∂b for every layer.
    // ------------------------------------------------------------------
    void backward(const Vector<T>& y_true) {
        if (y_hat_.size() != y_true.size()) {
            throw std::invalid_argument(
                "MLP::backward: y_true size mismatch");
        }

        // Output layer: combined Softmax+CE gradient at z level
        const Vector<T> delta_out = loss_fn_.compute_delta(y_hat_, y_true);
        layers_.back().backward_output_z(delta_out);

        // Hidden layers in reverse order
        for (int i = static_cast<int>(layers_.size()) - 2; i >= 0; --i) {
            layers_[i].backward(layers_[i + 1].delta(),
                                layers_[i + 1].W());
        }
    }

    // ------------------------------------------------------------------
    // SGD parameter update for all layers.
    // Also zeroes accumulated gradients.
    // ------------------------------------------------------------------
    void step(T learning_rate) {
        for (auto& layer : layers_) {
            layer.update(learning_rate);
        }
    }

    // ------------------------------------------------------------------
    // CrossEntropy loss for the last forward output.
    // Requires a prior call to forward().
    // ------------------------------------------------------------------
    [[nodiscard]] T compute_loss(const Vector<T>& y_true) const {
        return loss_fn_.compute_loss(y_hat_, y_true);
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    [[nodiscard]] std::size_t num_layers() const noexcept {
        return layers_.size();
    }

    [[nodiscard]] const Layer<T, Activation>& layer(std::size_t idx) const {
        if (idx >= layers_.size()) {
            throw std::out_of_range("MLP::layer: index out of range");
        }
        return layers_[idx];
    }

    Layer<T, Activation>& layer(std::size_t idx) {
        if (idx >= layers_.size()) {
            throw std::out_of_range("MLP::layer: index out of range");
        }
        return layers_[idx];
    }

    // Last forward output (y_hat after Softmax)
    [[nodiscard]] const Vector<T>& output() const noexcept { return y_hat_; }

private:
    std::vector<Layer<T, Activation>> layers_;
    Softmax<T>       softmax_;
    CrossEntropy<T>  loss_fn_;
    Vector<T>        y_hat_;  // Softmax output, cached for backward/loss
};

} // namespace mlp
