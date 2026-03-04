// src/mlp/trainer.hpp
// Phase 6 — Trainer: SGD / Mini-batch training loop with epoch logging.
//
// Mini-batch training loop per epoch:
//   shuffle indices (optional)
//   for each batch of B samples:
//       for each sample (x, y) in batch:
//           ŷ = mlp.forward(x)
//           loss += CE(ŷ, y)
//           correct += (argmax(ŷ) == argmax(y))
//           mlp.backward(y)          ← accumulates grad_W, grad_b
//       mlp.step(lr / B)             ← applies mean gradient, zeros grads
//
// With batch_size=1 (default) the behaviour is identical to plain SGD.
// After each epoch: store mean loss and accuracy.
// evaluate() computes metrics without touching gradients or weights.

#pragma once

#include "mlp/mlp.hpp"
#include "core/vector.hpp"
#include "loss/cross_entropy.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace mlp {

// ------------------------------------------------------------------
// Dataset: parallel arrays of inputs and one-hot labels.
// ------------------------------------------------------------------
template<typename T>
struct Dataset {
    std::vector<Vector<T>> inputs;
    std::vector<Vector<T>> labels;   // one-hot encoded

    [[nodiscard]] std::size_t size()  const noexcept { return inputs.size(); }
    [[nodiscard]] bool        empty() const noexcept { return inputs.empty(); }

    void validate() const {
        if (inputs.size() != labels.size()) {
            throw std::invalid_argument(
                "Dataset: inputs and labels must have the same size");
        }
    }
};

// ------------------------------------------------------------------
// Trainer<T, Activation>
// ------------------------------------------------------------------
template<typename T, typename Activation>
class Trainer {
public:
    Trainer() = default;

    // ------------------------------------------------------------------
    // train(): mini-batch (or SGD) training over 'epochs' passes.
    // Clears history before running.
    //
    // batch_size   — samples per gradient update (default 1 = SGD).
    //                The last batch in each epoch may be smaller.
    // shuffle      — if true, sample order is randomised each epoch.
    // shuffle_seed — RNG seed for reproducible shuffling.
    // ------------------------------------------------------------------
    void train(MLP<T, Activation>& mlp,
               const Dataset<T>&   data,
               std::size_t         epochs,
               T                   learning_rate,
               std::size_t         batch_size   = 1,
               bool                shuffle      = false,
               std::uint32_t       shuffle_seed = 42) {
        if (data.empty()) {
            throw std::invalid_argument("Trainer::train: empty dataset");
        }
        data.validate();
        if (epochs == 0) {
            throw std::invalid_argument(
                "Trainer::train: epochs must be > 0");
        }
        if (batch_size == 0) {
            throw std::invalid_argument(
                "Trainer::train: batch_size must be > 0");
        }

        epoch_losses_.clear();
        epoch_losses_.reserve(epochs);
        epoch_accuracies_.clear();
        epoch_accuracies_.reserve(epochs);

        const std::size_t n = data.size();

        // Index permutation — shuffled each epoch if requested.
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(shuffle_seed);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            if (shuffle) {
                std::shuffle(idx.begin(), idx.end(), rng);
            }

            T           total_loss = T{0};
            std::size_t correct    = 0;

            // --------------------------------------------------------
            // Mini-batch loop: process samples in chunks of batch_size.
            // The last chunk may be smaller than batch_size.
            // --------------------------------------------------------
            std::size_t start = 0;
            while (start < n) {
                const std::size_t end        = std::min(start + batch_size, n);
                const std::size_t cur_batch  = end - start;

                // Accumulate gradients across the batch (no step yet).
                for (std::size_t bi = start; bi < end; ++bi) {
                    const std::size_t i      = idx[bi];
                    const Vector<T>&  y_hat  = mlp.forward(data.inputs[i]);

                    total_loss += loss_fn_.compute_loss(y_hat, data.labels[i]);
                    correct    += (argmax(y_hat) == argmax(data.labels[i]))
                                  ? 1u : 0u;

                    mlp.backward(data.labels[i]);
                }

                // Apply mean gradient: step(lr / B) = lr * (sum / B).
                mlp.step(learning_rate / static_cast<T>(cur_batch));

                start = end;
            }

            epoch_losses_.push_back(total_loss / static_cast<T>(n));
            epoch_accuracies_.push_back(
                static_cast<T>(correct) / static_cast<T>(n));
        }
    }

    // ------------------------------------------------------------------
    // evaluate(): mean CE loss on 'data' without modifying weights.
    // ------------------------------------------------------------------
    T evaluate(MLP<T, Activation>& mlp, const Dataset<T>& data) {
        if (data.empty()) {
            throw std::invalid_argument("Trainer::evaluate: empty dataset");
        }
        data.validate();

        T total_loss = T{0};
        for (std::size_t i = 0; i < data.size(); ++i) {
            mlp.forward(data.inputs[i]);
            total_loss += mlp.compute_loss(data.labels[i]);
        }
        return total_loss / static_cast<T>(data.size());
    }

    // ------------------------------------------------------------------
    // compute_accuracy(): fraction of correctly classified samples.
    // Returns a value in [0, 1].
    // ------------------------------------------------------------------
    T compute_accuracy(MLP<T, Activation>& mlp, const Dataset<T>& data) {
        if (data.empty()) {
            throw std::invalid_argument(
                "Trainer::compute_accuracy: empty dataset");
        }
        data.validate();

        std::size_t correct = 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            const Vector<T>& y_hat = mlp.forward(data.inputs[i]);
            correct += (argmax(y_hat) == argmax(data.labels[i])) ? 1u : 0u;
        }
        return static_cast<T>(correct) / static_cast<T>(data.size());
    }

    // ------------------------------------------------------------------
    // History accessors
    // ------------------------------------------------------------------
    [[nodiscard]] const std::vector<T>& epoch_losses()     const noexcept {
        return epoch_losses_;
    }
    [[nodiscard]] const std::vector<T>& epoch_accuracies() const noexcept {
        return epoch_accuracies_;
    }

    // ------------------------------------------------------------------
    // Static helper: index of the largest element (argmax).
    // ------------------------------------------------------------------
    [[nodiscard]] static std::size_t argmax(const Vector<T>& v) {
        if (v.empty()) {
            throw std::invalid_argument("Trainer::argmax: empty vector");
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < v.size(); ++i) {
            if (v[i] > v[best]) { best = i; }
        }
        return best;
    }

private:
    CrossEntropy<T>  loss_fn_;
    std::vector<T>   epoch_losses_;
    std::vector<T>   epoch_accuracies_;
};

} // namespace mlp
