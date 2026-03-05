// src/mlp/trainer.hpp
// Phase 6  — Trainer: SGD / Mini-batch training loop with epoch logging.
// Phase 9  — Mini-batch with shuffle.
// Phase 10 — Validation split + Early stopping + Best-model restore.
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
//
// train_with_validation():
//   - Trains on train_data only.
//   - Evaluates on val_data after every epoch (no weight update).
//   - Stores train_loss_history, val_loss_history, val_macro_f1_history.
//   - Implements early stopping: if val_loss does not improve by min_delta
//     for 'patience' consecutive epochs, training halts.
//   - Saves a weight snapshot whenever val_loss improves; restores it at end.

#pragma once

#include "mlp/mlp.hpp"
#include "core/vector.hpp"
#include "loss/cross_entropy.hpp"
#include "evaluation/metrics.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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
// EarlyStoppingConfig<T>
//
// patience  — number of consecutive epochs without improvement before
//             training is halted (default 5).
// min_delta — minimum decrease in val_loss to count as an improvement
//             (default 1e-4).  Use 0 to stop on any non-decrease.
// ------------------------------------------------------------------
template<typename T>
struct EarlyStoppingConfig {
    std::size_t patience  = 5;
    T           min_delta = T{1e-4};
};

// ------------------------------------------------------------------
// Trainer<T, Activation>
// ------------------------------------------------------------------
template<typename T, typename Activation>
class Trainer {
public:
    // Snapshot type: W and b for every layer, ordered by layer index.
    using LayerWeights   = std::pair<Matrix<T>, Vector<T>>;
    using WeightSnapshot = std::vector<LayerWeights>;

    Trainer() = default;

    // ------------------------------------------------------------------
    // train(): mini-batch (or SGD) training over 'epochs' passes.
    // Clears epoch_losses_ / epoch_accuracies_ before running.
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

        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0u);
        std::mt19937 rng(shuffle_seed);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            if (shuffle) {
                std::shuffle(idx.begin(), idx.end(), rng);
            }

            T           total_loss = T{0};
            std::size_t correct    = 0;

            std::size_t start = 0;
            while (start < n) {
                const std::size_t end       = std::min(start + batch_size, n);
                const std::size_t cur_batch = end - start;

                for (std::size_t bi = start; bi < end; ++bi) {
                    const std::size_t i     = idx[bi];
                    const Vector<T>&  y_hat = mlp.forward(data.inputs[i]);

                    total_loss += loss_fn_.compute_loss(y_hat, data.labels[i]);
                    correct    += (argmax(y_hat) == argmax(data.labels[i]))
                                  ? 1u : 0u;

                    mlp.backward(data.labels[i]);
                }

                mlp.step(learning_rate / static_cast<T>(cur_batch));
                start = end;
            }

            epoch_losses_.push_back(total_loss / static_cast<T>(n));
            epoch_accuracies_.push_back(
                static_cast<T>(correct) / static_cast<T>(n));
        }
    }

    // ------------------------------------------------------------------
    // train_with_validation(): mini-batch training with per-epoch
    // validation, early stopping, and best-model restoration.
    //
    // Populates: train_loss_history_, val_loss_history_,
    //            val_macro_f1_history_.
    //
    // Early stopping:
    //   if val_loss does not decrease by at least es.min_delta for
    //   es.patience consecutive epochs → training halts.
    //
    // Best model:
    //   A weight snapshot is saved whenever val_loss improves.
    //   At the end of training (early stop or normal) the best
    //   snapshot is restored to the MLP.
    // ------------------------------------------------------------------
    void train_with_validation(
        MLP<T, Activation>&    mlp,
        const Dataset<T>&      train_data,
        const Dataset<T>&      val_data,
        std::size_t            epochs,
        T                      learning_rate,
        std::size_t            batch_size   = 1,
        bool                   shuffle      = false,
        std::uint32_t          shuffle_seed = 42,
        EarlyStoppingConfig<T> es           = {})
    {
        if (train_data.empty()) {
            throw std::invalid_argument(
                "Trainer::train_with_validation: empty train dataset");
        }
        if (val_data.empty()) {
            throw std::invalid_argument(
                "Trainer::train_with_validation: empty validation dataset");
        }
        train_data.validate();
        val_data.validate();
        if (epochs == 0) {
            throw std::invalid_argument(
                "Trainer::train_with_validation: epochs must be > 0");
        }
        if (batch_size == 0) {
            throw std::invalid_argument(
                "Trainer::train_with_validation: batch_size must be > 0");
        }

        train_loss_history_.clear();
        val_loss_history_.clear();
        val_macro_f1_history_.clear();
        train_loss_history_.reserve(epochs);
        val_loss_history_.reserve(epochs);
        val_macro_f1_history_.reserve(epochs);

        const std::size_t n = train_data.size();
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0u);
        std::mt19937 rng(shuffle_seed);

        T              best_val_loss    = std::numeric_limits<T>::infinity();
        std::size_t    patience_counter = 0;
        WeightSnapshot best_snapshot;

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {

            // ---- Train on train_data only ----------------------------
            if (shuffle) {
                std::shuffle(idx.begin(), idx.end(), rng);
            }

            T train_loss = T{0};
            std::size_t start = 0;
            while (start < n) {
                const std::size_t end       = std::min(start + batch_size, n);
                const std::size_t cur_batch = end - start;

                for (std::size_t bi = start; bi < end; ++bi) {
                    const std::size_t i     = idx[bi];
                    const Vector<T>&  y_hat = mlp.forward(train_data.inputs[i]);
                    train_loss += loss_fn_.compute_loss(y_hat, train_data.labels[i]);
                    mlp.backward(train_data.labels[i]);
                }
                mlp.step(learning_rate / static_cast<T>(cur_batch));
                start = end;
            }
            train_loss_history_.push_back(train_loss / static_cast<T>(n));

            // ---- Evaluate on val_data (no weight update) -------------
            const T val_loss = evaluate(mlp, val_data);
            val_loss_history_.push_back(val_loss);

            const T val_f1 = compute_val_macro_f1(mlp, val_data);
            val_macro_f1_history_.push_back(val_f1);

            // ---- Early stopping logic --------------------------------
            if (val_loss < best_val_loss - es.min_delta) {
                best_val_loss    = val_loss;
                patience_counter = 0;
                best_snapshot    = take_snapshot(mlp);
            } else {
                ++patience_counter;
                if (patience_counter >= es.patience) {
                    break;
                }
            }
        }

        // Restore best weights if we ever saved a snapshot.
        if (!best_snapshot.empty()) {
            restore_snapshot(mlp, best_snapshot);
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
    // take_snapshot(): copy W and b from every layer of 'mlp'.
    // ------------------------------------------------------------------
    [[nodiscard]] WeightSnapshot take_snapshot(
        const MLP<T, Activation>& mlp) const
    {
        WeightSnapshot snap;
        snap.reserve(mlp.num_layers());
        for (std::size_t i = 0; i < mlp.num_layers(); ++i) {
            snap.emplace_back(mlp.layer(i).W(), mlp.layer(i).b());
        }
        return snap;
    }

    // ------------------------------------------------------------------
    // restore_snapshot(): write saved W and b back into every layer.
    // ------------------------------------------------------------------
    void restore_snapshot(MLP<T, Activation>&   mlp,
                          const WeightSnapshot& snap) const
    {
        if (snap.size() != mlp.num_layers()) {
            throw std::invalid_argument(
                "Trainer::restore_snapshot: snapshot size mismatch");
        }
        for (std::size_t i = 0; i < mlp.num_layers(); ++i) {
            mlp.layer(i).W() = snap[i].first;
            mlp.layer(i).b() = snap[i].second;
        }
    }

    // ------------------------------------------------------------------
    // History accessors — train()
    // ------------------------------------------------------------------
    [[nodiscard]] const std::vector<T>& epoch_losses() const noexcept {
        return epoch_losses_;
    }
    [[nodiscard]] const std::vector<T>& epoch_accuracies() const noexcept {
        return epoch_accuracies_;
    }

    // ------------------------------------------------------------------
    // History accessors — train_with_validation()
    // ------------------------------------------------------------------
    [[nodiscard]] const std::vector<T>& train_loss_history() const noexcept {
        return train_loss_history_;
    }
    [[nodiscard]] const std::vector<T>& val_loss_history() const noexcept {
        return val_loss_history_;
    }
    [[nodiscard]] const std::vector<T>& val_macro_f1_history() const noexcept {
        return val_macro_f1_history_;
    }

    // ------------------------------------------------------------------
    // argmax(): index of the largest element in v.
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
    CrossEntropy<T> loss_fn_;

    // train() history
    std::vector<T> epoch_losses_;
    std::vector<T> epoch_accuracies_;

    // train_with_validation() history
    std::vector<T> train_loss_history_;
    std::vector<T> val_loss_history_;
    std::vector<T> val_macro_f1_history_;

    // ------------------------------------------------------------------
    // compute_val_macro_f1(): collect predictions on 'data' and return
    // macro F1 via Metrics<T>.  No weight update.
    // ------------------------------------------------------------------
    T compute_val_macro_f1(MLP<T, Activation>& mlp,
                           const Dataset<T>&   data)
    {
        std::vector<Vector<T>> y_true;
        std::vector<Vector<T>> y_pred;
        y_true.reserve(data.size());
        y_pred.reserve(data.size());

        for (std::size_t i = 0; i < data.size(); ++i) {
            y_true.push_back(data.labels[i]);
            // forward returns a const ref; copy into y_pred
            y_pred.push_back(Vector<T>(mlp.forward(data.inputs[i])));
        }

        const auto cm = Metrics<T>::compute_confusion_matrix(y_true, y_pred);
        return Metrics<T>::compute_macro_f1(cm);
    }
};

} // namespace mlp
