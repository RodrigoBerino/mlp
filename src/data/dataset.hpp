// src/data/dataset.hpp
// Phase 10 — train_validation_split: splits a Dataset<T> into train and
//            validation subsets with reproducible shuffling.
//
// Usage:
//   auto [train, val] = mlp::train_validation_split(ds, 0.8f, 42u);
//
//   ratio — fraction of samples assigned to training (must be in (0, 1))
//           e.g. 0.8 → 80 % train, 20 % validation
//   seed  — RNG seed for reproducible shuffling before the split
//
// The original dataset is left unmodified.
// Throws std::invalid_argument on bad ratio or empty dataset.

#pragma once

#include "mlp/trainer.hpp"   // mlp::Dataset<T>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mlp {

template<typename T>
std::pair<Dataset<T>, Dataset<T>>
train_validation_split(const Dataset<T>& dataset,
                       T                 ratio,
                       std::uint32_t     seed = 42u)
{
    if (ratio <= T{0} || ratio >= T{1}) {
        throw std::invalid_argument(
            "train_validation_split: ratio must be in (0, 1)");
    }
    dataset.validate();
    const std::size_t n = dataset.size();
    if (n == 0) {
        throw std::invalid_argument(
            "train_validation_split: empty dataset");
    }

    // Shuffle a fresh index permutation with the given seed.
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    const std::size_t n_train = static_cast<std::size_t>(
        ratio * static_cast<T>(n));
    const std::size_t n_val = n - n_train;

    Dataset<T> train_set;
    train_set.inputs.reserve(n_train);
    train_set.labels.reserve(n_train);

    Dataset<T> val_set;
    val_set.inputs.reserve(n_val);
    val_set.labels.reserve(n_val);

    for (std::size_t i = 0; i < n_train; ++i) {
        train_set.inputs.push_back(dataset.inputs[idx[i]]);
        train_set.labels.push_back(dataset.labels[idx[i]]);
    }
    for (std::size_t i = n_train; i < n; ++i) {
        val_set.inputs.push_back(dataset.inputs[idx[i]]);
        val_set.labels.push_back(dataset.labels[idx[i]]);
    }

    return {std::move(train_set), std::move(val_set)};
}

} // namespace mlp
