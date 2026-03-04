// src/data/data_pipeline.hpp
// Phase 7 — DataPipeline: MinMax normalisation + one-hot encoding + train/val/test split.
//
// Typical usage:
//
//   DataPipeline<double> pipe;
//   auto [train, val, test] = pipe.load_and_split("train.csv");
//   // train, val, test are mlp::Dataset<double> ready for Trainer.

#pragma once

#include "data/csv_reader.hpp"
#include "mlp/trainer.hpp"     // mlp::Dataset<T>, mlp::Vector<T>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp {

template<typename T>
class DataPipeline {
public:
    // ---------------------------------------------------------------
    // Output: three ready-to-use Dataset<T> objects.
    // ---------------------------------------------------------------
    struct Splits {
        Dataset<T> train;
        Dataset<T> val;
        Dataset<T> test;
    };

    // ---------------------------------------------------------------
    // load_and_split()
    //
    //   path       — CSV file path
    //   label_col  — 0-based column index for the label.
    //                SIZE_MAX (default) → last column.
    //   train_frac — fraction of samples for training   (default 0.70)
    //   val_frac   — fraction of samples for validation (default 0.15)
    //   seed       — RNG seed for shuffling
    //
    // MinMax parameters are fit on the TRAINING split only, then
    // applied to all three splits (no data leakage).
    //
    // Class names are sorted alphabetically; that order defines the
    // position of the '1' in each one-hot vector.
    // ---------------------------------------------------------------
    Splits load_and_split(
        const std::string& path,
        std::size_t        label_col  = std::numeric_limits<std::size_t>::max(),
        T                  train_frac = T{0.70},
        T                  val_frac   = T{0.15},
        std::uint32_t      seed       = 42
    ) {
        if (train_frac <= T{0} || val_frac <= T{0} ||
            train_frac + val_frac >= T{1}) {
            throw std::invalid_argument(
                "DataPipeline: train_frac and val_frac must be > 0 "
                "and their sum < 1");
        }

        CsvData csv = read_csv(path);
        if (csv.rows.empty()) {
            throw std::invalid_argument(
                "DataPipeline: CSV has no data rows");
        }

        const std::size_t n_cols = csv.header.size();
        if (label_col == std::numeric_limits<std::size_t>::max()) {
            label_col = n_cols - 1;
        }
        if (label_col >= n_cols) {
            throw std::invalid_argument(
                "DataPipeline: label_col is out of range");
        }

        // ----------------------------------------------------------
        // 1. Collect sorted class names from every row.
        // ----------------------------------------------------------
        std::set<std::string> label_set;
        for (const auto& row : csv.rows) {
            label_set.insert(row[label_col]);
        }
        class_names_.assign(label_set.begin(), label_set.end()); // sorted

        // ----------------------------------------------------------
        // 2. Parse rows → raw numeric features + string labels.
        // ----------------------------------------------------------
        const std::size_t n_feat = n_cols - 1;   // one column is the label
        const std::size_t n      = csv.rows.size();

        std::vector<std::vector<T>> all_features;
        std::vector<std::string>    all_labels;
        all_features.reserve(n);
        all_labels.reserve(n);

        for (const auto& row : csv.rows) {
            std::vector<T> feat;
            feat.reserve(n_feat);
            for (std::size_t j = 0; j < n_cols; ++j) {
                if (j == label_col) { continue; }
                feat.push_back(static_cast<T>(std::stod(row[j])));
            }
            all_features.push_back(std::move(feat));
            all_labels.push_back(row[label_col]);
        }

        // ----------------------------------------------------------
        // 3. Shuffle indices with the given seed.
        // ----------------------------------------------------------
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);

        // ----------------------------------------------------------
        // 4. Compute split sizes.
        // ----------------------------------------------------------
        const std::size_t n_train = static_cast<std::size_t>(train_frac * static_cast<T>(n));
        const std::size_t n_val   = static_cast<std::size_t>(val_frac   * static_cast<T>(n));
        const std::size_t n_test  = n - n_train - n_val;

        // ----------------------------------------------------------
        // 5. Fit MinMax on training features ONLY.
        // ----------------------------------------------------------
        {
            std::vector<std::vector<T>> train_raw;
            train_raw.reserve(n_train);
            for (std::size_t i = 0; i < n_train; ++i) {
                train_raw.push_back(all_features[idx[i]]);
            }
            fit(train_raw);
        }

        // ----------------------------------------------------------
        // 6. Build Dataset<T> for each split.
        // ----------------------------------------------------------
        auto make_split = [&](std::size_t start, std::size_t count) -> Dataset<T> {
            Dataset<T> ds;
            ds.inputs.reserve(count);
            ds.labels.reserve(count);
            for (std::size_t i = start; i < start + count; ++i) {
                ds.inputs.push_back(transform(all_features[idx[i]]));
                ds.labels.push_back(encode(all_labels[idx[i]]));
            }
            return ds;
        };

        Splits splits;
        splits.train = make_split(0,                    n_train);
        splits.val   = make_split(n_train,              n_val);
        splits.test  = make_split(n_train + n_val,      n_test);
        return splits;
    }

    // ---------------------------------------------------------------
    // Metadata accessors (valid after load_and_split)
    // ---------------------------------------------------------------
    [[nodiscard]] const std::vector<std::string>& class_names() const noexcept {
        return class_names_;
    }
    [[nodiscard]] const std::vector<T>& feature_min() const noexcept { return min_; }
    [[nodiscard]] const std::vector<T>& feature_max() const noexcept { return max_; }
    [[nodiscard]] std::size_t num_features() const noexcept { return min_.size(); }
    [[nodiscard]] std::size_t num_classes()  const noexcept { return class_names_.size(); }

    // ---------------------------------------------------------------
    // Public helpers (exposed for unit testing)
    // ---------------------------------------------------------------

    // fit(): learn per-feature min/max from a set of raw rows.
    void fit(const std::vector<std::vector<T>>& features) {
        if (features.empty()) {
            throw std::invalid_argument("DataPipeline::fit: empty feature set");
        }
        const std::size_t n_feat = features[0].size();
        min_.resize(n_feat);
        max_.resize(n_feat);
        for (std::size_t j = 0; j < n_feat; ++j) {
            min_[j] = features[0][j];
            max_[j] = features[0][j];
        }

        for (const auto& row : features) {
            for (std::size_t j = 0; j < n_feat; ++j) {
                if (row[j] < min_[j]) { min_[j] = row[j]; }
                if (row[j] > max_[j]) { max_[j] = row[j]; }
            }
        }
    }

    // transform(): apply MinMax normalisation to a single raw row.
    // If max == min for a feature, the normalised value is 0.
    [[nodiscard]] Vector<T> transform(const std::vector<T>& row) const {
        Vector<T> out(row.size(), T{0});
        for (std::size_t j = 0; j < row.size(); ++j) {
            const T range = max_[j] - min_[j];
            out[j] = (range > T{0}) ? (row[j] - min_[j]) / range : T{0};
        }
        return out;
    }

    // encode(): map a class name to a one-hot Vector<T>.
    // Throws if the label is not in class_names_.
    [[nodiscard]] Vector<T> encode(const std::string& label) const {
        Vector<T> v(class_names_.size(), T{0});
        auto it = std::find(class_names_.begin(), class_names_.end(), label);
        if (it == class_names_.end()) {
            throw std::invalid_argument(
                "DataPipeline::encode: unknown label '" + label + "'");
        }
        v[static_cast<std::size_t>(it - class_names_.begin())] = T{1};
        return v;
    }

private:
    std::vector<std::string> class_names_;
    std::vector<T>           min_;
    std::vector<T>           max_;
};

} // namespace mlp
