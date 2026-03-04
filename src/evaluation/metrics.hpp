// src/evaluation/metrics.hpp
// Phase 8 — Multiclass evaluation metrics.
//
// All methods are static; Metrics<T> holds no state.
//
// Confusion matrix convention:
//   cm(i, j)  =  number of samples with TRUE class i PREDICTED as class j
//   rows = true class,  columns = predicted class
//
// Per-class metrics (k = class index):
//   TP_k  = cm(k, k)
//   FP_k  = sum_col(k) - TP_k    (predicted k but were not k)
//   FN_k  = sum_row(k) - TP_k    (were k but predicted as something else)
//
//   Precision(k) = TP_k / (TP_k + FP_k)
//   Recall(k)    = TP_k / (TP_k + FN_k)
//   F1(k)        = 2 * Precision(k) * Recall(k) / (Precision(k) + Recall(k))
//
//   Macro-F1     = mean of F1 across all classes
//
// Division-by-zero: any zero denominator yields 0 for that metric.

#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mlp {

template<typename T>
class Metrics {
public:
    // ------------------------------------------------------------------
    // compute_confusion_matrix()
    //
    // y_true — one-hot ground-truth labels, shape [N x C]
    // y_pred — network outputs (softmax), shape [N x C]
    //
    // Returns a C×C matrix where entry (i,j) = # samples with true class i
    // predicted as class j.
    // Throws std::invalid_argument on size mismatch or empty input.
    // ------------------------------------------------------------------
    [[nodiscard]] static Matrix<std::size_t> compute_confusion_matrix(
        const std::vector<Vector<T>>& y_true,
        const std::vector<Vector<T>>& y_pred)
    {
        if (y_true.empty()) {
            throw std::invalid_argument(
                "Metrics::compute_confusion_matrix: empty y_true");
        }
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument(
                "Metrics::compute_confusion_matrix: y_true and y_pred "
                "have different sizes");
        }

        const std::size_t n_classes = y_true[0].size();
        if (n_classes == 0) {
            throw std::invalid_argument(
                "Metrics::compute_confusion_matrix: zero classes");
        }

        Matrix<std::size_t> cm(n_classes, n_classes);   // zero-initialised

        for (std::size_t i = 0; i < y_true.size(); ++i) {
            const std::size_t true_cls = argmax(y_true[i]);
            const std::size_t pred_cls = argmax(y_pred[i]);
            cm(true_cls, pred_cls) += 1u;
        }
        return cm;
    }

    // ------------------------------------------------------------------
    // compute_accuracy()
    //   = sum(diagonal) / total_samples
    // Returns 0 if the matrix is empty or has zero total.
    // ------------------------------------------------------------------
    [[nodiscard]] static T compute_accuracy(const Matrix<std::size_t>& cm) {
        const std::size_t n = cm.rows();
        if (n == 0) { return T{0}; }

        std::size_t correct = 0;
        std::size_t total   = 0;
        for (std::size_t i = 0; i < n; ++i) {
            correct += cm(i, i);
            for (std::size_t j = 0; j < n; ++j) {
                total += cm(i, j);
            }
        }
        return (total == 0u) ? T{0}
                             : static_cast<T>(correct) / static_cast<T>(total);
    }

    // ------------------------------------------------------------------
    // compute_precision_per_class()
    //   Precision(k) = cm(k,k) / sum_col(k)
    // Zero if no samples were predicted as class k.
    // ------------------------------------------------------------------
    [[nodiscard]] static std::vector<T> compute_precision_per_class(
        const Matrix<std::size_t>& cm)
    {
        const std::size_t n = cm.rows();
        std::vector<T> prec(n, T{0});
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t col_sum = 0u;
            for (std::size_t i = 0; i < n; ++i) { col_sum += cm(i, k); }
            prec[k] = (col_sum == 0u)
                      ? T{0}
                      : static_cast<T>(cm(k, k)) / static_cast<T>(col_sum);
        }
        return prec;
    }

    // ------------------------------------------------------------------
    // compute_recall_per_class()
    //   Recall(k) = cm(k,k) / sum_row(k)
    // Zero if class k has no true samples.
    // ------------------------------------------------------------------
    [[nodiscard]] static std::vector<T> compute_recall_per_class(
        const Matrix<std::size_t>& cm)
    {
        const std::size_t n = cm.rows();
        std::vector<T> rec(n, T{0});
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t row_sum = 0u;
            for (std::size_t j = 0; j < n; ++j) { row_sum += cm(k, j); }
            rec[k] = (row_sum == 0u)
                     ? T{0}
                     : static_cast<T>(cm(k, k)) / static_cast<T>(row_sum);
        }
        return rec;
    }

    // ------------------------------------------------------------------
    // compute_f1_per_class()
    //   F1(k) = 2 * P(k) * R(k) / (P(k) + R(k))
    // Zero if both precision and recall are 0.
    // ------------------------------------------------------------------
    [[nodiscard]] static std::vector<T> compute_f1_per_class(
        const Matrix<std::size_t>& cm)
    {
        const std::vector<T> prec = compute_precision_per_class(cm);
        const std::vector<T> rec  = compute_recall_per_class(cm);
        const std::size_t n = cm.rows();
        std::vector<T> f1(n, T{0});
        for (std::size_t k = 0; k < n; ++k) {
            const T denom = prec[k] + rec[k];
            f1[k] = (denom <= T{0}) ? T{0}
                                    : T{2} * prec[k] * rec[k] / denom;
        }
        return f1;
    }

    // ------------------------------------------------------------------
    // compute_macro_f1()
    //   Simple mean of per-class F1 scores.
    // Returns 0 for an empty matrix.
    // ------------------------------------------------------------------
    [[nodiscard]] static T compute_macro_f1(const Matrix<std::size_t>& cm) {
        const std::size_t n = cm.rows();
        if (n == 0) { return T{0}; }
        const std::vector<T> f1 = compute_f1_per_class(cm);
        T sum = T{0};
        for (T v : f1) { sum += v; }
        return sum / static_cast<T>(n);
    }

private:
    // argmax of a Vector<T> — returns index of the largest element.
    static std::size_t argmax(const Vector<T>& v) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < v.size(); ++i) {
            if (v[i] > v[best]) { best = i; }
        }
        return best;
    }
};

} // namespace mlp
