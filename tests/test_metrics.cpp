// tests/test_metrics.cpp
// Phase 8 — Tests for mlp::Metrics<T>.

#include "evaluation/metrics.hpp"
#include "core/vector.hpp"
#include "core/matrix.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using mlp::Matrix;
using mlp::Metrics;
using mlp::Vector;

// =========================================================================
// Helpers
// =========================================================================

// Build a one-hot Vector<double> of size n_classes with a 1 at position c.
static Vector<double> one_hot(std::size_t c, std::size_t n_classes) {
    Vector<double> v(n_classes, 0.0);
    v[c] = 1.0;
    return v;
}

// Build a "soft" prediction Vector where class c has value 0.9 and the
// rest share 0.1 / (n_classes - 1).  argmax always resolves to c.
static Vector<double> soft_pred(std::size_t c, std::size_t n_classes) {
    const double rest = (n_classes > 1)
                        ? 0.1 / static_cast<double>(n_classes - 1)
                        : 0.0;
    Vector<double> v(n_classes, rest);
    v[c] = 0.9;
    return v;
}

// =========================================================================
// Fixture: 10-sample, 3-class example with known ground truth.
//
// True:      [0,0,0, 1,1,1, 2,2,2,2]
// Predicted: [0,0,1, 1,1,0, 2,2,2,1]
//
// Confusion matrix:
//       P0  P1  P2
//  R0:   2   1   0      (row sum = 3)
//  R1:   1   2   0      (row sum = 3)
//  R2:   0   1   3      (row sum = 4)
//
// Accuracy     = 7 / 10 = 0.7
// Precision    = [2/3,  2/4,  3/3]  = [2/3, 0.5, 1.0]
// Recall       = [2/3,  2/3,  3/4]
// F1           = [2/3,  4/7,  6/7]
// Macro-F1     = (2/3 + 4/7 + 6/7) / 3 = 44/63
// =========================================================================

static Matrix<std::size_t> make_known_cm() {
    std::vector<Vector<double>> y_true, y_pred;

    const std::vector<std::size_t> true_cls = {0,0,0, 1,1,1, 2,2,2,2};
    const std::vector<std::size_t> pred_cls = {0,0,1, 1,1,0, 2,2,2,1};

    for (std::size_t t : true_cls) { y_true.push_back(one_hot(t, 3)); }
    for (std::size_t p : pred_cls) { y_pred.push_back(soft_pred(p, 3)); }

    return Metrics<double>::compute_confusion_matrix(y_true, y_pred);
}

// =========================================================================
// Confusion matrix structure
// =========================================================================

TEST(ConfusionMatrix, Dimensions) {
    auto cm = make_known_cm();
    EXPECT_EQ(cm.rows(), 3u);
    EXPECT_EQ(cm.cols(), 3u);
}

TEST(ConfusionMatrix, DiagonalValues) {
    auto cm = make_known_cm();
    EXPECT_EQ(cm(0, 0), 2u);
    EXPECT_EQ(cm(1, 1), 2u);
    EXPECT_EQ(cm(2, 2), 3u);
}

TEST(ConfusionMatrix, OffDiagonalValues) {
    auto cm = make_known_cm();
    EXPECT_EQ(cm(0, 1), 1u);   // true=0 predicted=1
    EXPECT_EQ(cm(0, 2), 0u);
    EXPECT_EQ(cm(1, 0), 1u);   // true=1 predicted=0
    EXPECT_EQ(cm(1, 2), 0u);
    EXPECT_EQ(cm(2, 0), 0u);
    EXPECT_EQ(cm(2, 1), 1u);   // true=2 predicted=1
}

TEST(ConfusionMatrix, TotalSamplesPreserved) {
    auto cm = make_known_cm();
    std::size_t total = 0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            total += cm(i, j);
    EXPECT_EQ(total, 10u);
}

TEST(ConfusionMatrix, PerfectPrediction) {
    std::vector<Vector<double>> y_true, y_pred;
    for (std::size_t c = 0; c < 3; ++c) {
        y_true.push_back(one_hot(c, 3));
        y_pred.push_back(soft_pred(c, 3));
    }
    auto cm = Metrics<double>::compute_confusion_matrix(y_true, y_pred);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(cm(i, j), (i == j) ? 1u : 0u);
}

TEST(ConfusionMatrix, EmptyYTrueThrows) {
    std::vector<Vector<double>> empty;
    std::vector<Vector<double>> y_pred = {soft_pred(0, 2)};
    EXPECT_THROW(
        { auto r = Metrics<double>::compute_confusion_matrix(empty, y_pred); (void)r; },
        std::invalid_argument);
}

TEST(ConfusionMatrix, SizeMismatchThrows) {
    std::vector<Vector<double>> y_true = {one_hot(0, 2), one_hot(1, 2)};
    std::vector<Vector<double>> y_pred = {soft_pred(0, 2)};
    EXPECT_THROW(
        { auto r = Metrics<double>::compute_confusion_matrix(y_true, y_pred); (void)r; },
        std::invalid_argument);
}

// =========================================================================
// Accuracy
// =========================================================================

TEST(AccuracyTest, KnownValue) {
    auto cm = make_known_cm();
    EXPECT_NEAR(Metrics<double>::compute_accuracy(cm), 0.7, 1e-12);
}

TEST(AccuracyTest, PerfectAccuracy) {
    Matrix<std::size_t> cm(3, 3);
    cm(0, 0) = 5; cm(1, 1) = 3; cm(2, 2) = 7;
    EXPECT_NEAR(Metrics<double>::compute_accuracy(cm), 1.0, 1e-12);
}

TEST(AccuracyTest, AllWrong) {
    // 2 classes, all predictions are wrong
    std::vector<Vector<double>> y_true = {one_hot(0,2), one_hot(0,2), one_hot(1,2), one_hot(1,2)};
    std::vector<Vector<double>> y_pred = {soft_pred(1,2), soft_pred(1,2), soft_pred(0,2), soft_pred(0,2)};
    auto cm = Metrics<double>::compute_confusion_matrix(y_true, y_pred);
    EXPECT_NEAR(Metrics<double>::compute_accuracy(cm), 0.0, 1e-12);
}

TEST(AccuracyTest, EmptyMatrixReturnsZero) {
    Matrix<std::size_t> cm(0, 0);
    EXPECT_NEAR(Metrics<double>::compute_accuracy(cm), 0.0, 1e-12);
}

// =========================================================================
// Precision per class
// =========================================================================

TEST(PrecisionTest, KnownValues) {
    auto cm = make_known_cm();
    auto prec = Metrics<double>::compute_precision_per_class(cm);

    ASSERT_EQ(prec.size(), 3u);
    EXPECT_NEAR(prec[0], 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(prec[1], 0.5,       1e-12);
    EXPECT_NEAR(prec[2], 1.0,       1e-12);
}

TEST(PrecisionTest, PerfectPrediction) {
    Matrix<std::size_t> cm(2, 2);
    cm(0, 0) = 4; cm(1, 1) = 6;
    auto prec = Metrics<double>::compute_precision_per_class(cm);
    EXPECT_NEAR(prec[0], 1.0, 1e-12);
    EXPECT_NEAR(prec[1], 1.0, 1e-12);
}

// =========================================================================
// Recall per class
// =========================================================================

TEST(RecallTest, KnownValues) {
    auto cm = make_known_cm();
    auto rec = Metrics<double>::compute_recall_per_class(cm);

    ASSERT_EQ(rec.size(), 3u);
    EXPECT_NEAR(rec[0], 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(rec[1], 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(rec[2], 3.0 / 4.0, 1e-12);
}

TEST(RecallTest, PerfectPrediction) {
    Matrix<std::size_t> cm(2, 2);
    cm(0, 0) = 3; cm(1, 1) = 5;
    auto rec = Metrics<double>::compute_recall_per_class(cm);
    EXPECT_NEAR(rec[0], 1.0, 1e-12);
    EXPECT_NEAR(rec[1], 1.0, 1e-12);
}

// =========================================================================
// F1 per class
// =========================================================================

TEST(F1Test, KnownValues) {
    auto cm = make_known_cm();
    auto f1 = Metrics<double>::compute_f1_per_class(cm);

    ASSERT_EQ(f1.size(), 3u);
    EXPECT_NEAR(f1[0], 2.0 / 3.0, 1e-12);    // 2/3
    EXPECT_NEAR(f1[1], 4.0 / 7.0, 1e-12);    // 4/7
    EXPECT_NEAR(f1[2], 6.0 / 7.0, 1e-12);    // 6/7
}

TEST(F1Test, AllWrongYieldsZero) {
    std::vector<Vector<double>> y_true = {one_hot(0,2), one_hot(1,2)};
    std::vector<Vector<double>> y_pred = {soft_pred(1,2), soft_pred(0,2)};
    auto cm = Metrics<double>::compute_confusion_matrix(y_true, y_pred);
    auto f1 = Metrics<double>::compute_f1_per_class(cm);
    for (double v : f1) {
        EXPECT_NEAR(v, 0.0, 1e-12);
    }
}

// =========================================================================
// Macro F1
// =========================================================================

TEST(MacroF1Test, KnownValue) {
    auto cm = make_known_cm();
    // Macro-F1 = (2/3 + 4/7 + 6/7) / 3 = 44/63
    const double expected = 44.0 / 63.0;
    EXPECT_NEAR(Metrics<double>::compute_macro_f1(cm), expected, 1e-12);
}

TEST(MacroF1Test, PerfectPrediction) {
    Matrix<std::size_t> cm(3, 3);
    cm(0, 0) = 5; cm(1, 1) = 3; cm(2, 2) = 7;
    EXPECT_NEAR(Metrics<double>::compute_macro_f1(cm), 1.0, 1e-12);
}

TEST(MacroF1Test, AllWrongYieldsZero) {
    Matrix<std::size_t> cm(2, 2);
    cm(0, 1) = 3;  // true=0 predicted=1
    cm(1, 0) = 5;  // true=1 predicted=0
    EXPECT_NEAR(Metrics<double>::compute_macro_f1(cm), 0.0, 1e-12);
}

TEST(MacroF1Test, EmptyMatrixReturnsZero) {
    Matrix<std::size_t> cm(0, 0);
    EXPECT_NEAR(Metrics<double>::compute_macro_f1(cm), 0.0, 1e-12);
}

// =========================================================================
// Edge cases — missing class (class never appears in y_true)
// =========================================================================

// 3 classes but class 1 never appears in y_true.
// True:      [0, 0, 2, 2]
// Predicted: [0, 0, 2, 2]
//
// CM (3×3):
//       P0  P1  P2
//  R0:   2   0   0
//  R1:   0   0   0    ← all zeros (class 1 absent)
//  R2:   0   0   2
//
// Precision_1 = 0/0 → 0
// Recall_1    = 0/0 → 0
// F1_1        = 0
// Macro-F1    = (1 + 0 + 1) / 3 = 2/3

TEST(EdgeCase, MissingClassYieldsZeroMetrics) {
    std::vector<Vector<double>> y_true = {
        one_hot(0, 3), one_hot(0, 3),
        one_hot(2, 3), one_hot(2, 3)
    };
    std::vector<Vector<double>> y_pred = {
        soft_pred(0, 3), soft_pred(0, 3),
        soft_pred(2, 3), soft_pred(2, 3)
    };
    auto cm = Metrics<double>::compute_confusion_matrix(y_true, y_pred);

    // Row 1 and column 1 are all zeros
    for (std::size_t j = 0; j < 3; ++j) { EXPECT_EQ(cm(1, j), 0u); }
    for (std::size_t i = 0; i < 3; ++i) { EXPECT_EQ(cm(i, 1), 0u); }

    auto prec = Metrics<double>::compute_precision_per_class(cm);
    auto rec  = Metrics<double>::compute_recall_per_class(cm);
    auto f1   = Metrics<double>::compute_f1_per_class(cm);

    // Class 1: all metrics should be 0 (no division-by-zero NaN)
    EXPECT_NEAR(prec[1], 0.0, 1e-12);
    EXPECT_NEAR(rec[1],  0.0, 1e-12);
    EXPECT_NEAR(f1[1],   0.0, 1e-12);

    // Classes 0 and 2 are perfect
    EXPECT_NEAR(prec[0], 1.0, 1e-12);
    EXPECT_NEAR(prec[2], 1.0, 1e-12);

    // Macro-F1 = 2/3
    EXPECT_NEAR(Metrics<double>::compute_macro_f1(cm), 2.0 / 3.0, 1e-12);
}

TEST(EdgeCase, MissingClassNoNaN) {
    std::vector<Vector<double>> y_true = {one_hot(0, 3), one_hot(2, 3)};
    std::vector<Vector<double>> y_pred = {soft_pred(0, 3), soft_pred(2, 3)};
    auto cm = Metrics<double>::compute_confusion_matrix(y_true, y_pred);

    auto prec = Metrics<double>::compute_precision_per_class(cm);
    auto rec  = Metrics<double>::compute_recall_per_class(cm);
    auto f1   = Metrics<double>::compute_f1_per_class(cm);

    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_TRUE(std::isfinite(prec[k])) << "  Precision[" << k << "] is not finite";
        EXPECT_TRUE(std::isfinite(rec[k]))  << "  Recall["    << k << "] is not finite";
        EXPECT_TRUE(std::isfinite(f1[k]))   << "  F1["        << k << "] is not finite";
    }
}

TEST(EdgeCase, SingleClass) {
    // 1-class problem: every sample is class 0
    std::vector<Vector<double>> y_true = {one_hot(0,1), one_hot(0,1), one_hot(0,1)};
    std::vector<Vector<double>> y_pred = {soft_pred(0,1), soft_pred(0,1), soft_pred(0,1)};
    auto cm = Metrics<double>::compute_confusion_matrix(y_true, y_pred);

    EXPECT_EQ(cm(0, 0), 3u);
    EXPECT_NEAR(Metrics<double>::compute_accuracy(cm),  1.0, 1e-12);
    EXPECT_NEAR(Metrics<double>::compute_macro_f1(cm),  1.0, 1e-12);
}

// =========================================================================
// Metrics are finite and in valid ranges
// =========================================================================

TEST(Validity, AllMetricsInValidRange) {
    auto cm = make_known_cm();

    const double acc     = Metrics<double>::compute_accuracy(cm);
    const double mf1     = Metrics<double>::compute_macro_f1(cm);
    const auto   prec    = Metrics<double>::compute_precision_per_class(cm);
    const auto   rec     = Metrics<double>::compute_recall_per_class(cm);
    const auto   f1      = Metrics<double>::compute_f1_per_class(cm);

    EXPECT_GE(acc, 0.0);  EXPECT_LE(acc, 1.0);
    EXPECT_GE(mf1, 0.0);  EXPECT_LE(mf1, 1.0);

    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_GE(prec[k], 0.0);  EXPECT_LE(prec[k], 1.0);
        EXPECT_GE(rec[k],  0.0);  EXPECT_LE(rec[k],  1.0);
        EXPECT_GE(f1[k],   0.0);  EXPECT_LE(f1[k],   1.0);
        EXPECT_TRUE(std::isfinite(prec[k]));
        EXPECT_TRUE(std::isfinite(rec[k]));
        EXPECT_TRUE(std::isfinite(f1[k]));
    }
}

// =========================================================================
// 2-class (binary) sanity check
// =========================================================================

// True:      [0, 0, 0, 1, 1]
// Predicted: [0, 0, 1, 1, 1]
//
// CM:   P0  P1
//  R0:   2   1
//  R1:   0   2
//
// Precision = [2/2=1.0, 2/3]
// Recall    = [2/3,     2/2=1.0]
// F1        = [2*(1.0*2/3)/(1.0+2/3) = 4/5,  2*(2/3*1.0)/(2/3+1.0) = 4/5]
// Macro-F1  = 4/5

TEST(BinaryClass, KnownValues) {
    std::vector<Vector<double>> y_true = {
        one_hot(0,2), one_hot(0,2), one_hot(0,2), one_hot(1,2), one_hot(1,2)
    };
    std::vector<Vector<double>> y_pred = {
        soft_pred(0,2), soft_pred(0,2), soft_pred(1,2),
        soft_pred(1,2), soft_pred(1,2)
    };
    auto cm = Metrics<double>::compute_confusion_matrix(y_true, y_pred);

    EXPECT_EQ(cm(0, 0), 2u);  EXPECT_EQ(cm(0, 1), 1u);
    EXPECT_EQ(cm(1, 0), 0u);  EXPECT_EQ(cm(1, 1), 2u);

    auto prec = Metrics<double>::compute_precision_per_class(cm);
    auto rec  = Metrics<double>::compute_recall_per_class(cm);
    auto f1   = Metrics<double>::compute_f1_per_class(cm);

    EXPECT_NEAR(prec[0], 1.0,        1e-12);
    EXPECT_NEAR(prec[1], 2.0 / 3.0,  1e-12);
    EXPECT_NEAR(rec[0],  2.0 / 3.0,  1e-12);
    EXPECT_NEAR(rec[1],  1.0,        1e-12);
    EXPECT_NEAR(f1[0],   4.0 / 5.0,  1e-12);
    EXPECT_NEAR(f1[1],   4.0 / 5.0,  1e-12);
    EXPECT_NEAR(Metrics<double>::compute_macro_f1(cm), 4.0 / 5.0, 1e-12);
}
