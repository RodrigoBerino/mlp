// tests/test_minibatch.cpp
// Phase 9 — Tests for mini-batch training in mlp::Trainer.

#include "mlp/trainer.hpp"
#include "mlp/mlp.hpp"
#include "activations/sigmoid.hpp"
#include "activations/relu.hpp"

#include <gtest/gtest.h>
#include <cmath>

using mlp::Dataset;
using mlp::MLP;
using mlp::ReLU;
using mlp::Sigmoid;
using mlp::Trainer;
using mlp::Vector;

// =========================================================================
// Fixture datasets (same as test_training.cpp)
// =========================================================================

static Dataset<double> make_binary_dataset() {
    Dataset<double> ds;
    ds.inputs = {
        Vector<double>{1.0, 0.0},
        Vector<double>{0.0, 1.0},
        Vector<double>{0.9, 0.1},
        Vector<double>{0.1, 0.9}
    };
    ds.labels = {
        Vector<double>{1.0, 0.0},
        Vector<double>{0.0, 1.0},
        Vector<double>{1.0, 0.0},
        Vector<double>{0.0, 1.0}
    };
    return ds;
}

static Dataset<double> make_identity_dataset() {
    Dataset<double> ds;
    ds.inputs = {
        Vector<double>{1.0, 0.0, 0.0},
        Vector<double>{0.0, 1.0, 0.0},
        Vector<double>{0.0, 0.0, 1.0}
    };
    ds.labels = {
        Vector<double>{1.0, 0.0, 0.0},
        Vector<double>{0.0, 1.0, 0.0},
        Vector<double>{0.0, 0.0, 1.0}
    };
    return ds;
}

// =========================================================================
// Backward compatibility: batch_size=1 ≡ original SGD
// =========================================================================

TEST(MiniBatchBackwardCompat, ExplicitBatch1MatchesDefault) {
    Dataset<double> data = make_binary_dataset();

    // Default call (no batch_size) — original SGD
    MLP<double, Sigmoid<double>> mlp1({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t1;
    t1.train(mlp1, data, 20, 0.05);

    // Explicit batch_size=1
    MLP<double, Sigmoid<double>> mlp2({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t2;
    t2.train(mlp2, data, 20, 0.05, 1);

    for (std::size_t i = 0; i < 20; ++i) {
        EXPECT_NEAR(t1.epoch_losses()[i], t2.epoch_losses()[i], 1e-12)
            << "  at epoch " << i;
        EXPECT_NEAR(t1.epoch_accuracies()[i], t2.epoch_accuracies()[i], 1e-12)
            << "  at epoch " << i;
    }
}

// =========================================================================
// Validation
// =========================================================================

TEST(MiniBatchValidation, ZeroBatchSizeThrows) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    EXPECT_THROW(
        trainer.train(mlp, make_binary_dataset(), 10, 0.05, 0),
        std::invalid_argument);
}

TEST(MiniBatchValidation, ZeroEpochsStillThrows) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    EXPECT_THROW(
        trainer.train(mlp, make_binary_dataset(), 0, 0.05, 2),
        std::invalid_argument);
}

TEST(MiniBatchValidation, EmptyDatasetStillThrows) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> empty;
    EXPECT_THROW(
        trainer.train(mlp, empty, 10, 0.05, 2),
        std::invalid_argument);
}

// =========================================================================
// History
// =========================================================================

TEST(MiniBatchHistory, LengthMatchesEpochs) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 20, 0.05, 2);

    EXPECT_EQ(trainer.epoch_losses().size(), 20u);
    EXPECT_EQ(trainer.epoch_accuracies().size(), 20u);
}

TEST(MiniBatchHistory, AllLossesFinite) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 30, 0.05, 2);

    for (double l : trainer.epoch_losses()) {
        EXPECT_TRUE(std::isfinite(l)) << "  non-finite loss: " << l;
        EXPECT_GE(l, 0.0);
    }
}

TEST(MiniBatchHistory, ClearedOnNewTrain) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 5, 0.05, 2);
    trainer.train(mlp, make_binary_dataset(), 10, 0.05, 2);
    EXPECT_EQ(trainer.epoch_losses().size(), 10u);
}

// =========================================================================
// Uneven batch: last batch is smaller
// =========================================================================

// 4 samples, batch_size=3 → batches of {3, 1}.  No sample should be lost.
TEST(MiniBatchUneven, LastBatchSmallerThanBatchSize) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 20, 0.05, 3);

    EXPECT_EQ(trainer.epoch_losses().size(), 20u);
    for (double l : trainer.epoch_losses()) {
        EXPECT_TRUE(std::isfinite(l));
        EXPECT_GE(l, 0.0);
    }
}

// batch_size > N → all samples in one batch (full-batch gradient descent).
TEST(MiniBatchUneven, BatchSizeLargerThanDataset) {
    MLP<double, Sigmoid<double>> mlp({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 50, 0.1, 100);

    EXPECT_EQ(trainer.epoch_losses().size(), 50u);
    for (double l : trainer.epoch_losses()) {
        EXPECT_TRUE(std::isfinite(l));
        EXPECT_GE(l, 0.0);
    }
}

// =========================================================================
// Loss convergence
// =========================================================================

TEST(MiniBatchLoss, DecreasesOverTrainingBatch2) {
    MLP<double, Sigmoid<double>> mlp({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 200, 0.1, 2);

    const auto& losses = trainer.epoch_losses();
    const std::size_t n = losses.size();

    double early = 0.0;
    for (std::size_t i = 0; i < 5; ++i) { early += losses[i]; }
    early /= 5.0;

    double late = 0.0;
    for (std::size_t i = n - 5; i < n; ++i) { late += losses[i]; }
    late /= 5.0;

    EXPECT_LT(late, early)
        << "  early_mean=" << early << "  late_mean=" << late;
}

TEST(MiniBatchLoss, FullBatchConverges) {
    // batch_size == dataset size → full-batch gradient descent
    Dataset<double> data = make_binary_dataset();
    MLP<double, Sigmoid<double>> mlp({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, data, 300, 0.1, data.size());

    const auto& losses = trainer.epoch_losses();
    EXPECT_LT(losses.back(), losses.front())
        << "  full-batch did not converge";
}

TEST(MiniBatchLoss, BothSGDAndMiniBatchConverge) {
    Dataset<double> data = make_binary_dataset();

    MLP<double, Sigmoid<double>> mlp_sgd({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> t_sgd;
    t_sgd.train(mlp_sgd, data, 200, 0.1, 1);

    MLP<double, Sigmoid<double>> mlp_mb({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> t_mb;
    t_mb.train(mlp_mb, data, 200, 0.1, 2);

    const auto& l_sgd = t_sgd.epoch_losses();
    const auto& l_mb  = t_mb.epoch_losses();

    EXPECT_LT(l_sgd.back(), l_sgd.front()) << "  SGD did not converge";
    EXPECT_LT(l_mb.back(),  l_mb.front())  << "  Mini-batch did not converge";
}

// =========================================================================
// Shuffle — order changes loss trajectory
// =========================================================================

// Same MLP seed, same data, batch_size=2, but shuffle=true vs false.
// The different sample ordering within batches should produce a different
// loss trajectory.
TEST(MiniBatchShuffle, ShuffleChangesTrajVsNoShuffle) {
    Dataset<double> data = make_binary_dataset();

    MLP<double, Sigmoid<double>> mlp1({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t1;
    t1.train(mlp1, data, 15, 0.05, 2, false);   // no shuffle

    MLP<double, Sigmoid<double>> mlp2({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t2;
    t2.train(mlp2, data, 15, 0.05, 2, true, 7); // shuffle with seed 7

    bool any_diff = false;
    for (std::size_t i = 0; i < 15; ++i) {
        if (std::abs(t1.epoch_losses()[i] - t2.epoch_losses()[i]) > 1e-12) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff) << "Shuffle produced identical loss trajectory";
}

// =========================================================================
// Reproducibility with shuffle
// =========================================================================

TEST(MiniBatchReproducibility, SameSeedSameHistory) {
    Dataset<double> data = make_binary_dataset();

    MLP<double, Sigmoid<double>> mlp1({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t1;
    t1.train(mlp1, data, 15, 0.05, 2, true, 99);

    MLP<double, Sigmoid<double>> mlp2({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t2;
    t2.train(mlp2, data, 15, 0.05, 2, true, 99);  // same seed

    for (std::size_t i = 0; i < 15; ++i) {
        EXPECT_NEAR(t1.epoch_losses()[i], t2.epoch_losses()[i], 1e-12)
            << "  at epoch " << i;
    }
}

TEST(MiniBatchReproducibility, DifferentSeedsDifferentHistory) {
    Dataset<double> data = make_binary_dataset();

    MLP<double, Sigmoid<double>> mlp1({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t1;
    t1.train(mlp1, data, 15, 0.05, 2, true, 1);

    MLP<double, Sigmoid<double>> mlp2({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> t2;
    t2.train(mlp2, data, 15, 0.05, 2, true, 99);  // different seed

    bool any_diff = false;
    for (std::size_t i = 0; i < 15; ++i) {
        if (std::abs(t1.epoch_losses()[i] - t2.epoch_losses()[i]) > 1e-12) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff) << "Different shuffle seeds produced identical histories";
}

// =========================================================================
// Overfitting with mini-batch
// =========================================================================

TEST(MiniBatchOverfit, IdentityDatasetBatch1) {
    MLP<double, Sigmoid<double>> mlp({3, 16, 3}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_identity_dataset(), 2000, 0.1, 1);

    EXPECT_NEAR(trainer.epoch_accuracies().back(), 1.0, 1e-9)
        << "  Expected 100% accuracy, got: " << trainer.epoch_accuracies().back();
}

TEST(MiniBatchOverfit, IdentityDatasetFullBatch) {
    // Full-batch (batch_size == N) should also overfit this trivial dataset.
    Dataset<double> data = make_identity_dataset();
    MLP<double, Sigmoid<double>> mlp({3, 16, 3}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, data, 3000, 0.1, data.size());

    EXPECT_NEAR(trainer.epoch_accuracies().back(), 1.0, 1e-9)
        << "  Expected 100% accuracy with full-batch, got: "
        << trainer.epoch_accuracies().back();
}

// =========================================================================
// Template: ReLU with mini-batch
// =========================================================================

TEST(MiniBatchTemplate, WorksWithReLU) {
    MLP<double, ReLU<double>> mlp({2, 8, 2}, 42);
    Trainer<double, ReLU<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 50, 0.01, 2);

    for (double l : trainer.epoch_losses()) {
        EXPECT_TRUE(std::isfinite(l));
        EXPECT_GE(l, 0.0);
    }
}
