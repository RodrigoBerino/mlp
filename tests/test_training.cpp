// tests/test_training.cpp
// Phase 6 — Tests for mlp::Trainer and mlp::Dataset.

#include "mlp/trainer.hpp"
#include "mlp/mlp.hpp"
#include "activations/sigmoid.hpp"
#include "activations/relu.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

using mlp::MLP;
using mlp::Dataset;
using mlp::Trainer;
using mlp::Sigmoid;
using mlp::ReLU;
using mlp::Vector;

// =========================================================================
// Fixture datasets
// =========================================================================

// 4-sample, 2-class linearly separable dataset.
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

// 3-sample, 3-class "identity" dataset — trivially overfittable.
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
// Dataset validation
// =========================================================================

TEST(DatasetTest, SizeAndEmpty) {
    Dataset<double> ds = make_binary_dataset();
    EXPECT_EQ(ds.size(), 4u);
    EXPECT_FALSE(ds.empty());
}

TEST(DatasetTest, EmptyIsEmpty) {
    Dataset<double> ds;
    EXPECT_TRUE(ds.empty());
    EXPECT_EQ(ds.size(), 0u);
}

TEST(DatasetTest, MismatchedSizesThrows) {
    Dataset<double> ds;
    ds.inputs = {Vector<double>{1.0, 0.0}};
    ds.labels = {};
    EXPECT_THROW(ds.validate(), std::invalid_argument);
}

// =========================================================================
// Trainer — construction & validation
// =========================================================================

TEST(TrainerValidation, EmptyDatasetThrows) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> empty;
    EXPECT_THROW(trainer.train(mlp, empty, 10, 0.1), std::invalid_argument);
}

TEST(TrainerValidation, ZeroEpochsThrows) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    EXPECT_THROW(trainer.train(mlp, make_binary_dataset(), 0, 0.1),
                 std::invalid_argument);
}

TEST(TrainerValidation, EvaluateEmptyDatasetThrows) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> empty;
    EXPECT_THROW({ auto r = trainer.evaluate(mlp, empty); (void)r; },
                 std::invalid_argument);
}

// =========================================================================
// Trainer — history length
// =========================================================================

TEST(TrainerHistory, LengthMatchesEpochs) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 20, 0.05);

    EXPECT_EQ(trainer.epoch_losses().size(), 20u);
    EXPECT_EQ(trainer.epoch_accuracies().size(), 20u);
}

TEST(TrainerHistory, ClearedOnNewTrain) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 5, 0.05);
    trainer.train(mlp, make_binary_dataset(), 10, 0.05);

    // Second train() clears history, so length == 10
    EXPECT_EQ(trainer.epoch_losses().size(), 10u);
}

TEST(TrainerHistory, AllLossesAreFinite) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 30, 0.05);

    for (double l : trainer.epoch_losses()) {
        EXPECT_TRUE(std::isfinite(l)) << "  non-finite loss: " << l;
        EXPECT_GE(l, 0.0);
    }
}

TEST(TrainerHistory, AllAccuraciesInRange) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 30, 0.05);

    for (double acc : trainer.epoch_accuracies()) {
        EXPECT_GE(acc, 0.0);
        EXPECT_LE(acc, 1.0);
    }
}

// =========================================================================
// Trainer — loss decreases
// =========================================================================

TEST(TrainerLoss, DecreasesOverTraining) {
    // Compare mean loss of first 5 epochs vs last 5 epochs.
    MLP<double, Sigmoid<double>> mlp({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 200, 0.1);

    const auto& losses = trainer.epoch_losses();
    const std::size_t n = losses.size();

    double early_mean = 0.0;
    for (std::size_t i = 0; i < 5; ++i) { early_mean += losses[i]; }
    early_mean /= 5.0;

    double late_mean = 0.0;
    for (std::size_t i = n - 5; i < n; ++i) { late_mean += losses[i]; }
    late_mean /= 5.0;

    EXPECT_LT(late_mean, early_mean)
        << "  early_mean=" << early_mean << "  late_mean=" << late_mean;
}

TEST(TrainerLoss, FirstEpochLossIsPositive) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 1, 0.01);
    EXPECT_GT(trainer.epoch_losses()[0], 0.0);
}

// =========================================================================
// Trainer — accuracy
// =========================================================================

TEST(TrainerAccuracy, IncreasesOrSaturatesHighAfterLongTraining) {
    MLP<double, Sigmoid<double>> mlp({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    trainer.train(mlp, make_binary_dataset(), 500, 0.1);

    // After 500 epochs on a 4-sample separable dataset, accuracy should be high
    const auto& accs = trainer.epoch_accuracies();
    EXPECT_GE(accs.back(), 0.75)
        << "  final accuracy too low: " << accs.back();
}

TEST(TrainerAccuracy, ComputeAccuracyIsConsistentWithHistory) {
    // After training, compute_accuracy on the training set should match
    // the last logged accuracy.
    MLP<double, Sigmoid<double>> mlp({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> data = make_binary_dataset();
    trainer.train(mlp, data, 300, 0.1);

    const double logged_acc = trainer.epoch_accuracies().back();
    const double computed_acc = trainer.compute_accuracy(mlp, data);

    // They may differ slightly since another forward pass is run,
    // but should be the same (no weights changed since last epoch's last step).
    EXPECT_NEAR(logged_acc, computed_acc, 1e-9);
}

// =========================================================================
// Trainer — evaluate
// =========================================================================

TEST(TrainerEvaluate, ReturnsPositiveLoss) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> data = make_binary_dataset();
    const double loss = trainer.evaluate(mlp, data);
    EXPECT_GT(loss, 0.0);
    EXPECT_TRUE(std::isfinite(loss));
}

TEST(TrainerEvaluate, DoesNotChangeWeights) {
    MLP<double, Sigmoid<double>> mlp({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> data = make_binary_dataset();

    // Save weight snapshot
    const double w00_before = mlp.layer(0).W()(0, 0);

    trainer.evaluate(mlp, data);

    // Weights must be unchanged
    EXPECT_NEAR(mlp.layer(0).W()(0, 0), w00_before, 1e-15);
}

TEST(TrainerEvaluate, LossDecreasesAfterTraining) {
    MLP<double, Sigmoid<double>> mlp({2, 8, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> data = make_binary_dataset();

    const double loss_before = trainer.evaluate(mlp, data);
    trainer.train(mlp, data, 200, 0.1);
    const double loss_after = trainer.evaluate(mlp, data);

    EXPECT_LT(loss_after, loss_before);
}

// =========================================================================
// Trainer — argmax
// =========================================================================

// Alias avoids confusing the EXPECT_* macros with the comma in template args
using TrainerSig = Trainer<double, Sigmoid<double>>;

TEST(TrainerArgmax, ReturnsIndexOfMax) {
    Vector<double> v{0.1, 0.7, 0.2};
    EXPECT_EQ(TrainerSig::argmax(v), 1u);
}

TEST(TrainerArgmax, FirstElement) {
    Vector<double> v{0.9, 0.05, 0.05};
    EXPECT_EQ(TrainerSig::argmax(v), 0u);
}

TEST(TrainerArgmax, LastElement) {
    Vector<double> v{0.1, 0.1, 0.8};
    EXPECT_EQ(TrainerSig::argmax(v), 2u);
}

TEST(TrainerArgmax, EmptyThrows) {
    EXPECT_THROW(
        { auto r = TrainerSig::argmax(Vector<double>{}); (void)r; },
        std::invalid_argument);
}

// =========================================================================
// Trainer — overfitting on small dataset
// =========================================================================

TEST(TrainerOverfit, ThreeSamplesIdentityDataset) {
    // A wide-enough MLP should perfectly memorise 3 samples.
    MLP<double, Sigmoid<double>> mlp({3, 16, 3}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> data = make_identity_dataset();

    trainer.train(mlp, data, 2000, 0.1);

    const double final_acc = trainer.epoch_accuracies().back();
    EXPECT_NEAR(final_acc, 1.0, 1e-9)
        << "  Expected 100% accuracy on identity dataset, got: " << final_acc;
}

TEST(TrainerOverfit, LossBelowThresholdAfterOverfitting) {
    MLP<double, Sigmoid<double>> mlp({3, 16, 3}, 42);
    Trainer<double, Sigmoid<double>> trainer;
    Dataset<double> data = make_identity_dataset();

    trainer.train(mlp, data, 2000, 0.1);

    const double final_loss = trainer.epoch_losses().back();
    EXPECT_LT(final_loss, 0.1)
        << "  Expected loss < 0.1, got: " << final_loss;
}

// =========================================================================
// Trainer — reproducibility
// =========================================================================

TEST(TrainerReproducibility, SameSeedSameHistory) {
    Dataset<double> data = make_binary_dataset();

    MLP<double, Sigmoid<double>> mlp1({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer1;
    trainer1.train(mlp1, data, 10, 0.05);

    MLP<double, Sigmoid<double>> mlp2({2, 4, 2}, 42);
    Trainer<double, Sigmoid<double>> trainer2;
    trainer2.train(mlp2, data, 10, 0.05);

    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(trainer1.epoch_losses()[i],
                    trainer2.epoch_losses()[i], 1e-12)
            << "  at epoch " << i;
        EXPECT_NEAR(trainer1.epoch_accuracies()[i],
                    trainer2.epoch_accuracies()[i], 1e-12)
            << "  at epoch " << i;
    }
}

TEST(TrainerReproducibility, DifferentSeedsDifferentHistory) {
    Dataset<double> data = make_binary_dataset();

    MLP<double, Sigmoid<double>> mlp1({2, 4, 2}, 1);
    MLP<double, Sigmoid<double>> mlp2({2, 4, 2}, 99);
    Trainer<double, Sigmoid<double>> t;

    t.train(mlp1, data, 5, 0.05);
    const double loss1 = t.epoch_losses()[0];

    t.train(mlp2, data, 5, 0.05);
    const double loss2 = t.epoch_losses()[0];

    // Different initialisation → different first-epoch loss
    EXPECT_NE(loss1, loss2);
}

// =========================================================================
// Trainer — template instantiation with ReLU
// =========================================================================

TEST(TrainerTemplate, WorksWithReLU) {
    MLP<double, ReLU<double>> mlp({2, 8, 2}, 42);
    Trainer<double, ReLU<double>> trainer;
    Dataset<double> data = make_binary_dataset();

    trainer.train(mlp, data, 50, 0.01);

    for (double l : trainer.epoch_losses()) {
        EXPECT_TRUE(std::isfinite(l));
        EXPECT_GE(l, 0.0);
    }
}
