// tests/test_validation.cpp
// Phase 10 — Validation Split + Early Stopping tests.
//
// Tests:
//   SplitSizes.*         — correct train/val sizes after split
//   SplitNoLeakage.*     — no sample appears in both subsets
//   SplitReproducible.*  — same seed produces same split
//   SplitEdge.*          — bad ratio throws
//   EarlyStopping.*      — training halts before max epochs
//   BestModel.*          — restored weights match best val_loss epoch
//   HistoryConsistency.* — history vectors are aligned
//   ValidationIsolation.*— validation set is not used during training

#include "data/dataset.hpp"
#include "mlp/trainer.hpp"
#include "mlp/mlp.hpp"
#include "activations/sigmoid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

// -----------------------------------------------------------------------
// Type aliases
// -----------------------------------------------------------------------
using F   = float;
using Act = mlp::Sigmoid<F>;
using TestMLP     = mlp::MLP<F, Act>;
using TestTrainer = mlp::Trainer<F, Act>;
using TestDataset = mlp::Dataset<F>;

// -----------------------------------------------------------------------
// Helper: build a dataset of 'n' samples with 2 features and 2 classes.
// Each sample's first feature encodes its index as a unique float so we
// can detect leakage across splits.
// -----------------------------------------------------------------------
static TestDataset make_dataset(std::size_t n) {
    TestDataset ds;
    ds.inputs.reserve(n);
    ds.labels.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        mlp::Vector<F> x(2);
        x[0] = static_cast<F>(i);                              // unique ID
        x[1] = static_cast<F>(n - 1 - i) / static_cast<F>(n);
        ds.inputs.push_back(x);

        mlp::Vector<F> y(2, F{0});
        y[i % 2] = F{1};
        ds.labels.push_back(y);
    }
    return ds;
}

// -----------------------------------------------------------------------
// SplitSizes — correct number of samples in each subset
// -----------------------------------------------------------------------
TEST(SplitSizes, EightyTwenty) {
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    EXPECT_EQ(train.size(), 8u);
    EXPECT_EQ(val.size(),   2u);
    EXPECT_EQ(train.size() + val.size(), ds.size());
}

TEST(SplitSizes, SeventyThirty) {
    const auto ds = make_dataset(20);
    auto [train, val] = mlp::train_validation_split(ds, F{0.7}, 0u);

    EXPECT_EQ(train.size(), 14u);
    EXPECT_EQ(val.size(),    6u);
    EXPECT_EQ(train.size() + val.size(), ds.size());
}

TEST(SplitSizes, NoSamplesLost) {
    const auto ds = make_dataset(13);   // odd / non-round
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 7u);
    EXPECT_EQ(train.size() + val.size(), 13u);
}

// -----------------------------------------------------------------------
// SplitNoLeakage — no sample appears in both train and val
// -----------------------------------------------------------------------
TEST(SplitNoLeakage, NoCommonSamples) {
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    // Each sample has a unique x[0] value (its index).
    // No x[0] from val should appear in train.
    for (const auto& vs : val.inputs) {
        for (const auto& ts : train.inputs) {
            EXPECT_NE(vs[0], ts[0])
                << "Leakage: sample with id=" << vs[0]
                << " appears in both sets";
        }
    }
}

// -----------------------------------------------------------------------
// SplitReproducible — same seed always produces same split
// -----------------------------------------------------------------------
TEST(SplitReproducible, SameSeedSameResult) {
    const auto ds = make_dataset(10);
    auto [train_a, val_a] = mlp::train_validation_split(ds, F{0.8}, 99u);
    auto [train_b, val_b] = mlp::train_validation_split(ds, F{0.8}, 99u);

    ASSERT_EQ(train_a.size(), train_b.size());
    ASSERT_EQ(val_a.size(),   val_b.size());

    for (std::size_t i = 0; i < train_a.size(); ++i) {
        EXPECT_EQ(train_a.inputs[i][0], train_b.inputs[i][0]);
    }
    for (std::size_t i = 0; i < val_a.size(); ++i) {
        EXPECT_EQ(val_a.inputs[i][0], val_b.inputs[i][0]);
    }
}

TEST(SplitReproducible, DifferentSeedsDifferentOrder) {
    const auto ds = make_dataset(10);
    auto [train_a, val_a] = mlp::train_validation_split(ds, F{0.8}, 1u);
    auto [train_b, val_b] = mlp::train_validation_split(ds, F{0.8}, 2u);

    // With two different seeds the ordering should differ at least once.
    bool any_diff = false;
    for (std::size_t i = 0; i < train_a.size() && !any_diff; ++i) {
        any_diff = (train_a.inputs[i][0] != train_b.inputs[i][0]);
    }
    EXPECT_TRUE(any_diff);
}

// -----------------------------------------------------------------------
// SplitEdge — bad ratio throws
// -----------------------------------------------------------------------
TEST(SplitEdge, RatioZeroThrows) {
    const auto ds = make_dataset(10);
    EXPECT_THROW(mlp::train_validation_split(ds, F{0}, 42u),
                 std::invalid_argument);
}

TEST(SplitEdge, RatioOneThrows) {
    const auto ds = make_dataset(10);
    EXPECT_THROW(mlp::train_validation_split(ds, F{1}, 42u),
                 std::invalid_argument);
}

TEST(SplitEdge, RatioNegativeThrows) {
    const auto ds = make_dataset(10);
    EXPECT_THROW(mlp::train_validation_split(ds, F{-0.5f}, 42u),
                 std::invalid_argument);
}

// -----------------------------------------------------------------------
// EarlyStopping — training halts before max epochs
//
// Strategy: use a very large min_delta (10.0f) so val_loss never
// improves by that margin after epoch 0.  With patience=2 the loop
// should stop after exactly 3 epochs (epoch 0 saves best, epochs 1 and
// 2 count as non-improvement → patience counter reaches 2 → break).
// -----------------------------------------------------------------------
TEST(EarlyStopping, StopsBeforeMaxEpochs) {
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    TestMLP     mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 2;
    es.min_delta = F{10.0};   // unreachable threshold → fires after patience

    trainer.train_with_validation(mlp, train, val,
                                  /*epochs=*/100,
                                  /*lr=*/F{0.01},
                                  /*batch_size=*/1,
                                  /*shuffle=*/false,
                                  /*seed=*/42u,
                                  es);

    EXPECT_LT(trainer.val_loss_history().size(), 100u)
        << "Early stopping should have halted training before 100 epochs";
}

TEST(EarlyStopping, PatientWaitsCorrectly) {
    // With patience=3 and min_delta=10, we expect exactly 4 epochs:
    // epoch 0 → improvement (saves best)
    // epochs 1,2,3 → no improvement → counter=3 → break
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    TestMLP     mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 3;
    es.min_delta = F{10.0};

    trainer.train_with_validation(mlp, train, val,
                                  /*epochs=*/100,
                                  F{0.01}, 1, false, 42u, es);

    // Expect patience+1 = 4 epochs total
    EXPECT_EQ(trainer.val_loss_history().size(), 4u);
}

// -----------------------------------------------------------------------
// BestModel — weights restored to the best val_loss checkpoint
// -----------------------------------------------------------------------
TEST(BestModel, RestoredWeightsMatchBestValLoss) {
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    TestMLP     mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    // Use min_delta=0 so every val_loss improvement triggers a snapshot save.
    // The final snapshot therefore corresponds to the epoch with the global
    // minimum val_loss in the history.  patience=100 keeps early stopping
    // from firing within these 5 epochs, so all epochs complete.
    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 100;
    es.min_delta = F{0};

    trainer.train_with_validation(mlp, train, val,
                                  /*epochs=*/5,
                                  F{0.01}, 1, false, 42u, es);

    // After training the MLP holds the best-epoch weights.
    // Re-evaluate and compare with the recorded minimum val_loss.
    const F val_loss_after = trainer.evaluate(mlp, val);

    const auto& hist = trainer.val_loss_history();
    const F best_val = *std::min_element(hist.begin(), hist.end());

    EXPECT_NEAR(val_loss_after, best_val, F{1e-5})
        << "Restored model val_loss should equal best recorded val_loss";
}

TEST(BestModel, RestoredIsBetterThanLast) {
    // Train without early stopping (enough epochs for divergence) and
    // verify the snapshot mechanism restores something at least as good
    // as the very last epoch.
    const auto ds = make_dataset(12);
    auto [train, val] = mlp::train_validation_split(ds, F{0.75}, 42u);

    TestMLP     mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    // Use a moderate min_delta so we do save some snapshot.
    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 5;
    es.min_delta = F{0.0};

    trainer.train_with_validation(mlp, train, val,
                                  /*epochs=*/30,
                                  F{0.05}, 1, false, 42u, es);

    const F val_loss_after  = trainer.evaluate(mlp, val);
    const auto& hist        = trainer.val_loss_history();
    const F best_val        = *std::min_element(hist.begin(), hist.end());

    // Restored model must be at most as bad as the best epoch.
    EXPECT_LE(val_loss_after, best_val + F{1e-4});
}

// -----------------------------------------------------------------------
// HistoryConsistency — all three history vectors have the same length
// and match the number of epochs actually run.
// -----------------------------------------------------------------------
TEST(HistoryConsistency, SizesMatch) {
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    TestMLP     mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 2;
    es.min_delta = F{10.0};

    trainer.train_with_validation(mlp, train, val,
                                  50, F{0.01}, 1, false, 42u, es);

    const std::size_t n = trainer.val_loss_history().size();
    EXPECT_EQ(trainer.train_loss_history().size(),    n);
    EXPECT_EQ(trainer.val_macro_f1_history().size(),  n);
    EXPECT_GT(n, 0u);
}

TEST(HistoryConsistency, MacroF1InRange) {
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    TestMLP     mlp({2, 4, 2}, 42u);
    TestTrainer trainer;

    trainer.train_with_validation(mlp, train, val,
                                  5, F{0.01});

    for (F f1 : trainer.val_macro_f1_history()) {
        EXPECT_GE(f1, F{0.0});
        EXPECT_LE(f1, F{1.0});
    }
}

// -----------------------------------------------------------------------
// ValidationIsolation — the validation set is evaluated but never
// used to update weights.
//
// Approach: train two identical MLPs on the same train set.  One uses
// train_with_validation (has a val set), the other uses plain train().
// Both should produce the same train_loss trajectory because val data
// does not affect the gradient updates.
// -----------------------------------------------------------------------
TEST(ValidationIsolation, ValNotUsedForWeightUpdates) {
    const auto ds = make_dataset(10);
    auto [train, val] = mlp::train_validation_split(ds, F{0.8}, 42u);

    // Two MLPs with identical seeds.
    TestMLP mlp_a({2, 4, 2}, 42u);
    TestMLP mlp_b({2, 4, 2}, 42u);

    TestTrainer trainer_a;
    TestTrainer trainer_b;

    const std::size_t epochs = 5;

    // mlp_a: trained with validation monitoring
    mlp::EarlyStoppingConfig<F> es;
    es.patience  = 100;    // never fires within 5 epochs
    es.min_delta = F{0.0};
    trainer_a.train_with_validation(mlp_a, train, val,
                                    epochs, F{0.01}, 1, false, 42u, es);

    // mlp_b: trained with plain train() on the same train set
    trainer_b.train(mlp_b, train, epochs, F{0.01});

    // Train loss histories should be identical (val set had no influence).
    ASSERT_EQ(trainer_a.train_loss_history().size(), epochs);
    ASSERT_EQ(trainer_b.epoch_losses().size(),       epochs);

    for (std::size_t e = 0; e < epochs; ++e) {
        EXPECT_NEAR(trainer_a.train_loss_history()[e],
                    trainer_b.epoch_losses()[e],
                    F{1e-5})
            << "Epoch " << e << ": train loss differs between "
               "plain train() and train_with_validation()";
    }
}
