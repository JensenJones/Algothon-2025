import os
import time
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from MLBacktester import SimpleBacktester

# === CONFIGURATION ===
N_INSTRUMENTS           = 50
N_SPLITS                = 5
TEST_WINDOW_SIZE        = 100
PENALTY_LAMBDA          = 0.5
N_TRIALS_PER_INSTRUMENT = 40

def main():
    # 1) Load once
    prices_matrix = np.loadtxt("../../sourceCode/prices.txt").T  # shape (50, 750)
    bt = SimpleBacktester(prices_matrix, commission_enabled=True)

    best_params = np.zeros((N_INSTRUMENTS, 4))
    start_time = time.perf_counter()

    for inst in range(N_INSTRUMENTS):
        instrument_prices = prices_matrix[inst]  # shape: (T,)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_WINDOW_SIZE)

        def objective(trial):
            # 2) Sample hyperparameters
            window         = trial.suggest_int("window", 5, 20)
            purchase_alpha = trial.suggest_int("purchaseAlpha", 5, 300)
            rsi_low        = trial.suggest_int("rsi_low", 5, 45)
            rsi_high       = trial.suggest_int("rsi_high", 55, 95)
            params         = np.array([window, purchase_alpha, rsi_low, rsi_high])

            val_profits = []
            # 3) Walk-forward folds
            for train_idx, test_idx in tscv.split(instrument_prices):
                # Convert 0-based idx to your backtester’s 1-based days
                tr_start, tr_end = train_idx[0] + 1, train_idx[-1] + 1
                te_start, te_end = test_idx[0]  + 1, test_idx[-1]  + 1

                try:
                    # train-warmup: build indicators
                    _ = bt.run(tr_start, tr_end, params, inst)
                    # test out-of-sample
                    results = bt.run(te_start, te_end, params, inst)
                    val_profits.append(results["final_value"])
                except ValueError:
                    # too few days → heavy penalty
                    return -1e9

            mean_val = np.mean(val_profits)
            std_val  = np.std(val_profits)
            # 4) risk-adjusted objective
            return mean_val - PENALTY_LAMBDA * std_val

        # 5) Optuna study with pruning & parallelism
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            storage="sqlite:///optuna.db",
            load_if_exists=True
        )
        study.optimize(objective, n_trials=N_TRIALS_PER_INSTRUMENT, n_jobs=os.cpu_count())

        print(f"Inst {inst} best score: {study.best_value:.2f}, params: {study.best_params}")
        p = study.best_params
        best_params[inst] = [p["window"], p["purchaseAlpha"], p["rsi_low"], p["rsi_high"]]

    # 6) Save
    np.save("best_params_per_instrument.npy", best_params)
    duration = time.perf_counter() - start_time
    print(f"All done in {duration:.1f}s")

if __name__ == "__main__":
    main()
