import os
import time
import numpy as np
import optuna
from numpy.f2py.auxfuncs import throw_error
from sklearn.model_selection import TimeSeriesSplit
from MLBacktester import SimpleBacktester

# === CONFIGURATION ===
N_INSTRUMENTS           = 50
N_SPLITS                = 3
PENALTY_LAMBDA          = 0.5
N_TRIALS_PER_INSTRUMENT = 40

TEST_WINDOW_SIZE        = 125
TRAIN_WINDOW_SIZE       = 125

TRAIN_END_DAY           = 500
CV_END_DAY              = 625

CROSS_VALIDATION_THRESHOLD = 1

def main():
    prices_matrix = np.loadtxt("../../sourceCode/prices.txt").T  # shape (50, 750)

    trainingPrices = prices_matrix[:, :TRAIN_END_DAY]
    cvPrices = prices_matrix[:, TRAIN_END_DAY:CV_END_DAY]

    trainingBacktester = SimpleBacktester(trainingPrices, commission_enabled=True)
    cvBacktester = SimpleBacktester(cvPrices, commission_enabled=False)

    best_params = np.zeros((N_INSTRUMENTS, 4))
    passedCrossValidation = set()
    start_time = time.perf_counter()

    for inst in range(N_INSTRUMENTS):
        instrument_prices = trainingPrices[inst]

        tscv = TimeSeriesSplit(n_splits=N_SPLITS,
                               # test_size = TEST_WINDOW_SIZE,
                               # max_train_size=TRAIN_WINDOW_SIZE
                               )

        for fold, (train_idx, test_idx) in enumerate(tscv.split(instrument_prices)):
            tr_days = (train_idx[0] + 1, train_idx[-1] + 1)
            te_days = (test_idx[0] + 1, test_idx[-1] + 1)
            print(
                f"Fold {fold}: TRAIN days {tr_days[0]}–{tr_days[1]} "
                f"({len(train_idx)} days),"
            )

        def objective(trial):
            rsi_window = 14
            bb_window = 20
            purchase_alpha = trial.suggest_int("purchaseAlpha", 0, 200)
            rsi_low = 30
            rsi_high = 70
            params         = np.array([rsi_window, bb_window, purchase_alpha, rsi_low, rsi_high])
            regularisation_lamda = trial.suggest_loguniform("alpha_reg", 1e-6, 1e-1)

            val_profits = []
            # 3) Walk-forward folds
            for train_idx, test_idx in tscv.split(instrument_prices):
                # Convert 0-based idx to your backtester’s 1-based days
                tr_start, tr_end = train_idx[0] + 1, train_idx[-1] + 1
                # te_start, te_end = test_idx[0]  + 1, test_idx[-1]  + 1

                try:
                    results = trainingBacktester.run(params, inst)
                    val_profits.append(results["final_value"])
                except ValueError:
                    # too few days → heavy penalty
                    throw_error("Shouldn't get here")
                    return -1e9

            mean_val = np.mean(val_profits)

            return mean_val

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            storage="sqlite:///optuna.db",
            load_if_exists=True
        )
        study.optimize(objective, n_trials=N_TRIALS_PER_INSTRUMENT, n_jobs=os.cpu_count())
        p = study.best_params
        # p["window"] = 14
        p["rsi_low"] = 30
        p["rsi_high"] = 70
        print(f"Inst {inst} best score: {study.best_value:.2f}, params: {p}")
        best_params[inst] = [p["rsi_window"], p["bb_window], p["purchaseAlpha"], p["rsi_low"], p["rsi_high"]]

        # Only use the instruments that are profitable on unseen data
        if cvBacktester.run(best_params[inst], inst)["mean_pl"] >= CROSS_VALIDATION_THRESHOLD:
            passedCrossValidation.add(inst)

    np.save("best_params_per_instrument.npy", best_params)
    passedTestValidationNpArray = np.array(list(passedCrossValidation))
    np.save("passedTestValidation.npy", passedTestValidationNpArray)

    duration = time.perf_counter() - start_time
    print(f"All done in {duration:.1f}s")
    print(f"Best params:\n{best_params}")
    print(f"Passed cross validation:\n{passedTestValidationNpArray}")
    print(f"Cross validation from day={TRAIN_END_DAY} -> {CV_END_DAY}")

if __name__ == "__main__":
    main()
