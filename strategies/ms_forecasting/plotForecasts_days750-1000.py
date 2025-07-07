import numpy as np
import pandas as pd
from skforecast.plot import set_dark_theme

import matplotlib.pyplot as plt

set_dark_theme()

def main():
    predictedLogReturns = np.load("./strategies/ms_forecasting/predicted_log_returns_days_750-1000.npy").T # (50, 249)
    print(f"Shape of predictions = {predictedLogReturns.shape}")

    prices = np.loadtxt("./sourceCode/1000Prices.txt").T[:, -250:] # (50, 250)

    actualLogReturns = np.log( prices[:, 1:] / prices[:, :-1] ) # (50, 249)
    print(f"Shape of actual logReturns = {actualLogReturns.shape}")

    instrument = 0

    plt.figure(figsize=(12, 5))
    plt.plot(predictedLogReturns[instrument, :], label="Predicted LogReturn", color='dodgerblue')
    plt.plot(actualLogReturns[instrument, :], label="Actual LogReturn", color='orange')
    plt.title(f"Predicted vs Actual LogReturns for {instrument}")
    plt.xlabel("Time")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    predictionAccuracyPrint(actualLogReturns, predictedLogReturns)


def predictionAccuracyPrint(actualLogReturns, predictedLogReturns):
    correctPosChange = 0
    correctNegChange = 0
    falsePosChange = 0
    falseNegChange = 0
    correctZeroChange = 0
    falseZeroChange = 0
    for inst in range(50):
        for day in range(249):
            if predictedLogReturns[inst, day] < 0:
                if actualLogReturns[inst, day] < 0:
                    correctNegChange += 1
                else:
                    falseNegChange += 1
            elif predictedLogReturns[inst, day] > 0:
                if actualLogReturns[inst, day] > 0:
                    correctPosChange += 1
                else:
                    falsePosChange += 1
            else:
                if actualLogReturns[inst, day] == 0:
                    correctZeroChange += 1
                else:
                    falseZeroChange += 1
    print(f"Correct Positive Change prediction count = {correctPosChange}")
    print(f"Correct Zero Change prediction count     = {correctZeroChange}")
    print(f"Correct Negative Change prediction count = {correctNegChange}")
    print(f"False Positive Change prediction count   = {falsePosChange}")
    print(f"False Zero Change prediction count       = {falseZeroChange}")
    print(f"False Negative Change prediction count   = {falseNegChange}")
    correct_pct = ((correctZeroChange + correctPosChange + correctNegChange) /
                   (249 * 50)) * 100
    pink_bold = "\033[1;35m"
    reset = "\033[0m"
    print(f"{pink_bold}Correct percentage = {correct_pct:.2f}%{reset}")


if __name__ == '__main__':
    main()