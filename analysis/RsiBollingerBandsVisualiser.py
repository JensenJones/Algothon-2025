import re
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    filePath = sys.argv[1]
    windowSize = [int(match.group()) for match in re.finditer(r'\d+', filePath)][0]

    bollBandsAndPrices = []
    with open(filePath, "rb") as logFile:
        while True:
            try:
                bollBandsAndPrices.append(np.load(logFile))
            except EOFError:
                break

    bollBandsAndPrices = np.stack(bollBandsAndPrices, axis = 1)

    saveGraphs(bollBandsAndPrices, windowSize)


def saveGraphs(bollBandsAndPrices, windowSize):
    fig, axes = plt.subplots(100, 1, figsize=(12, 200), gridspec_kw={'height_ratios': [3, 1]*50})
    fig.suptitle(f"Bollinger Bands + RSI (windowSize = {windowSize})", fontsize=20)

    for i in range(50):
        instrument = bollBandsAndPrices[i]
        days = np.arange(instrument.shape[0])

        ax_price = axes[2 * i]
        ax_rsi   = axes[2 * i + 1]

        # Price chart with bands
        ax_price.plot(days, instrument[:, 0], color='green', linewidth=1, label="Lower")
        ax_price.plot(days, instrument[:, 1], color='red', linewidth=1, label="Upper")
        ax_price.plot(days, instrument[:, 2], color='pink', linewidth=1, label="SMA")
        ax_price.plot(days, instrument[:, 4], color='blue', linewidth=1.5, label="Price")
        ax_price.set_title(f"Instrument {i}", fontsize=9)
        ax_price.grid(True)
        ax_price.tick_params(labelsize=6)

        # RSI histogram
        rsi = instrument[:, 3]
        colors = np.where(rsi > 70, 'red',
                 np.where(rsi < 30, 'green', 'lightgrey'))

        ax_rsi.bar(days, rsi, color=colors, width=1.0)
        ax_rsi.axhline(70, color='gray', linestyle='--', linewidth=0.5)
        ax_rsi.axhline(30, color='gray', linestyle='--', linewidth=0.5)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", fontsize=6)
        ax_rsi.grid(True)
        ax_rsi.tick_params(labelsize=6)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(f"./analysis/RSI_Histogram_BollBands_WindowSize={windowSize}.png", dpi=250, bbox_inches="tight")



if __name__ == "__main__":
    main()