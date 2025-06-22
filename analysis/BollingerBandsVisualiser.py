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
    rows, cols = 50, 1  # for 50 instruments
    fig, axes = plt.subplots(rows, cols, figsize=(8, 150))
    fig.suptitle(f"Bollinger Bands for All Instruments (windowSize = {windowSize})", fontsize=20)

    for i in range(50):
        r, c = divmod(i, cols)

        if cols == 1:
            ax = axes[r]
        else:
            ax = axes[r][c]

        instrument = bollBandsAndPrices[i]
        days = np.arange(instrument.shape[0])

        ax.plot(days, instrument[:, 0], color='green', linewidth=1, label="Lower")
        ax.plot(days, instrument[:, 1], color='red', linewidth=1, label="Upper")
        ax.plot(days, instrument[:, 2], color='pink', linewidth=1, label="SMA")
        ax.plot(days, instrument[:, 3], color='blue', linewidth=1.5, label="Price")

        ax.set_title(f"Instr {i}", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(rect=(0, 0, 0.95, 0.97))  # Leave space for title + legend
    plt.savefig(f"./analysis/AllInstrumentsBollBandsPrices_WindowSize={windowSize}.png", dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    main()