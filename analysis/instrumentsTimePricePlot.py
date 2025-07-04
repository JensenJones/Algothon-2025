import numpy as np
import matplotlib.pyplot as plt

def main():
    prices = np.loadtxt("./sourceCode/1000Prices.txt")
    prices = prices.T # transpose so we have shape 50 (instrument), 750(price at day)

    for instrument in prices:
        plt.plot(instrument, alpha=0.7)

    plt.title("Price History of all 50 Instruments")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig("./analysis/AllInstrumentsVisualised_1000Days", dpi = 200)


if __name__ == "__main__":
    main()