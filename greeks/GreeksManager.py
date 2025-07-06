import numpy as np
import pandas as pd

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek

class GreeksManager:
    def __init__(self, greeks: dict[str, Greek]):
        self.greeks = greeks
        self.greeksCount = len(greeks)

    def updateGreeks(self, newDayPrices: np.ndarray):
        for greek in self.greeks.values():
            greek.update(newDayPrices)

    def getGreeksList(self):
        return self.greeks.values()

    def getGreeksDict(self):
        greekHistoryArray = [
            np.swapaxes(greek.getGreeksHistory(), 0, 1)[-1:, :, np.newaxis]
            for greek in self.greeks.values()
        ]
        greekArrayNp = np.concatenate(greekHistoryArray, axis=-1)  # shape (1, nInst, num_greeks)
        featureNames = list(self.greeks.keys())

        exogDict = {
            f"inst_{i}": pd.DataFrame(greekArrayNp[:, i, :], columns=featureNames)
            for i in range(greekArrayNp.shape[1])
        }

        return exogDict

    def getGreeksHistoryDict(self) -> dict[str, pd.DataFrame]:
        greekHistoryArray = [
            np.swapaxes(greek.getGreeksHistory(), 0, 1)[:, :, np.newaxis]
            for greek in self.greeks.values()
        ]

        greekArrayNp = np.concatenate(greekHistoryArray, axis=-1)  # shape (days, nInst, num_greeks)
        featureNames = list(self.greeks.keys())

        exogDict = {
            f"inst_{i}": pd.DataFrame(greekArrayNp[:, i, :], columns=featureNames)
            for i in range(greekArrayNp.shape[1])
        }


        return exogDict