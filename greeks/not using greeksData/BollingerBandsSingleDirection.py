import numpy as np

from greeks.GreekGeneratingClasses.BollingerBandsCalculator import BollingerBandsCalculator
from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class BollingerBandsSingleDirection(Greek):
    def __init__(self, prices, bbc: BollingerBandsCalculator, focusBand: str, comparator):
        super().__init__()
        self.recentDayPrices = prices[:, -1]
        self.bbc = bbc
        self.focusBand = focusBand
        self.comparator = comparator

        self.signal = np.full(bbc.pricesSoFar.shape[0], np.nan)

        if bbc.getSma() is not None:
            self.calculateSignal()

    def update(self, newDayPrices: np.ndarray):
        self.bbc.update(newDayPrices)

        if self.bbc.getSma() is not None:
            self.calculateSignal()

    def getGreeks(self):
        return self.signal

    def calculateSignal(self):
        band = None
        if self.focusBand == "upper":
            band = self.bbc.getUpperBands()
        else:
            band = self.bbc.getLowerBands()

        self.signal = self.comparator(self.recentDayPrices, band)