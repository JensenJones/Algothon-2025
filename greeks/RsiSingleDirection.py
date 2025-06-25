import numpy as np

from greeks.GreekBaseClass import Greek
from greeks.RsiCalculator import RsiCalculator


class RsiSingleDirection(Greek):
    def __init__(self, rsic: RsiCalculator, direction: str, threshold: int):
        super().__init__()
        assert direction in {"long", "short"}, "Direction must be 'long' or 'short'\n"
        self.rsic = rsic
        self.direction = direction
        self.threshold = threshold
        self.signal = np.full(self.rsic.prices.shape[0], np.nan)

        rsi = rsic.getRsi()
        if rsi is not None:
            self.calculateSignal()

    def update(self, newDayPrices: np.ndarray):
        self.rsic.updateWithNewDay(newDayPrices)

        if self.rsic.getRsi() is not None:
            self.calculateSignal()

    def calculateSignal(self):
        rsi = self.rsic.getRsi()

        if self.direction == "long":
            self.signal = (rsi < self.threshold).astype(int)
        else:
            self.signal = (rsi > self.threshold).astype(int)

    def getGreeks(self):
        return self.signal