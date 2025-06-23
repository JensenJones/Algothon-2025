from typing import List
import numpy as np

from greeks import Greek


class GreeksManager(Greek):
    def __init__(self, greeks: List[Greek]):
        self.greeks = greeks
        self.greeksCount = len(greeks)

    def update(self, newDayPrices: np.ndarray):
        for greek in self.greeks:
            greek.update(newDayPrices)

    def getGreeks(self):
        greeks = np.array((self.greeksCount, 50))

        for i, greek in range(self.greeksCount), self.greeks:
            greeks[i] = greek.getGreeks()

        return greeks