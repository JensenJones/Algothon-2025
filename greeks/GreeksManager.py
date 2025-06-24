from typing import List
import numpy as np

from greeks.GreekBaseClass import Greek


class GreeksManager(Greek):
    def __init__(self, greeks: List[Greek]):
        self.greeks = greeks
        self.greeksCount = len(greeks)

    def update(self, newDayPrices: np.ndarray):
        for greek in self.greeks:
            greek.update(newDayPrices)

    def getGreeks(self) -> List[Greek]:
        return self.greeks