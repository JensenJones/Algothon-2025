from typing import List
import numpy as np

from greeks.GreekGeneratingClasses.GreekBaseClass import Greek


class GreeksManager(Greek):
    def __init__(self, greeks: List[Greek]):
        self.greeks = greeks
        self.greeksCount = len(greeks)

    def update(self, newDayPrices: np.ndarray):
        for greek in self.greeks:
            greek.update(newDayPrices)

    def getGreeksList(self) -> List[Greek]:
        return self.greeks

    def getGreeks(self):
        greeksList = self.greeks
        greeksData = []

        for i, greek in enumerate(greeksList):
            greeksData.append(greek.getGreeks())

        return np.array(greeksData)