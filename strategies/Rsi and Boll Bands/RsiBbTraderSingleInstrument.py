import numpy as np

class RsiBollingerBandsTrader:
    def __init__(self, logFilePath, prices, bollingerBandsCalculator, rsiCalculator, purchaseAlpha: int, rsiLongThreshold=30, rsiShortThreshold=70):
        super().__init__()

        # Always ensure prices is shape (1, n_days)
        self.prices = prices if prices.ndim == 2 else prices[np.newaxis, :]
        self.logFilePath = logFilePath
        self.bbc = bollingerBandsCalculator
        self.rsiC = rsiCalculator
        self.purchaseAlpha = purchaseAlpha
        self.rsiLongThreshold = rsiLongThreshold
        self.rsiShortThreshold = rsiShortThreshold

        if logFilePath is not None:
            open(logFilePath, "wb").close()

        self.logAll()

    def updatePosition(self, newDayPrice, position):
        # newDayPrice: float or shape (1,)
        # position: float or shape (1,)

        newDayPrice = np.atleast_1d(newDayPrice)
        position = np.atleast_1d(position)
        self.prices = np.hstack([self.prices, newDayPrice.reshape(1, 1)])

        self.bbc.updateWithNewDay(newDayPrice)
        self.rsiC.updateWithNewDay(newDayPrice)

        if self.prices.shape[1] < max(self.bbc.windowSize, self.rsiC.windowSize):
            return position

        upperBand = self.bbc.getUpperBands()
        lowerBand = self.bbc.getLowerBands()
        rsi = self.rsiC.getRsi()

        if rsi is None:
            print("rsi is none in update position\n")
            return position

        # Simple, not mask: 1D instrument
        if newDayPrice[0] < lowerBand[0] and rsi[0] < self.rsiLongThreshold:
            position[0] += (self.purchaseAlpha * (self.rsiLongThreshold - rsi[0]) / self.rsiLongThreshold)
        if newDayPrice[0] > upperBand[0] and rsi[0] > self.rsiShortThreshold:
            position[0] -= (self.purchaseAlpha * (rsi[0] - self.rsiShortThreshold) / (100 - self.rsiShortThreshold))

        self.logAll()
        return position

    def logAll(self):
        if self.logFilePath is None:
            return

        with open(self.logFilePath, "ab") as logFile:
            lowerBand = self.bbc.getLowerBands()
            upperBand = self.bbc.getUpperBands()
            sma = self.bbc.getSma()
            rsi = self.rsiC.getRsi()
            dayPrice = self.prices[:, -1]  # shape (1,)

            if lowerBand is None or upperBand is None or sma is None or rsi is None or dayPrice is None:
                return

            # All are 1-element arrays
            log = np.hstack([
                lowerBand.reshape(-1, 1),
                upperBand.reshape(-1, 1),
                sma.reshape(-1, 1),
                rsi.reshape(-1, 1),
                dayPrice.reshape(-1, 1)
            ])  # shape (1, 5)

            np.save(logFile, log)
