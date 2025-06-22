import numpy as np
import pandas as pd
from BbCalculator import BollingerBandsCalculator
from RsiCalculator import RsiCalculator
from RsiBbTraderSingleInstrument import RsiBollingerBandsTrader as RsiBbTrader

INSTRUMENT_POSITION_LIMIT = 10000
COMMISSION_RATE = 0.0005
TRADING_DAYS_PER_YEAR = 249

class SimpleBacktester:
    def __init__(self, prices, commission_enabled: bool = True):
        self.commission_enabled = commission_enabled
        self.prices = prices

    def run(self, startDay: int, endDay: int, params, instIndex: int) -> dict:
        prices = self.prices[instIndex, :]  # shape: (num_days,)
        n_days = prices.shape[0]
        # Defensive: don't run past available prices
        endDay = min(endDay, n_days)
        # Make sure window fits
        window = int(params[0])

        if endDay <= startDay + window:
            raise ValueError("Not enough days for window and trading.")

        # Set up
        position = 0.0
        cash = 0.0
        value = 0.0
        dailyPL = []

        # Initialize indicators with window size
        pricesSoFar = prices[startDay : startDay + window].reshape(1, -1)
        bbC = BollingerBandsCalculator(pricesSoFar, window)
        rsiC = RsiCalculator(pricesSoFar, window)
        trader = RsiBbTrader(None, pricesSoFar, bbC, rsiC, params[1], params[2], params[3])

        for day in range(startDay + window, endDay):
            newPrice = prices[day]  # scalar
            prevPosition = position

            # No trading on last day
            if day < endDay - 1:
                new_position = trader.updatePosition(np.array([newPrice]), np.array([prevPosition]))
                # Get first element, since single instrument
                new_position = new_position[0] if isinstance(new_position, np.ndarray) else float(new_position)
                posLimit = int(INSTRUMENT_POSITION_LIMIT // newPrice)
                position = np.clip(new_position, -posLimit, posLimit)

                deltaPos = position - prevPosition
                dvolume = abs(deltaPos) * newPrice
                commission = COMMISSION_RATE * dvolume if self.commission_enabled else 0.0
                cash -= deltaPos * newPrice + commission
            else:
                position = prevPosition

            posValue = position * newPrice
            today_pl = cash + posValue - value
            value = cash + posValue
            dailyPL.append(today_pl)

        dailyPL = np.array(dailyPL[1:])  # skip first day to match comp convention
        pl_mean = np.mean(dailyPL)
        pl_std = np.std(dailyPL)
        ann_sharpe = (np.sqrt(TRADING_DAYS_PER_YEAR) * pl_mean / pl_std) if pl_std > 0 else 0.0

        return {
            "final_value": value,
            "mean_pl": pl_mean,
            "sharpe": ann_sharpe,
        }
