import unittest

import numpy as np

from Greeks import Greeks

class MyTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.startingDays = 750
        prices = np.loadtxt("../sourceCode/prices.txt").T
        self.greeks = Greeks(prices[:, :self.startingDays])

    def test_should_initialise_correct(self):
        self.assertEqual(self.greeks.prices.shape, (50, self.startingDays))
        self.assertEqual(self.greeks.logReturns.shape, (50, self.startingDays - 1))




if __name__ == '__main__':
    unittest.main()
