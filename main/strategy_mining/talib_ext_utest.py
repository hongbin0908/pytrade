#!/usr/bin/env python
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import talib_ext as te
import unittest
import numpy as np
class TestTalibExt(unittest.TestCase):

    def setUp(self):
        pass

    def test_ADX_ext1(self):
        high = np.random.randn(60) * 1.12

        low = np.random.randn(60)* 0.9
        close = np.random.randn(60)*1.0
            
        #self.assertEqual(self.seq, range(10))
    def test_ADX_ext201(self):
        high = np.random.randn(60) * 1.5
        low = np.random.randn(60)* 0.7
        close = np.random.randn(60)*1.0
        te.ADX_ext201(high, low, close)
    def test_MACD_ext101(self):
        close = np.random.randn(60)
        print te.MACD_ext101(close, 1)
if __name__ == '__main__':
    unittest.main()
