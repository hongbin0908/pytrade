#!/usr/bin/env python
#@author 
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import model_base as base
import unittest
class ModelBaseTest (unittest.TestCase):
    def SetUp(self):
        pass
    def test_get_stock_data(self):
        dates, opens, highs, lows, close, adjusts, volumes = base.get_stock_data(local_path + "/testdata/A.csv")
        self.assertEqual(9, len(dates))
        dates, opens, highs, lows, close, adjusts, volumes = base.get_stock_data(local_path + "/testdata/A.csv", "2014-11-14")
        self.assertEqual(8, len(dates))
        self.assertEqual("2014-11-13", dates[-1])
    def test_get_date_str(self):
        print base.get_date_str()
def main():
    unittest.main()
if __name__ == '__main__':
    main()
        
