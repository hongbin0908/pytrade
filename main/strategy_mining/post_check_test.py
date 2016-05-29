#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@author 
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import unittest

from post_check import *



class post_check_test(unittest.TestCase):
    def SetUp(self):
        pass
    def test_readinput(self):
        readinput("testdata/post_check/1/", "/home/work/workplace/stock_data/")
def main():
    unittest.main()
if __name__ == '__main__':
    main()
