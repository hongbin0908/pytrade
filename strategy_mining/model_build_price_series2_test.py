
#!/usr/bin/env python
#@author 
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import model_build_price_series2 as model
import unittest
from optparse import OptionParser


class ModuleBuildPriceSeries2Test (unittest.TestCase):
    def setUp(self):
        pass
    def test_parse_options_1(self): #{{{
        """
        check the default options
        """
        parser = OptionParser()
        (options, args) = model.parse_options(parser)
        self.assertEqual(60, options.window)
        self.assertEqual('data/prices_series/', options.output)
    # }}}

    def test_extract_features_and_classes_1(self): # {{{
        close_prices = [1,2,4,2]
        self.assertEqual("2.0,2.0,0\n",
                model.extract_features_and_classes(close_prices, 2) )
    # }}}

    def test_extract_features_and_classes_2(self): # {{{
        close_prices = [1,2,4,2,4]
        self.assertEqual("2.0,2.0,0\n" + 
                         "2.0,0.5,1\n",
                model.extract_features_and_classes(close_prices, 2) )
    # }}}

    def test_extract_features_and_classes_3(self): # {{{
        close_prices = [1,2,4,2,1]
        self.assertEqual("2.0,2.0,0\n" + 
                         "2.0,0.5,0\n",
                model.extract_features_and_classes(close_prices, 2) )
    # }}}

    def test_extract_features_and_classes_4(self): # {{{
        close_prices = [1,2,4,2,1]
        self.assertEqual("2.0,2.0,0\n" + 
                         "2.0,0.5,0\n",
                model.extract_features_and_classes(close_prices, 2) )
    # }}}

    def test_extract_features_and_classes_5(self): # {{{
        close_prices = [1,2,4,2,1,2,4]
        self.assertEqual("2.0,2.0,0.5,0.5,1\n" + 
                         "2.0,0.5,0.5,2.0,1\n",
                model.extract_features_and_classes(close_prices, 4) )
    # }}}

    def test_extract_features_and_classes_5(self): # {{{
        close_prices = [1,2,4,2,1,2,4]
        dates = [1,2,3,4,5,6,7]
        self.assertEqual("SYM,7,0.5,0.5,2.0,2.0\n",
                model.extract_last_features("SYM", dates, close_prices, 4) )
    # }}}

    def test_extract_features_and_classes_5(self): # {{{
        close_prices = [1,2,4,2,1,2,4]
        dates = [1,2,3,4,5,6,7]
        self.assertEqual("SYM,7,0.5,0.5,2.0,2.0\n",
                model.extract_last_features("SYM", dates, close_prices, 4) )
    # }}}

    def test_main_1(self): # {{{
        parser = OptionParser()
        (options, args) = model.parse_options(parser)
        options.stocks_path = local_path + "/" + "testdata/stock_data"
        options.output = local_path + "/" + "testdata/data/"
        options.window = 2
        options.limit = 2
        model.main(options, args)
    # }}}


def main():
    unittest.main()
if __name__ == '__main__':
    main()

