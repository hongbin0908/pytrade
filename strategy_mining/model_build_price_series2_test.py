
#!/usr/bin/env python
#@author 
import sys,os
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(local_path + "/./")

import model_build_price_series2 as model
from model_build_price_series2 import *
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
        self.assertEqual(local_path + '/data/prices_series/', options.output)
        self.assertEqual('Extractor1', options.extractor)
    # }}}

    def test_extract_features_and_classes_1(self): # {{{
        close_prices = [1,2,4,2]
        extractor = Extractor1("abc",None, close_prices, close_prices, close_prices, close_prices, 2)
        self.assertEqual("2.0,2.0,0\n",
                extractor.extract_features_and_classes() )
    # }}}

    def test_extract_features_and_classes_2(self): # {{{
        close_prices = [1,2,4,2,4]
        extractor = Extractor1("abc",None, None, None, None, close_prices, 2)
        self.assertEqual("2.0,2.0,0\n" + 
                         "2.0,0.5,1\n",
                extractor.extract_features_and_classes() )
    # }}}

    def test_extract_features_and_classes_3(self): # {{{
        close_prices = [1,2,4,2,1]
        extractor = Extractor1("abc",None, None, None, None, close_prices, 2)

        self.assertEqual("2.0,2.0,0\n" + 
                         "2.0,0.5,0\n",
                extractor.extract_features_and_classes() )
    # }}}

    def test_extract_features_and_classes_4(self): # {{{
        close_prices = [1,2,4,2,1]
        extractor = Extractor1("abc",None, None, None, None, close_prices, 2)
        self.assertEqual("2.0,2.0,0\n" + 
                         "2.0,0.5,0\n",
                extractor.extract_features_and_classes() )
    # }}}

    def test_extract_features_and_classes_5(self): # {{{
        close_prices = [1,2,4,2,1,2,4]
        extractor = Extractor1("abc",None, None, None, None, close_prices, 4)
        self.assertEqual("2.0,2.0,0.5,0.5,1\n" + 
                         "2.0,0.5,0.5,2.0,1\n",
                extractor.extract_features_and_classes() )
    # }}}

    def test_extract_features_and_classes_6(self): # {{{
        close_prices = [1,2,4,2,1,2,4]
        dates = [1,2,3,4,5,6,7]
        extractor = Extractor1("SYM",dates, None, None, None, close_prices, 4)
        self.assertEqual("SYM,7,0.5,0.5,2.0,2.0\n",
                extractor.extract_last_features() )
    # }}}

    def test_extract_features_and_classes_7(self): # {{{
        close_prices = [1,2,4,2,1,2,4]
        dates = [1,2,3,4,5,6,7]
        extractor = Extractor1("SYM",dates, None, None, None, close_prices, 4)
        self.assertEqual("SYM,7,0.5,0.5,2.0,2.0\n",
                extractor.extract_last_features() )
    # }}}

    def test_extract_2(self):
        close_prices = [1,2,4,2,1,2,4]
        dates = [1,2,3,4,5,6,7]
        extractor = Extractor2("SYM", dates, None, None, None, close_prices, 4)
        self.assertEqual("2.0,4.0,2.0,1.0,1\n" + 
                         "2.0,1.0,0.5,1.0,1\n",
                         extractor.extract_features_and_classes() )
        self.assertEqual("SYM,7,0.5,0.25,0.5,1.0\n",
                extractor.extract_last_features() )
    def test_extract_2(self):
        open_prices =  [1,2,4,2,1,2,4]
        high_prices =  [1,2,4,2,1,2,4]
        low_prices =   [1,2,4,2,1,2,4]
        close_prices = [1,2,4,2,1,2,4]
        dates = [1,2,3,4,5,6,7]
        extractor = Extractor3("SYM", dates, open_prices, high_prices, low_prices, close_prices, 4)
        self.assertEqual("2.0,2.0,2.0,2.0,4.0,4.0,4.0,4.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,1\n" + 
                         "2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1\n",
                         extractor.extract_features_and_classes() )
        self.assertEqual("SYM,7,0.5,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0\n",
                extractor.extract_last_features() )

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

