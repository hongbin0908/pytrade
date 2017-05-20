import sys,os
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main import base
from main.base import stock_fetcher as sf

#def test_is_test_flag():
#    assert base.is_test_flag()
#
#
#def test_split_by_year():
#    df = pd.read_csv(os.path.join(root, 'data', 'yeod', 'sp500_snapshot_20091231', 'IBM.csv'))
#    len_total = len(df)
#    len2 = 0
#    for each in base.split_by_year(df):
#        len2 += len(each)
#    assert len_total == len2
#
#def test_random_sort():
#    df = pd.read_csv(os.path.join(root, 'data', 'yeod', 'sp500_snapshot_20091231', 'IBM.csv'))
#    df = base.random_sort(df)
#
#def test_get_last_trade_date():
#    assert '2017-05-18' == base.get_last_trade_date(is_force=True)
#
#def test_yahoo_finance():
#    import yahoo_finance as yf
#    yahoo = yf.Share('YHOO')
#    print(yahoo.get_historical('2014-01-01', '2017-01-01'))
#
def test_dict():
    res = {'datatable': {'data': [
        ['YHOO', '1996-04-12', 25.25, 43.0, 24.5, 33.0, 17030000.0, 0.0, 1.0, 1.0520833333333333, 1.7916666666666665,
         1.0208333333333333, 1.375, 408720000.0],]}}
    df = pd.DataFrame(res['datatable']['data'])
    print(df)

def test_get_stock():
    #df = sf.get_stock('YHOO')
    #assert len(df) > 1000
    df = sf.get_stock('IBM')
    assert len(df) > 10000


if __name__ == '__main__':
    for i in range(1, 10000):
        print(i)
        test_get_stock()
