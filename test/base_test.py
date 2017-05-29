import os
import sys

import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main import base
from main.base import stock_fetcher as sf
from main.score import score
from main.model import ana

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

def test_get_last_trade_date_local():
    base.get_last_trade_date_local('sp500_snapshot_20091231')
def test_dict():
    res = {'datatable': {'data': [
        ['YHOO', '1996-04-12', 25.25, 43.0, 24.5, 33.0, 17030000.0, 0.0, 1.0, 1.0520833333333333, 1.7916666666666665,
         1.0208333333333333, 1.375, 408720000.0],]}}
    df = pd.DataFrame(res['datatable']['data'])
    print(df)
    assert True

def test_get_stock():
    #df = sf.get_stock('YHOO')
    #assert len(df) > 1000
    df = sf.get_stock('IBM')
    assert len(df) > 10000


def test_roi_topN():
    type = 'short'
    tscore = score.ScoreLabel(5, 1.0)
    score_name = tscore.get_name()
    df = pd.DataFrame(columns = ['pred', 'close', 'date', score_name])
    df.loc[0] = [0.2, 20, '20160520', 1.1]
    df.loc[1] = [0.3, 40, '20160520', 0.9]
    df.loc[2] = [0.2, 20, '20160521', 1.1]
    res = ana.roi_topN(df, tscore, 1, 'short')
    assert res['roi4'][0] == -100

    res = ana.roi_topN(df, tscore, 1, 'long')
    assert res['roi4'][0] == -100

def test_roi_level_per_year():
    tscore = score.ScoreLabel(5, 1.0)
    score_name = tscore.get_name()
    df = pd.DataFrame(columns = ['pred', 'close', 'date', score_name])
    df.loc[0] = [0.2, 20, '20160520', 1.1]
    df.loc[1] = [0.3, 40, '20160520', 0.9]
    df.loc[2] = [0.2, 20, '20170521', 1.1]
    df.loc[3] = [0.4, 20, '20170521', 0.8]
    res = ana.roi_level_per_year (df, tscore, 0.2, 0.3, 'long')
    assert res[res.year=='2016'].iloc[0,2] == 2
    assert res[res.year=='2016'].iloc[0,3] == 0
    assert res[res.year=='2016'].iloc[0,5] == 1
    assert res[res.year=='2016'].iloc[0,6] == -100

    res = ana.roi_level_per_year(df, tscore, 0.2, 0.3, 'short')
    assert res[res.year=='2017'].iloc[0,2] == 1
    assert res[res.year=='2017'].iloc[0,3] == -100
    assert res[res.year=='2017'].iloc[0,5] == 1
    assert res[res.year=='2017'].iloc[0,6] == -100


def test_roi_level():
    tscore = score.ScoreLabel(5, 1.0)
    score_name = tscore.get_name()
    df = pd.DataFrame(columns=['pred', 'close', 'date', score_name])
    df.loc[0] = [0.1, 20, '20160520', 1.1]
    df.loc[1] = [0.3, 40, '20160519', 0.9]
    df.loc[2] = [0.2, 20, '20170521', 1.1]
    df.loc[3] = [0.4, 20, '20170522', 0.8]
    df.loc[4] = [0.25, 20, '20170523', 1]

    levels = [1,2,3,4,-1]
    res = ana.roi_level(df, tscore, 'short', levels)

    assert res[res.top == 1.0].iloc[0, 2] == 1
    assert res[res.top == 1.0].iloc[0, 1] == 0.1
    assert res[res.top == 1.0].iloc[0, 3] == -100

    assert res[res.top == 2.0].iloc[0, 2] == 2
    assert res[res.top == 2.0].iloc[0, 1] == 0.2
    assert res[res.top == 2.0].iloc[0, 3] == -100

    assert res[res.top == -1].iloc[0, 2] == 5
    assert res[res.top == -1].iloc[0, 1] == 0.4
    assert res[res.top == -1].iloc[0, 3] == 20

    res = ana.roi_level(df, tscore, 'long', levels)
    assert res[res.top == 1.0].iloc[0, 2] == 1
    assert res[res.top == 1.0].iloc[0, 1] == 0.4
    assert res[res.top == 1.0].iloc[0, 3] == -200

    assert res[res.top == 2.0].iloc[0, 2] == 2
    assert res[res.top == 2.0].iloc[0, 1] == 0.3
    assert res[res.top == 2.0].iloc[0, 3] == -150

    assert res[res.top == -1].iloc[0, 2] == 5
    assert res[res.top == -1].iloc[0, 1] == 0.1
    assert res[res.top == -1].iloc[0, 3] == -20


def test_roi_last_months():
    tscore = score.ScoreLabel(5, 1.0)
    score_name = tscore.get_name()
    df = pd.DataFrame(columns=['pred', 'close', 'date', score_name])
    df.loc[0] = [0.1, 20, '2016-05-31', 1.1]
    df.loc[1] = [0.3, 40, '2016-05-30', 0.9]
    df.loc[2] = [0.2, 20, '2016-06-01', 1.1]
    df.loc[3] = [0.4, 20, '2016-06-02', 0.8]
    df.loc[4] = [0.25, 20, '2016-06-03', 1]

    res = ana.roi_last_months(df, tscore, 0.2, 0.3, 'long')
    assert res[res.month=='2016-05'].iloc[0,1] == 0.2
    assert res[res.month=='2016-05'].iloc[0,2] == 1
    assert res[res.month=='2016-05'].iloc[0,3] == -100
    assert res[res.month=='2016-05'].iloc[0,4] == 0.3
    assert res[res.month=='2016-05'].iloc[0,5] == 1
    assert res[res.month=='2016-05'].iloc[0,6] == -100

    assert res[res.month=='2016-06'].iloc[0,1] == 0.2
    assert res[res.month=='2016-06'].iloc[0,2] == 3
    assert res[res.month=='2016-06'].iloc[0,3] == -100.0/3
    assert res[res.month=='2016-06'].iloc[0,4] == 0.3
    assert res[res.month=='2016-06'].iloc[0,5] == 1
    assert res[res.month=='2016-06'].iloc[0,6] == -200


    res = ana.roi_last_months(df, tscore, 0.2, 0.3, 'short')
    assert res[res.month == '2016-05'].iloc[0, 1] == 0.2
    assert res[res.month == '2016-05'].iloc[0, 2] == 1
    assert res[res.month == '2016-05'].iloc[0, 3] == -100
    assert res[res.month == '2016-05'].iloc[0, 4] == 0.3
    assert res[res.month == '2016-05'].iloc[0, 5] == 2
    assert res[res.month == '2016-05'].iloc[0, 6] == 0

    assert res[res.month == '2016-06'].iloc[0, 1] == 0.2
    assert res[res.month == '2016-06'].iloc[0, 2] == 1
    assert res[res.month == '2016-06'].iloc[0, 3] == -100.0
    assert res[res.month == '2016-06'].iloc[0, 4] == 0.3
    assert res[res.month == '2016-06'].iloc[0, 5] == 2
    assert res[res.month == '2016-06'].iloc[0, 6] == -50

def test_accurate_level():
    tscore = score.ScoreLabel(5, 1.0)
    score_name = tscore.get_name()
    df = pd.DataFrame(columns=['pred', score_name])
    df.loc[0] = [0.1, 1]
    df.loc[1] = [0.9, 1]
    df.loc[2] = [0.2, 0]
    df.loc[3] = [0.6, 0]
    res = ana.accurate_level(df, tscore, 'long', [1,2,3,4])
    assert res[res.top == '1'].iloc[0, 1] == 1.0
    assert res[res.top == '1'].iloc[0, 2] == 0.9
    assert res[res.top == '2'].iloc[0, 1] == 0.5
    assert res[res.top == '2'].iloc[0, 2] == 0.6
    assert res[res.top == '3'].iloc[0, 1] == 1.0/3
    assert res[res.top == '3'].iloc[0, 2] == 0.2

    res = ana.accurate_level(df, tscore, 'short', [1, 2, 3, 4])
    assert res[res.top == '1'].iloc[0, 1] == 0
    assert res[res.top == '1'].iloc[0, 2] == 0.1
    assert res[res.top == '2'].iloc[0, 1] == 0.5
    assert res[res.top == '2'].iloc[0, 2] == 0.2
    assert res[res.top == '3'].iloc[0, 1] == 2.0/3
    assert res[res.top == '3'].iloc[0, 2] == 0.6

def test_accurate_topN():
    tscore = score.ScoreLabel(5, 1.0)
    score_name = tscore.get_name()
    df = pd.DataFrame(columns=['pred', score_name])
    df.loc[0] = [0.1, 1]
    df.loc[1] = [0.9, 1]
    df.loc[2] = [0.2, 0]
    df.loc[3] = [0.6, 0]

    res = ana.accurate_topN(df, tscore, 2, 'long')
    assert res[res.top == '2'].iloc[0, 1] == 0.5
    assert res[res.top == '2'].iloc[0, 2] == 0.6

    res = ana.accurate_topN(df, tscore, 2, 'short')
    assert res[res.top == '2'].iloc[0, 1] == 0.5
    assert res[res.top == '2'].iloc[0, 2] == 0.2

if __name__ == '__main__':
    for i in range(1, 10000):
        print(i)
        test_get_stock()
