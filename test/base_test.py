import sys,os
import pandas as pd

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main import base
from main.base import stock_fetcher as sf

def test_is_test_flag():
    assert base.is_test_flag()


def test_split_by_year():
    df = pd.read_csv(os.path.join(root, 'data', 'yeod', 'sp500_snapshot_20091231', 'IBM.csv'))
    len_total = len(df)
    len2 = 0
    for each in base.split_by_year(df):
        len2 += len(each)
    assert len_total == len2

def test_random_sort():
    df = pd.read_csv(os.path.join(root, 'data', 'yeod', 'sp500_snapshot_20091231', 'IBM.csv'))
    df = base.random_sort(df)

def test_get_stock():
    df = sf.get_stock('YHOO')
    df2 = sf.get_stock2('YHOO')
    assert len(df) == len(df2)
    assert len(df) > 1000


if __name__ == '__main__':
    test_get_stock()
