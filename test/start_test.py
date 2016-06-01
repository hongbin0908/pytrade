#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
import numpy as np

local_path = os.path.dirname(__file__)
main_root = os.path.join(local_path, '..', 'main')
print main_root
sys.path.append(main_root)

import train
from train import start


def test_get_all_from():
    symToTa = start.get_all_from(os.path.join(local_path, 'start_test.dir', 'ta'))
    assert 2 == len(symToTa)
    assert (20, 89) == symToTa["MSFT"].shape
    assert (20, 89) == symToTa["YHOO"].shape
    df = symToTa["MSFT"].sort_index()
    npDates = df.index.values
    assert np.datetime64('2016-05-02') == npDates[0]
    assert np.datetime64('2016-05-27') == npDates[-1]

def test_merge():
    symToTa = start.get_all_from(os.path.join(local_path, 'start_test.dir', 'ta'))
    df = start.merge(symToTa, '2016-05-02', '2016-05-28')
    assert (40, 89) == df.shape
    df = start.merge(symToTa, '2016-05-03', '2016-05-28')
    assert (38, 89) == df.shape
    df = start.merge(symToTa, '2016-05-02', '2016-05-27')
    assert (40, 89) == df.shape
    df = start.merge(symToTa, '2016-05-02', '2016-05-26')
    assert (38, 89) == df.shape

def test_get_feat_names():
    symToTa = start.get_all_from(os.path.join(local_path, 'start_test.dir', 'ta'))
    df = start.merge(symToTa, '2016-05-02', '2016-05-28')
    assert 70 == len(start.get_feat_names(df))

def test_build_trains():
    symToTa = start.get_all_from(os.path.join(local_path, 'start_test.dir', 'ta'))
    df = start.merge(symToTa, '2016-05-02', '2016-05-28')
    lLabel = [x for x in df.columns if x.startswith('label')]
    for sym in symToTa:
        for each in lLabel[3:]:
            del symToTa[sym][each]
        del symToTa[sym]["close_shift"]
    df = start.build_trains(symToTa, '2016-05-02', '2016-05-27')
    assert (34, 79) == df.shape




# content of test_sample.py
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 4

import pytest
def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit): f()

def test_needsfiles(tmpdir):
    print (tmpdir)
    assert 1
