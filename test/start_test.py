#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os

local_path = os.path.dirname(__file__)
main_root = os.path.join(local_path, '..', 'main')
print main_root
sys.path.append(main_root)

import train
from train import start


def test_get_all_from():
    symToTa = start.get_all_from(os.path.join(local_path, 'start_test.dir', 'ta'))
    print symToTa
    assert 0


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
