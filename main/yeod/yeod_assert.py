#!/usr/bin/env python3

import os, sys, time
import pandas as pd
import numpy as np
import multiprocessing
import logging

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', '..')
sys.path.append(root)

from main.base import stock_fetcher as sf

logging.basicConfig(level=logging.WARN)
local_path = os.path.dirname(__file__)

def work(conf):
    yeod_dir1 = conf.get_yeod_dir(backtrace=0)
    yeod_dir2 = conf.get_yeod_dir(backtrace=1)
