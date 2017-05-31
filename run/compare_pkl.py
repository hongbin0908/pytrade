#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-
# @author  Bin Hong

import sys
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal


local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

from main import base


df1 = pd.read_pickle(sys.argv[1])
df1.reset_index(drop=True, inplace=True)
df2 = pd.read_pickle(sys.argv[2])
df2.reset_index(drop=True, inplace=True)

last_date = list(df1.date.unque())[-1]

df2 = df2[df2.date <= last_date]
assert df1.shape == df2.shape


for col in base.get_feat_names(df1):
    print(col)
    assert_frame_equal(df1[[col]], df2[[col]])
