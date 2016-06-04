#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
from sklearn.ensemble import GradientBoostingClassifier
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import train.model_param_set as params

l_params = params.d_all.keys()
