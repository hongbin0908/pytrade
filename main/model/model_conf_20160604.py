#!/usr/bin/env python
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os
local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

import train.model_param_set as params

l_params = [
        "ta1_GBCv1n1000md3_l3_s2000e2009",
        "ta1_GBCv1n1000md3_l3_s2001e2010",
        "ta1_GBCv1n1000md3_l3_s2002e2011",
        "ta1_GBCv1n1000md3_l3_s2003e2012",
        "ta1_GBCv1n1000md3_l3_s2004e2013",
        "ta1_GBCv1n1000md3_l3_s2005e2014",
        ]
