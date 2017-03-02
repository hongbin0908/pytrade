#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @author  Bin Hong

import sys
import os


local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..', "..")
sys.path.append(root)

from main.bitlize import build

def work(confer):
    build.work(confer)
