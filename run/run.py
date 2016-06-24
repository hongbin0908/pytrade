#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#@author  Bin Hong

import sys,os

local_path = os.path.dirname(__file__)
root = os.path.join(local_path,'..')
sys.path.append(root)


from main.yeod import yeod
from main.ta import build
from main.utils import time_me
from main.model import modeling as model

@time_me
def main(argv):
    yeod.main(["dow", 5])
    build.main(["dow", 'call1',5])
    model.main(['model_conf',1])
if __name__ == '__main__':
    main(sys.argv[1:])
