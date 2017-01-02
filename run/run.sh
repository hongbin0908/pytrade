#!/bin/sh
############################
#@author hongbin@youzan.com
#@date
#@desc TODO
############################
export PATH=/usr/bin:$PATH
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script
pushd . > /dev/null
cd "$SCRIPT_PATH"/../

rm -rf data/ta/*
rm -rf data/cross/*
scl enable rh-python34 bash
python3 run/run.py

 
popd  > /dev/null # return the directory orignal
