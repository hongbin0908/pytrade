#!/bin/sh 
######################### 
#@author binred@outlook.com
#@date 
#@desc TODO 
######################### 
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script 
pushd . > /dev/null 

cd "$SCRIPT_PATH" 
 
cd ../tools/ ; 
python daily_bars_getter.py

cd "$SCRIPT_PATH" 
python model_build_price_series2.py  --window=14
python model_tuner.py 

popd  > /dev/null # return the directory orignal

