#!/bin/sh 
######################### 
#@author binred@outlook.com
#@date 
#@desc TODO 
######################### 
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
export SCRIPT_PATH=`dirname $(readlink -f $0)` # get the path of the script 

cd "$SCRIPT_PATH" 
 
python model_tuner_test.py --input=data/price_series4/

popd  > /dev/null # return the directory orignal

