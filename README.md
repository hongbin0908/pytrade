pytrade
=======

my trading system depending on pyalgotrade


install
=======
1. install CPython
   sudo easy_install-2.7 cython
2. install TA-Lib
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   untar and cd
   ./configure --prefix=/usr
   make
   sudo make install
   sudo easy_install-2.7 TA-Lib
usage
=====
1. download the sp500 history stocks data
./tools/run_daily_bars_getter.sh 
defaultly the data is located in "/home/work/workplace/stock_data/"

2. cd strategy_mining; python model_trainig.py

