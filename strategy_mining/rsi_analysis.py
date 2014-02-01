#-*-encoding:gbk-*-
'''
功能：
1.将价格数组输入，获得rsi数据
2.将价格数据输入，获得指定窗口内的平均价格数据
3.获取在指定index下的前后价格变动
4.根据指定阈值，输出分类 
'''
import sys
import os
import talib
import numpy
'''
根据输入的数据，调用talibrsi函数，计算rsi
'''
def get_rsi(price_list, timewindow):
    size = len(price_list)
    if timewindow < 0 or size < timewindow:
        return None
    rsi_index = talib.RSI(price_list, timewindow)
    return rsi_index

'''
根据股票价格数据，获取平均值函数
'''
def get_avg_price(price_list, timewindow):
    size = len(price_list)
    if timewindow < 0 or size < timewindow:
        return None
    price_sum = talib.SUM(price_list, timewindow)
    return price_sum

'''
根据输入的平均值函数及index，判断是涨还是跌
'''
def is_priceup(price_sum, index, threshold):
    if len(price_sum)<= index + 1 or index < 0:
        return None
    if price_sum[index + 1]/price_sum[index] > threshold + 1:
        return True
    elif price_sum[index + 1]/price_sum[index] < 1 - threshold:
        return False
    else:
        return None

'''
根据输入价格，获取rsi/price平均值，并判断在指定index下的价格变化，若其与rsi理论一致，则标为1， 否则，为0
'''
def get_sample(price_list):
    rsi_index = get_rsi(price_list, 14)
    price_sum = get_avg_price(price_list, 7)
    for s in range(0, len(rsi_index)):
        if rsi_index[s] == None:
            continue
        sign = None
        result = is_priceup(price_sum, s, 0.02)
        if None == result:
            continue
        if rsi_index[s] > 80:
            if False == result:
                sign = 0
            else:
                sign = 1
        elif rsi_index[s] < 15:
            if True == result:
                sign = 0
            else:
                sign = 1
        else:
            continue
        if sign == None:
            continue
        print "%.4f\t%.4f\t%.4f\t%d" %(rsi_index[s], price_sum[s], price_sum[s+1], sign)
    
'''
加载文件，获取price_list
'''        
def load_data(filename):
    tprice = []
    fd = open(filename, "r")
    for j in fd:
        line_list = j.rstrip().split(",")
        try:
            price = float(line_list[4])
        except Exception,e:
            continue
        tprice.append(price)
    price_list = numpy.array(tprice)
    return price_list

if __name__ == "__main__":
    price_list = load_data(sys.argv[1])
    get_sample(price_list)    
