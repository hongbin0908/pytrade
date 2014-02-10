#-*-encoding:gbk-*-
'''
功能：
1.将价格数组输入，获得rsi数据
2.将价格数据输入，获得指定窗口内的平均价格数据
3.获取在指定index下的前后价格变动
4.根据指定阈值，输出分类
功能update：
1. 根据index，获取指定时间窗口内的线性拟合值(斜率)
2. 根据斜率获取price_signal,并进行样本采样
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
    if len(price_sum)<= index + 7 or index < 0:
        return None
    if price_sum[index + 7]/price_sum[index] > threshold + 1:
        return True
#    elif price_sum[index + 1]/price_sum[index] < 1 - threshold:
#        return False
    else:
        return False

'''
根据输入价格，指定时间窗，生成时间窗内数组，并进行信号量判断，输出数组，元素格式为(信号类型， 是否符合预期)
'''
def get_signal_result(price_list, timewindow):
    if len(price_list) <= timewindow or timewindow <= 0 :
        return []
    result_list = []
    for s in range(timewindow, price_list):
        prices = price_list[s-timewindow:s] 
        up_or_down = is_up_or_down(prices)
        is_signal = is_signal_get(prices, up_or_down)
        is_truly_up = is_priceup(price_sum, s, 0.05)
        insert_into_sample(is_truly_up, is_signal, result_list)

    return result_list

'''
根据输入价格，判断价格变动趋势
'''
def is_up_or_down(prices):
    slope = talib.LINEARREG_SLOPE(prices, len(prices))
    return slope[-1]
'''
根据价格变动趋势，获取信号： 若up_or_down >0:表示上升趋势，需要计算最小值，并判断是否突破该最小值
若up_or_down<0:则相反
'''
def is_signal_get(prices, up_or_down):
    if up_or_down >0 :
        return is_up_signal_get(prices)
    elif up_or_down < 0:
        return is_down_signal_get(prices)
    else:
        return None

'''
若up_or_down >0:表示上升趋势，需要计算最小值，并判断是否突破该最小值
'''
def is_up_signal_get(prices):
    maxindex_1 = -1
    maxvalue_1 = 0
    maxindex_2 = -1
    maxvalue_2 = 0
    min_0 = 6000000
    for s in range(0, len(prices)):
        if maxvalue_1 < prices[s]:
            maxvalue_1 = prices[s]
            maxindex_1 = s
    for s in range(0, len(prices)):
        if maxvalue_2 < prices[s] and prices[s] != maxvalue_1:
            maxvalue_2 = prices[s]
            maxindex_2 = s
    begin_index = maxindex_1
    if begin_index > maxindex_2:
        begin_index = maxindex_2
  
    for s in range(begin_index, abs(maxindex_1-maxindex_2)+begin_index):
        if min_0 > prices[s]:
            min_0 = prices[s]
    
    for s in range(abs(maxindex_1-maxindex_2)+begin_index, len(prices)):
        if prices[s] < min_0:
            if s != len(prices)-1:
                return False
            else:
                return True
    
    return False
        
'''
若up_or_down <0:表示下降趋势，需要计算最大值，并判断是否突破该最大值
'''
def is_up_signal_get(prices):
    minindex_1 = -1
    minvalue_1 = 600000
    minindex_2 = -1
    minvalue_2 = 600000
    max_0 = -1
    for s in range(0, len(prices)):
        if maxvalue_1 < prices[s]:
            maxvalue_1 = prices[s]
            maxindex_1 = s
    for s in range(0, len(prices)):
        if maxvalue_2 < prices[s] and prices[s] != maxvalue_1:
            maxvalue_2 = prices[s]
            maxindex_2 = s
    begin_index = maxindex_1
    if begin_index > maxindex_2:
        begin_index = maxindex_2
  
    for s in range(begin_index, abs(maxindex_1-maxindex_2)+begin_index):
        if min_0 > prices[s]:
            min_0 = prices[s]
    
    for s in range(abs(maxindex_1-maxindex_2)+begin_index, len(prices)):
        if prices[s] < min_0:
            if s != len(prices)-1:
                return False
            else:
                return True
    
    return False

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
        result = is_priceup(price_sum, s, 0.000)
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
        print "%.4f\t%.4f\t%.4f\t%d" %(rsi_index[s], price_sum[s], price_sum[s+7], sign)
    
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
