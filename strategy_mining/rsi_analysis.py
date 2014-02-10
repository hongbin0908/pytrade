#-*-encoding:gbk-*-
'''
���ܣ�
1.���۸��������룬���rsi����
2.���۸��������룬���ָ�������ڵ�ƽ���۸�����
3.��ȡ��ָ��index�µ�ǰ��۸�䶯
4.����ָ����ֵ���������
����update��
1. ����index����ȡָ��ʱ�䴰���ڵ��������ֵ(б��)
2. ����б�ʻ�ȡprice_signal,��������������
'''
import sys
import os
import talib
import numpy
'''
������������ݣ�����talibrsi����������rsi
'''
def get_rsi(price_list, timewindow):
    size = len(price_list)
    if timewindow < 0 or size < timewindow:
        return None
    rsi_index = talib.RSI(price_list, timewindow)
    return rsi_index

'''
���ݹ�Ʊ�۸����ݣ���ȡƽ��ֵ����
'''
def get_avg_price(price_list, timewindow):
    size = len(price_list)
    if timewindow < 0 or size < timewindow:
        return None
    price_sum = talib.SUM(price_list, timewindow)
    return price_sum

'''
���������ƽ��ֵ������index���ж����ǻ��ǵ�
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
��������۸�ָ��ʱ�䴰������ʱ�䴰�����飬�������ź����жϣ�������飬Ԫ�ظ�ʽΪ(�ź����ͣ� �Ƿ����Ԥ��)
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
��������۸��жϼ۸�䶯����
'''
def is_up_or_down(prices):
    slope = talib.LINEARREG_SLOPE(prices, len(prices))
    return slope[-1]
'''
���ݼ۸�䶯���ƣ���ȡ�źţ� ��up_or_down >0:��ʾ�������ƣ���Ҫ������Сֵ�����ж��Ƿ�ͻ�Ƹ���Сֵ
��up_or_down<0:���෴
'''
def is_signal_get(prices, up_or_down):
    if up_or_down >0 :
        return is_up_signal_get(prices)
    elif up_or_down < 0:
        return is_down_signal_get(prices)
    else:
        return None

'''
��up_or_down >0:��ʾ�������ƣ���Ҫ������Сֵ�����ж��Ƿ�ͻ�Ƹ���Сֵ
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
��up_or_down <0:��ʾ�½����ƣ���Ҫ�������ֵ�����ж��Ƿ�ͻ�Ƹ����ֵ
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
��������۸񣬻�ȡrsi/priceƽ��ֵ�����ж���ָ��index�µļ۸�仯��������rsi����һ�£����Ϊ1�� ����Ϊ0
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
�����ļ�����ȡprice_list
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
