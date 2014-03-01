#-*-encoding:gbk-*-
import numpy
class prices_judgement:
    def judge(self, prices, threshold, window):
        judge_list = numpy.array(prices)
        judge_list.fill(-2)
        for index, s in enumerate(prices):
            if index < window or index > len(prices) - window:
                continue
            b_sum = sum(prices[index-window:index])
            a_sum = sum(prices[index+1: index+window+1])
            judge_list[index] = a_sum*1.0/b_sum
#            if a_sum*1.0/b_sum > 1 + threshold:
#                judge_list[index] = 1
#            elif a_sum*1.0/b_sum < 1 - threshold :
#                judge_list[index] = 0
#            else:
#                judge_list[index] = -2
#
        return judge_list

if __name__ == "__main__":
    judgeer = prices_judgement()
    price_list = [1, 2, 3, 4, 5]
    result = judgeer.judge(price_list, 0.05, 2)
    print result
    price_list = [1, 1, 1, 1, 1]
    result = judgeer.judge(price_list, 0.05, 2)
    print result
