#-*-encoding:gbk-*-
class prices_judgement:
    def judge(self, prices, threshold):
        middle_index = len(prices)/2
        b_prices = 0.0
        for s in range(0, middle_index):
            b_prices += prices[s]
        a_prices = 0.0
        for s in range(middle_index+1, len(prices)):
            a_prices += prices[s]
        if a_prices/b_prices > 1+ threshold:
            return 1
        elif a_prices/b_prices < 1-threshold:
            return 0
        else:
            return None

if __name__ == "__main__":
    judgeer = prices_judgement()
    price_list = [1, 2, 3, 4, 5]
    result = judgeer.judge(price_list, 0.05)
    print result
    price_list = [1, 1, 1, 1, 1]
    result = judgeer.judge(price_list, 0.05)
    print result
