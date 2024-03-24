mid=(low+high)/2

start = datetime(2014, 1, 1)                # 回测起始时间
end   = datetime(2014, 12, 10)                # 回测结束时间
benchmark = 'HS300'                            # 使用沪深 300 作为参考标准
universe = set_universe('SH50')    # 股票池
capital_base = 100000                       # 起始资金


refresh_rate = 1
window = 10

# 本策略对于window非常非常敏感！！！

histFish = pd.DataFrame(0.0, index = universe, columns = ['preDiff', 'preFish', 'preState'])

def initialize(account):                    # 初始化虚拟账户状态
    account.amount = 10000
    account.universe = universe
    add_history('hist', window)


def handle_data(account):                # 每个交易日的买入卖出指令

    for stk in account.universe:
        prices = account.hist[stk]
        if prices is None:
            return

        preDiff = histFish.at[stk, 'preDiff']
        preFish = histFish.at[stk, 'preFish']
        preState = histFish.at[stk, 'preState']

        diff, fish = FisherTransIndicator(prices, preDiff, preFish)
        if fish > preFish:
            state = 1
        elif fish < preFish:
            state = -1
        else:
            state = 0

        if state == 1 and preState == -1:
            #stkAmount = int(account.amount / prices.iloc[-1]['openPrice'])
            order(stk, account.amount)
        elif state == -1 and preState == 1:
            order_to(stk, 0)

        histFish.at[stk, 'preDiff'] = diff
        histFish.at[stk, 'preFish'] = fish
        histFish.at[stk, 'preState'] = state


def FisherTransIndicator(windowData, preDiff, preFish):
    # This function calculate the Fisher Transform indicator based on the data
    # in the windowData.
    minLowPrice = min(windowData['lowPrice'])
    maxHghPrice = max(windowData['highPrice'])
    tdyMidPrice = (windowData.iloc[-1]['lowPrice'] + windowData.iloc[-1]['highPrice'])/2.0

    diffRatio = 0.33
    # 本策略对于diffRatio同样非常敏感！！！

    diff = (tdyMidPrice - minLowPrice)/(maxHghPrice - minLowPrice) - 0.5
    diff = 2 * diff
    diff = diffRatio * diff + (1.0 - diffRatio) * preDiff

    if diff > 0.99:
        diff = 0.999
    elif diff < -0.99:
        diff = -0.999

    fish = np.log((1.0 + diff)/(1.0 - diff))
    fish = 0.5 * fish + 0.5 * fish

    return diff, fish

    def FisherTransIndicator(windowData, preDiff, preFish, state):
    # This function calculate the Fisher Transform indicator based on the data
    # in the windowData.
    minLowPrice = min(windowData['lowestIndex'])
    maxHghPrice = max(windowData['highestIndex'])
    tdyMidPrice = (windowData.iloc[-1]['lowestIndex'] + windowData.iloc[-1]['highestIndex'])/2.0

    diffRatio = 0.5

    diff = (tdyMidPrice - minLowPrice)/(maxHghPrice - minLowPrice) - 0.5
    diff = 2 * diff

    if state == 1:
        diff = diffRatio * diff + (1 - diffRatio) * preDiff

    if diff > 0.995:
        diff = 0.999
    elif diff < -0.995:
        diff = -0.999

    fish = np.log((1 + diff)/(1 - diff))
    if state == 1:
        fish = 0.5 * fish + 0.5 * fish

    return diff, fish

    window = 10

index['diff'] = 0.0
index['fish'] = 0.0
index['preFish'] = 0.0

for i in range(window, index.shape[0]):
    windowData = index.iloc[i-window : i]
    if i == window:
        diff, fish = FisherTransIndicator(windowData, 0, 0, 1)
        index.at[i,'preFish'] = 0
        index.at[i,'diff'] = diff
        index.at[i,'fish'] = fish
    else:
        preDiff = index.iloc[i-1]['diff']
        preFish = index.iloc[i-1]['fish']
        diff, fish = FisherTransIndicator(windowData, preDiff, preFish, 1)
        index.at[i,'preFish'] = preFish
        index.at[i,'diff'] = diff
        index.at[i,'fish'] = fish


Plot(index, settings = {'x':'tradeDate','y':'closeIndex', 'title':u'沪深300指数历史收盘价'})
Plot(index, settings = {'x':'tradeDate','y':['fish', 'preFish'], 'title':u'沪深300指数Fisher Transform Indicator'})
