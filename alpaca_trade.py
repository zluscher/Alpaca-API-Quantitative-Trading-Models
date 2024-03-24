import alpaca_trade_api as api
from math import floor

# btc = "ETHUSD"
alpaca = api.REST('AKQ44O2G5H6LCTVJYR1E', 'IrkFYQ6ndFhQSnpViY2122xfM8jT91ad6i79UpxY', 'https://api.alpaca.markets')

# account = alpaca.get_account()
# account_cash = float(account.cash)
# cash_to_spend = account_cash
#
# btc_snapshot = alpaca.get_crypto_snapshot(symbol=btc, exchange="CBSE")
# # aapl_snapshot = alpaca.get_snapshots(Stock_List[0])
# # aapl_latest_price = aapl_snapshot.latest_quote.ap
# btc_latest_price = btc_snapshot.latest_quote.ap
#
# def calculate_order_size(cash_to_spend, latest_price):
#     change = 0
#
#     precision_factor = 10000
#     units_to_buy = floor(cash_to_spend * precision_factor / latest_price)
#     units_to_buy /= precision_factor
#     return units_to_buy
#
#
# btc_units = calculate_order_size(cash_to_spend, btc_latest_price)
# alpaca.submit_order(symbol=btc, qty=0.001, side="buy", type='market', time_in_force='gtc')



btc = 'LTCUSD'

print("Order Excecuted")

alpaca.submit_order(symbol=btc, qty=0.005, side="buy", type='market', time_in_force='gtc')

# print(bs + units_to_buy)
