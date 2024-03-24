import alpaca_trade_api as tradeapi
from alpaca_trade_api import StreamConn
# 439f5dc6-240d-4ffc-a3d0-280d42e567cd

ALPACE_BASE_URL = "https://paper-api.alpaca.markets";

class PythonTradingBot:
    def __init__(self):
        self.alpaca = tradeapi.REST('API_KEY','API_SECRET_KEY', ALPACE_BASE_URL, api_versino='v2')
    def run(self):
        async def on_minute(conn, channel, bar):
