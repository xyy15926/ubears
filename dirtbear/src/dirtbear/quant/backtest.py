#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: backtest.py
#   Author: xyy15926
#   Created: 2026-04-07 19:14:16
#   Updated: 2026-04-20 21:56:46
#   Description:
# ---------------------------------------------------------

# %%
import logging

import numpy as np

# from pathlib import Path
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload

    from dirtbear.visual import kline

    reload(kline)

from talib import MA

from dirtbear.visual.kline import compose_kline

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class SimpleBacktest:
    """A simple backtest.

    Attrs:
    ---------------------------
    initital_capital: float.
    trades_lot: NDA
    trades_prices: NDA.
    cash: NDA.
    positions_lot: NDA.
    capital: NDA.
    rules: Dict of constants representing the trade rules.
    """

    def __init__(
        self,
        initital_capital: float = 10000,
        allow_borrow_money: bool = False,
        allow_borrow_stocks: bool = False,
        rules: dict | None = None,
    ):
        """Init the backtest.

        Params:
        ---------------------------
        initital_capital:
        allow_borrow_money:
        allow_borrow_stocks:
        rules: Trading rules that will override the default rules.
        """
        self.initital_capital = initital_capital
        self.trades_lot = None
        self.trades_prices = None
        self.cash = None
        self.positions_lot = None
        self.capital = None
        self.rules = {
            "LOT": 100,
            "SLIPPAGE": 0.001,
            "COMMISION_BUY_RATE": 0.0005,
            "COMMISION_BUY_MIN": 2,
            "COMMISION_SELL_RATE": 0.001,
            "COMMISION_SELL_MIN": 5,
            "BORROW_MONEY_MIN": 10000,
            "BORROW_MONEY_RATE": 0.001,
            "BORROW_STOCKS_LOT_MIN": 1,
            "BORROW_STOCKS_RATE": 0.001,
        }
        if rules is not None:
            self.rules.update(rules)

        # Only valid when borrowing is allowed.
        self.allow_borrow_money = allow_borrow_money
        self.allow_borrow_stocks = allow_borrow_stocks
        self.borrowed_money = None

    def _init_arrs(self, parr):
        """Init attributes as NDA."""
        arr_len = parr.shape[0]
        self.trades_lot = np.zeros(arr_len, dtype=np.int32)
        self.trades_prices = np.zeros(arr_len, dtype=np.float32)
        self.cash = np.zeros(arr_len, dtype=np.float32)
        self.cash[0] = self.initital_capital
        self.positions_lot = np.zeros(arr_len, dtype=np.int32)
        self.capital = np.zeros(arr_len, dtype=np.float32)
        self.capital[0] = self.initital_capital
        if self.allow_borrow_money or self.allow_borrow_stocks:
            self.borrowed_money = np.zeros(arr_len, dtype=np.float32)

    def process(self, parr: np.ndarray, strategy_func: callable):
        """Run the strategy.

        Params:
        -------------------------------
        parr: [[open, close, low, high, volume],]
        strategy_func: Callable strategy that return the number of lots
          and the prices to buy or sell.

        Return:
        -------------------------------
        Dict representing the evaluation of the return.
        """
        arr_len = parr.shape[0]
        self._init_arrs(parr)
        trade_signals, trade_prices = strategy_func(parr)
        if trade_prices is None:
            logger.warning(
                "No trade price provided, the average of open "
                "and close price will be used."
            )
            trade_prices = parr[:, 1]
        for idx in range(1, arr_len):
            sig = trade_signals[idx - 1]
            price = trade_prices[idx - 1]
            lotn, borrowed = 0, 0
            if sig > 0:
                lotn, borrowed = self.try_buy(idx, sig, price)
            elif sig < 0:
                lotn, borrowed = self.try_sell(idx, sig, price)
            self.update_daily(idx, price, lotn, borrowed, parr[idx, 1])

        eva_return = evaluate_return(self.initital_capital, self.capital)
        return eva_return

    def try_buy(self, idx, sig, price):
        """Try to buy stock.

        Params:
        -------------------------------
        idx: The index of the trading date.
        sig: The percentage of the initital capital to buy.
        price: The price to buy.

        Return:
        -------------------------------
        try_lot: Lots to buy or sell.
        borrowed: Money borrowed to buy stock.
        """
        SLIPPAGE = self.rules["SLIPPAGE"]
        BORROW_MONEY_MIN = self.rules["BORROW_MONEY_MIN"]
        LOT = self.rules["LOT"]

        unit_buy = price * (1 + SLIPPAGE) * LOT
        try_cash = self.capital[idx - 1] * abs(sig)
        borrowed = 0
        if not self.allow_borrow_money:
            max_cash = min(self.cash[idx - 1], try_cash)
            try_lot = max_cash // unit_buy
        else:
            # Borrow money.
            if try_cash > self.cash[idx - 1]:
                borrowed = (
                    (try_cash - self.cash[idx - 1]) // BORROW_MONEY_MIN + 1
                ) * BORROW_MONEY_MIN
            try_lot = try_cash // (price * (1 + SLIPPAGE) * LOT)

        return try_lot, borrowed

    def try_sell(self, idx, sig, price):
        """Try to buy stock.

        Params:
        -------------------------------
        idx: The index of the trading date.
        sig: The percentage of the initital capital to sell.
        price: The price to sell.

        Return:
        -------------------------------
        try_lot: Lots to buy or sell.
        borrowed: Capital of the stock borrowed to sell.
        """
        SLIPPAGE = self.rules["SLIPPAGE"]
        BORROW_STOCKS_LOT_MIN = self.rules["BORROW_STOCKS_LOT_MIN"]
        LOT = self.rules["LOT"]

        unit_sell = price * (1 - SLIPPAGE) * LOT
        unit_buy = price * (1 + SLIPPAGE) * LOT
        try_lot = self.capital[idx - 1] * abs(sig) // unit_sell
        borrowed = 0
        if not self.allow_borrow_stocks:
            try_lot = min(self.positions_lot[idx - 1], try_lot)
        else:
            # Borrow stocks.
            if try_lot > self.positions_lot[idx - 1]:
                borrowed_lot = try_lot - self.positions_lot[idx - 1]
                borrowed = (
                    (borrowed_lot // BORROW_STOCKS_LOT_MIN + 1)
                    * BORROW_STOCKS_LOT_MIN
                    * unit_buy
                )

        return -try_lot, borrowed

    def update_daily(self, idx, bs_price, lotn, borrowed, cls_price):
        """Update the daily capital.

        Params:
        -------------------------------
        idx: The index of the trading date.
        bs_price: The price to buy or sell.
        lotn: Lots to buy or sell.
        borrowed: Value of borrowed cash or stock.
        cls_price: Close price of the day.
        """
        SLIPPAGE = self.rules["SLIPPAGE"]
        LOT = self.rules["LOT"]
        BORROW_STOCKS_RATE = self.rules["BORROW_STOCKS_RATE"]
        BORROW_MONEY_RATE = self.rules["BORROW_MONEY_RATE"]
        COMMISION_BUY_RATE = self.rules["COMMISION_BUY_RATE"]
        COMMISION_BUY_MIN = self.rules["COMMISION_BUY_MIN"]
        COMMISION_SELL_RATE = self.rules["COMMISION_SELL_RATE"]
        COMMISION_SELL_MIN = self.rules["COMMISION_SELL_MIN"]

        slipped_price = bs_price * (1 + SLIPPAGE * (lotn > 0))
        self.trades_prices[idx] = slipped_price
        self.trades_lot[idx] = lotn
        self.positions_lot[idx] = self.positions_lot[idx - 1] + lotn

        # Calculate the trade commisions.
        comission = 0
        cash_add = lotn * LOT * slipped_price
        if lotn > 0:
            comission = max(cash_add * COMMISION_BUY_RATE, COMMISION_BUY_MIN)
        elif lotn < 0:
            comission = max(
                -cash_add * COMMISION_SELL_RATE, COMMISION_SELL_MIN
            )
        self.cash[idx] = self.cash[idx - 1] - cash_add - comission

        # Calculate the borrowing commisions.
        if self.allow_borrow_money or self.allow_borrow_stocks:
            self.borrowed_money[idx] = self.borrowed_money[idx - 1] + borrowed
            brate = (
                BORROW_MONEY_RATE
                if self.positions_lot[idx] > 0
                else BORROW_STOCKS_RATE
            )
            self.cash[idx] -= self.borrowed_money[idx] * brate
            if self.cash[idx] < 0:
                logger.warning(
                    f"Cash is not enough for the commissions "
                    f"for borrowing at {idx}."
                )
        self.capital[idx] = (
            self.cash[idx] + self.positions_lot[idx] * LOT * cls_price
        )

    def compose_kline(
        self,
        dates: np.ndarray,
        parr: np.ndarray,
    ):
        """Compose a kline.

        Params:
        ---------------------------
        dates: Trade dates.
        parr: [[open, close, low, high, volume],]

        Return:
        ---------------------------
        GirdChart derived from pyecharts.charts.Grid
        """
        chart = compose_kline(
            dates,
            parr[:, :4].tolist(),
            list(
                zip(
                    range(parr.shape[0]),
                    parr[:, 4].tolist(),
                    ((parr[:, 1] > parr[:, 0]).astype(int) * 2 - 1).tolist(),
                    strict=True,
                )
            ),
            {
                "MA5": MA(parr[:, 1], 5).round(2).tolist(),
                "MA30": MA(parr[:, 1], 30).round(2).tolist(),
            },
            {
                "Capital": self.capital.round(2).tolist(),
            },
            list(
                zip(
                    dates,
                    self.trades_prices.tolist(),
                    self.trades_lot.tolist(),
                    strict=True,
                )
            ),
            self.cash.round(2).tolist(),
            (self.capital - self.cash).round(2).tolist(),
        )

        return chart


# %%
def evaluate_return(initital_capital: float, capital: np.ndarray) -> dict:
    """Evaluate the return."""
    NO_RISK_RETURN = 0.015

    total_return = capital[-1] / initital_capital - 1
    daily_returns = capital[1:] / capital[:-1] - 1
    daily_returns_mean = np.mean(daily_returns)
    annualized_return = daily_returns_mean * 252
    sharpe_ratio = (annualized_return - NO_RISK_RETURN) / np.std(daily_returns)
    max_drawback = np.min(capital / np.maximum.accumulate(capital)) - 1

    return {
        "final_capital": capital[-1],
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawback": max_drawback,
    }
