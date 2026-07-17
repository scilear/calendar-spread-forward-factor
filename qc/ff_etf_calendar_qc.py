# region imports
from AlgorithmImports import *
import math
# endregion

"""
FF ETF Calendar — combo-fill bracketing backtest (pre-registered 2026-07-08,
CSFF_practical_review_2026-07-07.md, "QC ETF combo-fill backtest" section).

Question: does the forward-factor calendar edge on LIQUID SECTOR ETFs survive
realistic fills? One run brackets the fill spectrum: legs execute as MARKET
orders (= crossing the full half-spread, the k=0.5 base case) while quoted
bid/ask of every leg is RECORDED at entry and exit minutes. Post-analysis
reconstructs P&L at k in {0 (mid), 0.25, 0.5, 1.0} x leg-sum and solves k*
(breakeven crossing fraction). Decision rule: backtest alone can KILL
(EV<0 at k=0.25); only live BAG combo snapshots (R0) can KEEP.

Structure: ATM PUT calendar (sell front ~30 DTE put, buy back ~60 DTE put,
same strike) = the live scanner's primary structure. Entry 15:15 ET when
FF > 20 (FF computed from BS-inverted put-mid IVs — NOT QC Greeks, gotcha 8).
Exit: TP40 (mark >= 1.4x entry mid debit) or front DTE <= 1 (hard close
before expiry, no assignment handling needed, gotcha 5/5b). One position per
ticker, 1 contract, re-entry allowed after exit.

Built against quantconnect-gotchas.md items 1, 6, 8, 10 (minute resolution).
Output: "FFTRADE|" log lines + ObjectStore csv "ff_etf_calendar_trades.csv".
"""


class FFEtfCalendar(QCAlgorithm):

    TICKERS = ["XLY", "KRE", "XLV", "XLC", "XLE", "XLI", "XLU", "XLK", "SPY"]
    FF_MIN = 20.0
    FRONT_LO, FRONT_HI, FRONT_TGT = 20, 40, 30
    BACK_LO, BACK_HI, BACK_TGT = 45, 75, 60
    TP_MULT = 1.40          # TP40: exit when calendar mid >= 1.4x entry mid debit
    CLOSE_DTE = 1           # hard close both legs when front DTE <= 1
    RISK_FREE = 0.04        # BS inversion; FF is a vol RATIO, r error ~cancels

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2026, 7, 1)
        self.SetCash(100_000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Margin)
        self.opts = {}
        for t in self.TICKERS:
            eq = self.AddEquity(t, Resolution.Minute)
            eq.SetDataNormalizationMode(DataNormalizationMode.Raw)  # strikes nominal
            o = self.AddOption(t, Resolution.Minute)
            # gotcha 1: lower bound 0 so held contracts stay in universe
            o.SetFilter(lambda u: u.Strikes(-3, 3).Expiration(0, self.BACK_HI))
            self.opts[t] = o.Symbol
        self.pos = {}           # ticker -> dict(trade record)
        self.rows = []          # completed trade rows for ObjectStore
        self.Schedule.On(self.DateRules.EveryDay("SPY"),
                         self.TimeRules.At(15, 15), self.Scan)

    # ---------- pricing helpers ----------

    @staticmethod
    def _bs_put(s, k, t, r, iv):
        if iv <= 0 or t <= 0:
            return max(k - s, 0.0)
        d1 = (math.log(s / k) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
        d2 = d1 - iv * math.sqrt(t)
        N = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        return k * math.exp(-r * t) * N(-d2) - s * N(-d1)

    def _put_iv(self, s, k, t, price):
        if price <= max(k * math.exp(-self.RISK_FREE * t) - s, 0.0):
            return None
        lo, hi = 0.01, 3.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._bs_put(s, k, t, self.RISK_FREE, mid) < price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _quote(c):
        b, a = c.BidPrice, c.AskPrice
        if b is None or a is None or b <= 0 or a <= 0 or a < b:  # gotcha 6
            return None
        return float(b), float(a)

    # ---------- daily scan ----------

    def Scan(self):
        slice_ = self.CurrentSlice
        if slice_ is None:
            return
        for t in self.TICKERS:
            chain = slice_.OptionChains.get(self.opts[t])
            if t in self.pos:
                self._manage(t, chain)
            elif chain is not None:
                self._try_enter(t, chain)

    def _pick(self, chain, spot):
        """Return (front_put, back_put) at a shared ATM strike, or None."""
        puts = [c for c in chain if c.Right == OptionRight.Put]
        if not puts:
            return None
        today = self.Time.date()
        def band(lo, hi, tgt):
            exps = {c.Expiry.date() for c in puts
                    if lo <= (c.Expiry.date() - today).days <= hi}
            return min(exps, key=lambda e: abs((e - today).days - tgt)) if exps else None
        fe = band(self.FRONT_LO, self.FRONT_HI, self.FRONT_TGT)
        be = band(self.BACK_LO, self.BACK_HI, self.BACK_TGT)
        if fe is None or be is None or fe >= be:
            return None
        fs = {c.Strike: c for c in puts if c.Expiry.date() == fe}
        bs = {c.Strike: c for c in puts if c.Expiry.date() == be}
        shared = set(fs) & set(bs)
        if not shared:
            return None
        k = min(shared, key=lambda x: abs(x - spot))
        return fs[k], bs[k]

    def _try_enter(self, t, chain):
        spot = self.Securities[t].Price
        if spot <= 0:
            return
        pair = self._pick(chain, spot)
        if pair is None:
            return
        f, b = pair
        qf, qb = self._quote(f), self._quote(b)
        if qf is None or qb is None:
            return
        today = self.Time.date()
        tf = (f.Expiry.date() - today).days / 365.0
        tb = (b.Expiry.date() - today).days / 365.0
        ivf = self._put_iv(spot, float(f.Strike), tf, 0.5 * (qf[0] + qf[1]))
        ivb = self._put_iv(spot, float(b.Strike), tb, 0.5 * (qb[0] + qb[1]))
        if ivf is None or ivb is None:
            return
        fwd_var = (tb * ivb * ivb - tf * ivf * ivf) / (tb - tf)
        sigma_fwd = math.sqrt(max(fwd_var, 0.0))
        ff = (ivf - sigma_fwd) / ivf * 100.0
        if ff < self.FF_MIN:
            return
        mid_debit = 0.5 * (qb[0] + qb[1]) - 0.5 * (qf[0] + qf[1])
        if mid_debit <= 0:
            return
        self.Sell(f.Symbol, 1)
        self.Buy(b.Symbol, 1)
        self.pos[t] = {
            "ticker": t, "entry_date": str(today), "ff": round(ff, 2),
            "strike": float(f.Strike), "front_exp": str(f.Expiry.date()),
            "back_exp": str(b.Expiry.date()), "spot_entry": round(spot, 2),
            "e_f_bid": qf[0], "e_f_ask": qf[1], "e_b_bid": qb[0], "e_b_ask": qb[1],
            "mid_debit": round(mid_debit, 4),
            "fsym": f.Symbol, "bsym": b.Symbol,
        }
        self.Log(f"FFOPEN|{t}|{today}|ff={ff:.1f}|debit={mid_debit:.3f}")

    def _sec_quote(self, sym):
        # held legs may leave the filtered chain when spot drifts beyond the
        # +/-3 strike band; invested securities stay subscribed -> quote from
        # Securities, not the chain (else TP40 silently stops firing)
        s = self.Securities.get(sym)
        if s is None:
            return None
        b, a = s.BidPrice, s.AskPrice
        if b is None or a is None or b <= 0 or a <= 0 or a < b:  # gotcha 6
            return None
        return float(b), float(a)

    def _manage(self, t, chain):
        p = self.pos[t]
        qf = self._sec_quote(p["fsym"])
        qb = self._sec_quote(p["bsym"])
        dte = (datetime.strptime(p["front_exp"], "%Y-%m-%d").date()
               - self.Time.date()).days
        reason = None
        if qf and qb:
            mark = 0.5 * (qb[0] + qb[1]) - 0.5 * (qf[0] + qf[1])
            if mark >= self.TP_MULT * p["mid_debit"]:
                reason = "tp40"
        if reason is None and dte <= self.CLOSE_DTE:
            reason = "time"
        if reason is None:
            return
        self._close(t, reason, qf, qb)

    def _close(self, t, reason, qf, qb):
        p = self.pos.pop(t)
        self.Buy(p["fsym"], 1)      # buy back short front
        self.Sell(p["bsym"], 1)     # sell long back
        p.update({
            "exit_date": str(self.Time.date()), "exit_reason": reason,
            "x_f_bid": qf[0] if qf else "", "x_f_ask": qf[1] if qf else "",
            "x_b_bid": qb[0] if qb else "", "x_b_ask": qb[1] if qb else "",
        })
        p.pop("fsym"); p.pop("bsym")
        self.rows.append(p)
        self.Log(f"FFCLOSE|{t}|{p['exit_date']}|{reason}")

    def OnAssignmentOrderEvent(self, event):
        # rare early assignment on the short front put: flatten everything for
        # that underlying and drop the trade (recorded as 'assigned')
        for t, p in list(self.pos.items()):
            if event.Symbol == p["fsym"]:
                self.Log(f"FFASSIGN|{t}|{self.Time.date()}")
                self._close(t, "assigned", self._sec_quote(p["fsym"]),
                            self._sec_quote(p["bsym"]))
                self.Liquidate(t)

    def OnEndOfAlgorithm(self):
        for t in list(self.pos):
            self._close(t, "end_of_data", None, None)
        if not self.rows:
            return
        cols = list(self.rows[0].keys())
        csv = ",".join(cols) + "\n" + "\n".join(
            ",".join(str(r.get(c, "")) for c in cols) for r in self.rows)
        self.ObjectStore.Save("ff_etf_calendar_trades.csv", csv)
        self.Log(f"FFDONE|{len(self.rows)} trades saved to ObjectStore")
