# region imports
from AlgorithmImports import *
import math
from datetime import datetime
# endregion

"""
R5.2 A1 combo-execution probe (pre-registered 2026-09-11, RESEARCH_PLAN.md R5.2).

Base: ff_etf_calendar_qc.py (Scan/FF/TP40/front-minus-1d logic kept). Leg
MarketOrders replaced by an entry worker:

  PRIMARY: debit-paid ComboLimitOrder (positive net debit, per-leg Limit left
    None, IC0dte convention) pegged at scanner fill_mid, walked toward
    fill_aggressive in 3 x 10-min steps (cancel/replace), never crossing
    fill_aggressive. 30-min max-wait -> cancel + skip.
  FALLBACK (only if the combo path is rejected/unsupported): buy-first
    sequenced leg markets with 60s stagger - entry buys back-month then sells
    front-month; exit buys front (short) then sells back (long). Sell-first is
    never attempted (known Tastytrade buying-power rejection).

Scanner fill ladder (ff_trade_scanner.py:580-586):
  fill_aggressive = back_ask - front_bid   (limit_end, never cross)
  fill_mid        = back_mid - front_mid   (limit_start)
  fill_passive    = back_bid - front_ask
k_realized = (fill_px - mid) / (agg - mid); k=0 at mid, k=1 at aggressive.

Brokerage: Tastytrade (order validation is the test). PROCEED bar: median
realized k <= 0.5 AND zero buying-power rejections.

Tag convention: em=;spr=;mid=;ex=;lim= (+aid= attempt id, path=combo/fallback).
Every attempt logs (requested_lim, fill_px, spr, mid, k_realized).

Universe is ONE constant (swap for the A2 census lock without code changes).
"""


class FFCalendarAutomationR52(QCAlgorithm):

    # ---- swap surface: full universe arrives from the parallel A2 census ----
    TICKERS = ["XLV", "XLK", "SPY", "AAPL", "MSFT"]

    FF_MIN = 20.0
    FRONT_LO, FRONT_HI, FRONT_TGT = 20, 40, 30
    BACK_LO, BACK_HI, BACK_TGT = 45, 75, 60
    TP_MULT = 1.40
    CLOSE_DTE = 1
    RISK_FREE = 0.04

    WALK_STEP_MIN = 10.0
    WALK_STEPS = 3
    MAX_WAIT_MIN = 30.0
    FALLBACK_STAGGER_SEC = 60.0

    def Initialize(self):
        self.SetStartDate(2026, 1, 2)  # extended window: recent 10 sessions
        # produced zero FF>20 firings (max 6.7), so the mechanics validation
        # runs over YTD minute replay; the 2026-08-27..09-10 slice is reported
        # separately as the recent-window read (zero attempts).
        self.SetEndDate(2026, 9, 11)
        self.SetCash(100_000)
        self.SetTimeZone("America/New_York")
        self.SetBrokerageModel(BrokerageName.Tastytrade, AccountType.Margin)
        self.opts = {}
        for t in self.TICKERS:
            eq = self.AddEquity(t, Resolution.Minute)
            eq.SetDataNormalizationMode(DataNormalizationMode.Raw)
            o = self.AddOption(t, Resolution.Minute)
            # gotcha 1: lower bound 0 so held contracts stay subscribed
            o.SetFilter(lambda u: u.Strikes(-3, 3).Expiration(0, self.BACK_HI))
            self.opts[t] = o.Symbol
        self.state = {}       # ticker -> attempt/position dict
        self.attempt_rows = []  # one row per entry attempt (the A1 evidence)
        self.trade_rows = []    # one row per completed round trip
        self.bp_rejections = 0
        self.other_rejections = 0
        self.diag = self.GetParameter("diag_scan", "1") in (1, "1", "true", "True")
        self.Schedule.On(self.DateRules.EveryDay("SPY"),
                         self.TimeRules.At(15, 15), self.Scan)

    # ---------- pricing helpers (unchanged from base) ----------

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

    def _sec_quote(self, sym):
        s = self.Securities.get(sym)
        if s is None:
            return None
        b, a = s.BidPrice, s.AskPrice
        if b is None or a is None or b <= 0 or a <= 0 or a < b:  # gotcha 6
            return None
        return float(b), float(a)

    # ---------- scan ----------

    def Scan(self):
        sl = self.CurrentSlice
        if sl is None:
            return
        for t in self.TICKERS:
            chain = sl.OptionChains.get(self.opts[t])
            st = self.state.get(t)
            if st is not None:
                self._manage_open(t, st)
                continue
            if chain is not None:
                self._try_enter(t, chain)

    def OnData(self, slice_):
        # per-minute worker: walk steps, fallback stagger, TP40 checks
        for t in list(self.state):
            st = self.state[t]
            phase = st["phase"]
            if phase == "combo_working":
                self._combo_poll(t, st)
            elif phase == "fb_buy_placed":
                self._fallback_poll(t, st)
            elif phase == "held":
                self._manage_open(t, st)
            elif phase == "exit_working":
                self._exit_poll(t, st)

    def _pick(self, chain, spot):
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

    # ---------- entry ----------

    def _try_enter(self, t, chain):
        spot = self.Securities[t].Price
        if spot <= 0:
            if self.diag:
                self.Log(f"CSFFDIAG|{t}|{self.Time.date()}|no_spot")
            return
        pair = self._pick(chain, spot)
        if pair is None:
            if self.diag:
                n = len(list(chain)) if chain is not None else -1
                self.Log(f"CSFFDIAG|{t}|{self.Time.date()}|no_pick|chain={n}")
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
        if self.diag:
            self.Log(f"CSFFDIAG|{t}|{self.Time.date()}|ff={ff:.1f}|"
                     f"spot={spot:.1f}|K={float(f.Strike):.0f}")
        if ff >= 99.0:  # scanner FF>=99 guard (:608-609): artifact, skip
            return
        if ff < self.FF_MIN:
            return
        f_mid = 0.5 * (qf[0] + qf[1])
        b_mid = 0.5 * (qb[0] + qb[1])
        mid = b_mid - f_mid
        agg = qb[1] - qf[0]
        if mid <= 0:
            return
        if agg <= mid:
            # locked/crossed ladder: never cross -> immediate skip, logged
            self._record_attempt(t, today, ff, "combo", mid, mid, agg,
                                 qf, qb, None, "skip_badspread", f, b, spot)
            return
        aid = f"{today:%Y%m%d}-{t}"
        st = {
            "phase": "combo_working", "ticker": t, "aid": aid,
            "ff": round(ff, 2), "strike": float(f.Strike),
            "front_exp": str(f.Expiry.date()), "back_exp": str(b.Expiry.date()),
            "spot_entry": round(spot, 2),
            "e_f_bid": qf[0], "e_f_ask": qf[1],
            "e_b_bid": qb[0], "e_b_ask": qb[1],
            "mid": mid, "agg": agg, "spr": (qf[1] - qf[0]) + (qb[1] - qb[0]),
            "fsym": f.Symbol, "bsym": b.Symbol,
            "start": self.Time, "walk_step": 0, "ticket_id": None,
            "requested_lim": mid, "fallback_used": False,
            "combo_rejected": False,
        }
        self.state[t] = st
        self._place_combo(t, st, mid)

    def _combo_legs(self, st):
        return [Leg.Create(st["fsym"], -1), Leg.Create(st["bsym"], 1)]

    def _entry_tag(self, st, lim, path):
        return (f"em=1515;spr={st['spr']:.2f};mid={st['mid']:.3f};"
                f"ex={st['agg']:.3f};lim={lim:.3f};aid={st['aid']};path={path}")

    def _place_combo(self, t, st, lim):
        st["requested_lim"] = lim
        try:
            ticket = self.ComboLimitOrder(self._combo_legs(st), 1, lim,
                                          tag=self._entry_tag(st, lim, "combo"))
        except Exception as e:
            self.Log(f"CSFFREJECT|{st['aid']}|combo_submit_fail|{e}")
            st["combo_rejected"] = True
            self._start_fallback(t, st, "combo_submit_fail")
            return
        st["ticket_id"] = ticket.OrderId
        self.Log(f"CSFFATTEMPT|{st['aid']}|ff={st['ff']:.1f}|"
                 f"lim={lim:.3f}|mid={st['mid']:.3f}|agg={st['agg']:.3f}|"
                 f"step={st['walk_step']}")

    def _combo_poll(self, t, st):
        # ground truth first: fully held -> entered (fill-guard pattern)
        if self._entry_fully_held(st):
            self._mark_entered(t, st, "combo")
            return
        if st.get("combo_rejected"):
            return  # fallback already started via OnOrderEvent/submit path
        age_min = (self.Time - st["start"]).total_seconds() / 60.0
        if age_min >= self.MAX_WAIT_MIN:
            self._cancel_ticket(st)
            self._record_attempt(t, self.Time.date(), st["ff"], "combo",
                                 st["requested_lim"], st["mid"], st["agg"],
                                 (st["e_f_bid"], st["e_f_ask"]),
                                 (st["e_b_bid"], st["e_b_ask"]),
                                 None, "skipped", st["fsym"], st["bsym"],
                                 st["spot_entry"])
            self.Log(f"CSFFSKIP|{st['aid']}|maxwait_30min|"
                     f"lim={st['requested_lim']:.3f}")
            self.state.pop(t, None)
            return
        step = min(int(age_min / self.WALK_STEP_MIN), self.WALK_STEPS)
        if step > st["walk_step"]:
            st["walk_step"] = step
            frac = step / self.WALK_STEPS
            lim = st["mid"] + frac * (st["agg"] - st["mid"])
            self._cancel_ticket(st)
            self._place_combo(t, st, lim)

    def _cancel_ticket(self, st):
        tid = st.get("ticket_id")
        if tid is None:
            return
        try:
            for item in self.Transactions.GetOpenOrders():
                oid = getattr(item, "Id", None)
                if oid == tid:
                    self.Transactions.CancelOrder(tid)
                    break
        except Exception as e:
            self.Log(f"CSFFCANCEL|{st['aid']}|cancel_fail|{e}")
        st["ticket_id"] = None

    def _entry_fully_held(self, st):
        try:
            fq = self.Portfolio[st["fsym"]].Quantity
            bq = self.Portfolio[st["bsym"]].Quantity
        except Exception:
            return False
        return fq == -1 and bq == 1

    def _avg_fill_debit(self, st):
        # Portfolio average prices = ground-truth net debit paid
        try:
            fa = self.Portfolio[st["fsym"]].AveragePrice
            ba = self.Portfolio[st["bsym"]].AveragePrice
        except Exception:
            return None
        if fa == 0 or ba == 0:
            return None
        return float(ba) - float(fa)

    @staticmethod
    def _k_of(fill_px, mid, agg):
        width = agg - mid
        if width < 1e-9:
            return 0.0
        return (fill_px - mid) / width

    def _mark_entered(self, t, st, path):
        fill_px = self._avg_fill_debit(st)
        k = self._k_of(fill_px, st["mid"], st["agg"]) if fill_px else None
        st["phase"] = "held"
        st["fill_px"] = fill_px
        st["path"] = path
        self._record_attempt(t, self.Time.date(), st["ff"], path,
                             st["requested_lim"], st["mid"], st["agg"],
                             (st["e_f_bid"], st["e_f_ask"]),
                             (st["e_b_bid"], st["e_b_ask"]),
                             fill_px, "filled", st["fsym"], st["bsym"],
                             st["spot_entry"])
        self.Log(f"CSFFFILL|{st['aid']}|path={path}|fill={fill_px:.3f}|"
                 f"lim={st['requested_lim']:.3f}|mid={st['mid']:.3f}|"
                 f"agg={st['agg']:.3f}|k={k:.3f}" if fill_px else
                 f"CSFFFILL|{st['aid']}|path={path}|fill=NA")

    # ---------- buy-first fallback ----------

    def _start_fallback(self, t, st, why):
        # BUY back-month FIRST, then sell front-month after 60s stagger.
        # Sell-first is never attempted (Tastytrade naked-short rejection).
        st["phase"] = "fb_buy_placed"
        st["fallback_used"] = True
        st["fb_t0"] = self.Time
        st["fb_why"] = why
        try:
            self.Buy(st["bsym"], 1)
        except Exception as e:
            self.Log(f"CSFFREJECT|{st['aid']}|fallback_buy_fail|{e}")
            self._record_attempt(t, self.Time.date(), st["ff"], "fallback",
                                 st["requested_lim"], st["mid"], st["agg"],
                                 (st["e_f_bid"], st["e_f_ask"]),
                                 (st["e_b_bid"], st["e_b_ask"]),
                                 None, "fallback_buy_fail",
                                 st["fsym"], st["bsym"], st["spot_entry"])
            self.state.pop(t, None)
            return
        self.Log(f"CSFFFALLBACK|{st['aid']}|why={why}|bought_back|"
                 f"lim={st['requested_lim']:.3f}")

    def _fallback_poll(self, t, st):
        try:
            bq = self.Portfolio[st["bsym"]].Quantity
        except Exception:
            bq = 0
        if bq != 1:
            age = (self.Time - st["fb_t0"]).total_seconds() / 60.0
            if age >= self.MAX_WAIT_MIN:
                self._record_attempt(t, self.Time.date(), st["ff"], "fallback",
                                     st["requested_lim"], st["mid"], st["agg"],
                                     (st["e_f_bid"], st["e_f_ask"]),
                                     (st["e_b_bid"], st["e_b_ask"]),
                                     None, "fallback_buy_nofill",
                                     st["fsym"], st["bsym"], st["spot_entry"])
                self.state.pop(t, None)
            return
        if (self.Time - st["fb_t0"]).total_seconds() < self.FALLBACK_STAGGER_SEC:
            return
        try:
            fq = self.Portfolio[st["fsym"]].Quantity
        except Exception:
            fq = 0
        if fq == 0:
            try:
                self.Sell(st["fsym"], 1)
                self.Log(f"CSFFFALLBACK|{st['aid']}|sold_front|stagger_60s")
            except Exception as e:
                self.Log(f"CSFFREJECT|{st['aid']}|fallback_sell_fail|{e}")
                return
        if self._entry_fully_held(st):
            self._mark_entered(t, st, "fallback")

    # ---------- exits: TP40 + front-1d, buy-first sequenced ----------

    def _manage_open(self, t, st):
        if st["phase"] != "held":
            return
        qf = self._sec_quote(st["fsym"])
        qb = self._sec_quote(st["bsym"])
        dte = (datetime.strptime(st["front_exp"], "%Y-%m-%d").date()
               - self.Time.date()).days
        reason = None
        if qf and qb and st.get("fill_px"):
            mark = 0.5 * (qb[0] + qb[1]) - 0.5 * (qf[0] + qf[1])
            if mark >= self.TP_MULT * st["fill_px"]:
                reason = "tp40"
        if reason is None and dte <= self.CLOSE_DTE:
            reason = "time"
        if reason is None:
            return
        # buy-first exit: buy front (short) first, sell back (long) after 60s
        st["phase"] = "exit_working"
        st["exit_reason"] = reason
        st["exit_t0"] = self.Time
        st["exit_stage"] = 0
        self.Log(f"CSFFEXITSTART|{st['aid']}|{reason}|dte={dte}")

    def _exit_poll(self, t, st):
        try:
            fq = self.Portfolio[st["fsym"]].Quantity
            bq = self.Portfolio[st["bsym"]].Quantity
        except Exception:
            fq, bq = -1, 1
        if fq == 0 and bq == 0:
            self._finish_trade(t, st)
            return
        if st["exit_stage"] == 0:
            if fq != 0:
                try:
                    self.Buy(st["fsym"], 1)  # buy back short front first
                except Exception as e:
                    self.Log(f"CSFFREJECT|{st['aid']}|exit_buy_fail|{e}")
                    return
                st["exit_stage"] = 1
                st["exit_t1"] = self.Time
        elif st["exit_stage"] == 1:
            if fq == 0 and bq != 0:
                if (self.Time - st["exit_t1"]).total_seconds() >= self.FALLBACK_STAGGER_SEC:
                    try:
                        self.Sell(st["bsym"], 1)  # then sell long back
                    except Exception as e:
                        self.Log(f"CSFFREJECT|{st['aid']}|exit_sell_fail|{e}")
                        return
                    st["exit_stage"] = 2

    def _finish_trade(self, t, st):
        self.trade_rows.append({
            "aid": st["aid"], "ticker": t,
            "entry_date": st["aid"].split("-")[0],
            "exit_date": str(self.Time.date()),
            "exit_reason": st.get("exit_reason", ""),
            "path": st.get("path", ""),
            "ff": st["ff"], "strike": st["strike"],
            "fill_px": round(st.get("fill_px") or 0, 4),
        })
        self.Log(f"CSFFCLOSE|{st['aid']}|{st.get('exit_reason', '')}")
        self.state.pop(t, None)

    # ---------- order events: rejection accounting ----------

    def OnOrderEvent(self, orderEvent):
        o = orderEvent
        try:
            status = str(o.Status)
        except Exception:
            status = ""
        msg = ""
        try:
            msg = str(getattr(o, "Message", "") or "")
        except Exception:
            pass
        tag = ""
        try:
            tag = str(getattr(o, "Tag", "") or "")
        except Exception:
            pass
        if "CSFF" not in tag and "em=1515" not in tag:
            # still count brokerage rejections on our option symbols
            pass
        low = (status + " " + msg).lower()
        if "invalid" in status.lower() or "buying power" in low or "buyingpower" in low \
                or "insufficient" in low or "margin" in low or "notsupported" in low \
                or "not supported" in low:
            if "buying power" in low or "buyingpower" in low or "insufficient" in low:
                self.bp_rejections += 1
                self.Log(f"CSFFBPREJECT|{tag}|{status}|{msg}")
            else:
                self.other_rejections += 1
                self.Log(f"CSFFORDERREJECT|{tag}|{status}|{msg}")
            # a rejected combo attempt falls back to buy-first (once)
            for t, st in list(self.state.items()):
                if st.get("phase") == "combo_working" and st["aid"] in tag \
                        and not st.get("fallback_used"):
                    try:
                        self._cancel_ticket(st)
                    except Exception:
                        pass
                    st["combo_rejected"] = True
                    self._start_fallback(t, st, f"rejected:{status}:{msg}"[:120])
                    break

    # ---------- attempt record ----------

    def _record_attempt(self, t, date, ff, path, requested_lim, mid, agg,
                        qf, qb, fill_px, outcome, fsym, bsym, spot):
        k = round(self._k_of(fill_px, mid, agg), 4) if fill_px else ""
        self.attempt_rows.append({
            "date": str(date), "ticker": t, "ff": ff, "path": path,
            "requested_lim": round(requested_lim, 4),
            "fill_px": round(fill_px, 4) if fill_px else "",
            "spr": round((qf[1] - qf[0]) + (qb[1] - qb[0]), 4),
            "mid": round(mid, 4), "agg": round(agg, 4),
            "k_realized": k, "outcome": outcome,
            "strike": "", "spot": spot if spot else "",
        })
        self.Log(f"CSFFLOG|{t}|{date}|path={path}|lim={requested_lim:.3f}|"
                 f"fill={fill_px if fill_px else 'NA'}|"
                 f"mid={mid:.3f}|agg={agg:.3f}|k={k}|{outcome}")

    # ---------- assignment + teardown ----------

    def OnAssignmentOrderEvent(self, event):
        for t, st in list(self.state.items()):
            if st.get("phase") == "held" and event.Symbol == st.get("fsym"):
                self.Log(f"CSFFASSIGN|{st['aid']}")
                try:
                    self.Liquidate(st["bsym"])
                except Exception:
                    pass
                try:
                    self.Liquidate(t)
                except Exception:
                    pass
                self.state.pop(t, None)

    def OnEndOfAlgorithm(self):
        for t in list(self.state):
            st = self.state[t]
            if st.get("phase") == "held":
                self._finish_trade(t, st)
            elif st.get("phase") in ("combo_working", "fb_buy_placed",
                                     "exit_working"):
                try:
                    self._cancel_ticket(st)
                except Exception:
                    pass
                if not any(r.get("aid") == st["aid"]
                           for r in self.trade_rows):
                    self._record_attempt(t, self.Time.date(), st["ff"],
                                         st.get("path", "combo") if st.get("path")
                                         else "combo",
                                         st["requested_lim"], st["mid"], st["agg"],
                                         (st["e_f_bid"], st["e_f_ask"]),
                                         (st["e_b_bid"], st["e_b_ask"]),
                                         st.get("fill_px"), "end_of_data",
                                         st["fsym"], st["bsym"], st.get("spot_entry"))
        self.Log(f"CSFFSUMMARY|attempts={len(self.attempt_rows)}|"
                 f"trades={len(self.trade_rows)}|"
                 f"bp_rejections={self.bp_rejections}|"
                 f"other_rejections={self.other_rejections}")
        if self.attempt_rows:
            cols = list(self.attempt_rows[0].keys())
            csv = ",".join(cols) + "\n" + "\n".join(
                ",".join(str(r.get(c, "")) for c in cols)
                for r in self.attempt_rows)
            self.ObjectStore.Save("ff_r52_attempts_live", csv)
        if self.trade_rows:
            cols = list(self.trade_rows[0].keys())
            csv = ",".join(cols) + "\n" + "\n".join(
                ",".join(str(r.get(c, "")) for c in cols)
                for r in self.trade_rows)
            self.ObjectStore.Save("ff_r52_trades_live", csv)
