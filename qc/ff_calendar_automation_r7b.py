# region imports
from AlgorithmImports import *
import math
from datetime import datetime
# endregion

"""
R7b full-19 frozen-model scoring in-QC (pre-registered 2026-09-14,
RESEARCH_PLAN.md Test R7b -- the true R6.2; row 9's 9/19 amputated offline
score is RETRACTED as a model verdict, bias unknown).

Base: COPIED from qc/ff_calendar_automation_r61.py (base untouched). Same
locked-20 x same entry worker (mid-pegged debit ComboLimitOrder, 3x10min
walk, 30min skip, buy-first fallback, TP40 + front-1d, regular-hours Limit
assignment flatten, Tastytrade model, $2.60 k-spectrum post) plus in-algo
frozen put-ML scoring at every FF>20 entry signal:

  At signal time (_try_enter, all quote/expiry locals present) the algo
  builds the FULL 19-feature vector per ff_ml_v2.py build_features /
  build_features_put put-native semantics (entry_debit = put mid,
  back_straddle = back PUT mid, max_profit = back-put-mid*sqrt(t_fwd/t_back)
  - put mid, t_* integer days, month/quarter from entry date, is_etf from
  the shipped ETF set) and scores it with the FROZEN scaler+coefs from
  ff_ml_lr_coefs_put_v1.json (embedded verbatim below -- NO refit, NO
  training, NO online update). The 19 model inputs + ml_score_put are logged
  on a CSFFFEAT line keyed by aid for every recorded attempt (filled,
  skipped, badspread, fallback). NO offline imputation, NO reconstruction:
  feature computation is wrapped so any failure logs CSFFFEATGAP instead of
  a score, and the harvest ABORTS the ranking verdict (reports the gap) if
  any recorded attempt lacks a verifiable FEAT line. Offline harvest
  RECOMPUTES the score from the logged 19 inputs with the frozen JSON and
  asserts agreement (verification, not refit).

Frozen bar: rescue iff Q5 EV(k=0.25)>=0 AND Q5-Q1 spread >20pp AND 95% CI on
spread excludes 0. Else the ranking lever is KILLED on valid evidence
(replaces row 9's verdict).
"""


class FFCalendarAutomationR7B(QCAlgorithm):

    # ---- R5.1 locked universe: 20 stocks, 0 ETFs (median >=1 FF>20/mo) ----
    LOCKED20 = ["SQ", "DIS", "ADBE", "SHOP", "PYPL", "NFLX", "CRM", "NKE",
                "ORCL", "UBER", "PLTR", "PFE", "WMT", "SMCI", "T", "NVDA",
                "LLY", "UAL", "UNH", "MRNA"]
    TICKERS = list(LOCKED20)

    FF_MIN = 20.0
    FRONT_LO, FRONT_HI, FRONT_TGT = 20, 40, 30
    BACK_LO, BACK_HI, BACK_TGT = 45, 75, 60
    TP_MULT = 1.40
    CLOSE_DTE = 1
    RISK_FREE = 0.04

    # ---- Frozen put-ML (ff_ml_lr_coefs_put_v1.json, embedded verbatim) ----
    # NO refit: these numbers are the shipped scaler+LR. Feature order =
    # scaler_mean key order. Semantics per ff_ml_v2.py:43-99 put-native
    # (build_features_put halves straddle-level debit/straddle, so put-mid
    # inputs below are exactly what the put scaler was fit on).
    ML_FEATS = ["entry_ff", "entry_ff_sq", "entry_ff_pos", "entry_debit",
                "log_debit", "back_straddle", "log_back_straddle",
                "debit_ratio", "t_front", "t_back", "t_fwd", "t_fwd_ratio",
                "max_profit_ratio", "iv_proxy", "ff_x_tfwd", "month_sin",
                "month_cos", "quarter", "is_etf"]
    ML_INTERCEPT = -0.5460295880190413
    ML_COEFS = {"t_fwd_ratio": 7.492245, "t_fwd": -4.182649,
                "t_front": 4.036069, "ff_x_tfwd": -1.886666,
                "t_back": -1.739271, "entry_ff": 1.424481,
                "max_profit_ratio": 0.716782, "debit_ratio": -0.596453,
                "log_debit": 0.511189, "back_straddle": -0.443657,
                "log_back_straddle": -0.424555, "iv_proxy": -0.342965,
                "entry_debit": 0.306796, "entry_ff_pos": 0.304351,
                "quarter": -0.127769, "entry_ff_sq": 0.113383,
                "month_sin": -0.076028, "is_etf": 0.053588,
                "month_cos": -0.010422}
    ML_MEAN = {"entry_ff": 2.5968715546983465, "entry_ff_sq": 764.323314499096,
               "entry_ff_pos": 10.340693854697514,
               "entry_debit": 2.879636390466135,
               "log_debit": 1.0835288048941303,
               "back_straddle": 10.283017948739264,
               "log_back_straddle": 2.0275322619986125,
               "debit_ratio": 0.28045726681890293, "t_front": 28.8097716295194,
               "t_back": 55.836998179354374, "t_fwd": 27.027226549834978,
               "t_fwd_ratio": 0.48051569368108066,
               "max_profit_ratio": 3.4543267754623512,
               "iv_proxy": 0.12019634391458672,
               "ff_x_tfwd": 13.036958171935616,
               "month_sin": -0.04569074403555305,
               "month_cos": 0.033883611364095095,
               "quarter": 2.596440180568141, "is_etf": 0.08887078404150074}
    ML_STD = {"entry_ff": 27.52416343192822, "entry_ff_sq": 3001.72307205228,
              "entry_ff_pos": 21.902636193726696,
              "entry_debit": 4.570216585923975,
              "log_debit": 0.6397342855359517,
              "back_straddle": 16.22112176213881,
              "log_back_straddle": 0.792839859534931,
              "debit_ratio": 0.08327133033883657, "t_front": 2.9834691525663044,
              "t_back": 6.114446765311267, "t_fwd": 5.421485178671507,
              "t_fwd_ratio": 0.05749418200991853,
              "max_profit_ratio": 2.166201014727535,
              "iv_proxy": 0.04468528005554621,
              "ff_x_tfwd": 140.83500336628748,
              "month_sin": 0.7028519598952577,
              "month_cos": 0.709058093009169,
              "quarter": 1.1255347466707688, "is_etf": 0.28455714326888665}
    # ETF set copied from ff_ml_v2.py:36-39 (is_etf input).
    ML_ETF = {"SPY", "DIA", "MDY", "XLK", "XLF", "XLV", "XLE", "XLI", "XLU",
              "XLY", "XLP", "XLB", "XLC", "XLRE", "XBI", "XRT", "XHB", "XME",
              "XOP", "KRE", "XSD", "XAR"}

    WALK_STEP_MIN = 10.0
    WALK_STEPS = 3
    MAX_WAIT_MIN = 30.0
    FALLBACK_STAGGER_SEC = 60.0

    def Initialize(self):
        # R6.1 window params: QC cloud keeps only ~100KB of backtest log per
        # run, and the full 2024-01->2026-09 window overflows it -- so the
        # window runs as ENTRY-PARTITIONED quarterly runs: entries are gated
        # to [start, entry_end], while the run continues to `end`
        # (= entry_end + 45d tail) so every gated entry can close inside its
        # own run (max hold ~40d via front-1d time exit). No aid ever appears
        # in two runs; harvest concatenates without dedupe. Worker economics
        # are window-invariant; only the date slice changes per push.
        self.start_str = self.GetParameter("start", "2024-01-02")
        self.entry_end_str = self.GetParameter("entry_end", "2024-03-31")
        self.end_str = self.GetParameter("end", "2024-05-15")
        self.SetStartDate(*map(int, self.start_str.split("-")))
        self.entry_end = datetime.strptime(self.entry_end_str, "%Y-%m-%d").date()
        self.SetEndDate(*map(int, self.end_str.split("-")))
        self.SetCash(100_000)
        self.SetTimeZone("America/New_York")
        self.SetBrokerageModel(BrokerageName.Tastytrade, AccountType.Margin)
        # sharding: 20 names x minute options x 2.7y is too heavy for one
        # backtest (R5.1 used 4 shards) -- params shard/nshards partition
        # LOCKED20 round-robin; ObjectStore keys carry the shard suffix.
        self.shard = int(self.GetParameter("shard", "0"))
        self.nshards = max(1, int(self.GetParameter("nshards", "1")))
        self.TICKERS = [t for i, t in enumerate(self.LOCKED20)
                        if i % self.nshards == self.shard]
        self.Log(f"CSFFSHARD|shard={self.shard}/{self.nshards}|"
                 f"names={','.join(self.TICKERS)}|"
                 f"start={self.start_str}|entry_end={self.entry_end_str}|"
                 f"end={self.end_str}|"
                 f"diag={self.GetParameter('diag_scan', '0')}|"
                 f"verbose={self.GetParameter('verbose', '0')}")
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
        self.diag = self.GetParameter("diag_scan", "0") in (1, "1", "true", "True")
        # verbose=1 restores per-step text lines (ATTEMPT/FILL/SKIP/EXITSTART)
        # for mechanics debugging; default 0 keeps only the harvest contract
        # (CSFFLOG filled + CSFFCLOSE + ASSIGN + SUMMARY) inside the cloud
        # ~100KB per-backtest log cap.
        self.verbose = self.GetParameter("verbose", "0") in (1, "1", "true", "True")
        self.n_skipped = 0
        self.n_badspread = 0
        self.n_fallback = 0
        self.n_assigned = 0
        self.skip_by_ticker = {}
        # EveryDay needs a SUBSCRIBED symbol for the exchange calendar --
        # R52 used "SPY" but SPY is not in the locked 20, so use the shard's
        # first name (same US-equity hours for all 20).
        self.Schedule.On(self.DateRules.EveryDay(self.TICKERS[0]),
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
            # entry partition: past entry_end the run only manages/exits
            if self.Time.date() > self.entry_end:
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
            elif phase == "assign_pending":
                self._assign_poll(t, st)

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
        # R7b: full-19 frozen scoring at signal time (all locals present).
        # Any feature not computable -> GAP line, never imputed (see _ml_feat_str).
        t_front_d = (f.Expiry.date() - today).days
        t_back_d = (b.Expiry.date() - today).days
        feat = self._ml_feat_str(t, today, ff, mid, qf, qb, float(f.Strike),
                                 t_front_d, t_back_d)
        if agg <= mid:
            # locked/crossed ladder: never cross -> immediate skip, logged
            self._record_attempt(t, today, ff, "combo", mid, mid, agg,
                                 qf, qb, None, "skip_badspread", f, b, spot,
                                 feat)
            return
        aid = f"{today:%Y%m%d}-{t}"
        st = {
            "phase": "combo_working", "ticker": t, "aid": aid,
            "ff": round(ff, 2), "strike": float(f.Strike),
            "front_exp": str(f.Expiry.date()), "back_exp": str(b.Expiry.date()),
            "spot_entry": round(spot, 2),
            "feat": feat,
            "e_f_bid": qf[0], "e_f_ask": qf[1],
            "e_b_bid": qb[0], "e_b_ask": qb[1],
            "mid": mid, "agg": agg, "spr": (qf[1] - qf[0]) + (qb[1] - qb[0]),
            "fsym": f.Symbol, "bsym": b.Symbol,
            "start": self.Time, "walk_step": 0, "ticket_ids": [],
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
            tickets = self.ComboLimitOrder(self._combo_legs(st), 1, lim,
                                           tag=self._entry_tag(st, lim, "combo"))
        except Exception as e:
            self.Log(f"CSFFREJECT|{st['aid']}|combo_submit_fail|{e}")
            st["combo_rejected"] = True
            self._start_fallback(t, st, "combo_submit_fail")
            return
        # LEAN returns a LIST of per-leg tickets for combo orders (found
        # 2026-09-11: YTD run crashed on ticket.OrderId at the first FF>20
        # firing). Track every leg id for the scoped cancel below.
        if not isinstance(tickets, list):
            tickets = [tickets]
        st["ticket_ids"] = [tk.OrderId for tk in tickets]
        if self.verbose:
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
                                 st["spot_entry"], st.get("feat") or "")
            if self.verbose:
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
        tids = st.get("ticket_ids") or ([st["ticket_id"]] if st.get("ticket_id") else [])
        if not tids:
            return
        try:
            open_ids = {getattr(item, "Id", None)
                        for item in self.Transactions.GetOpenOrders()}
            for tid in tids:
                if tid in open_ids:
                    self.Transactions.CancelOrder(tid)
        except Exception as e:
            self.Log(f"CSFFCANCEL|{st['aid']}|cancel_fail|{e}")
        st["ticket_ids"] = []

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

    @classmethod
    def _ml_inputs(cls, ff, mid, qf, qb, strike, t_front_d, t_back_d,
                   entry_date, ticker):
        """FULL 19-feature vector, put-native, mirroring ff_ml_v2.py:43-99
        exactly (clip ranges, log1p, ratios). Returns dict in ML_FEATS order.
        Raises on any failure (caller logs CSFFFEATGAP, never imputes)."""
        eff = min(max(ff, -100.0), 200.0)
        debit = min(max(mid, 0.01), 50.0)
        bmid = 0.5 * (qb[0] + qb[1])
        bstrad = min(max(bmid, 0.01), 200.0)
        t_fwd_d = t_back_d - t_front_d
        f = {}
        f["entry_ff"] = eff
        f["entry_ff_sq"] = eff * eff
        f["entry_ff_pos"] = max(eff, 0.0)
        f["entry_debit"] = debit
        f["log_debit"] = math.log1p(debit)
        f["back_straddle"] = bstrad
        f["log_back_straddle"] = math.log1p(bstrad)
        f["debit_ratio"] = min(max(mid / max(bmid, 0.01), 0.0), 1.0)
        f["t_front"] = float(t_front_d)
        f["t_back"] = float(t_back_d)
        f["t_fwd"] = float(t_fwd_d)
        f["t_fwd_ratio"] = min(max(t_fwd_d / max(t_back_d, 1), 0.0), 1.0)
        # max_profit put-model semantics (ff_ml_v2.py:66-67 on the HALVED
        # frame): build_features_put halves entry_debit/back_straddle but NOT
        # max_profit, so the put ratio runs 2x the straddle ratio. Reproduce
        # exactly: straddle-level max_theo over put-level debit.
        max_theo = 2.0 * bmid * math.sqrt(t_fwd_d / t_back_d) if t_back_d > 0 else 0.0
        f["max_profit_ratio"] = min(max((max_theo - 2.0 * mid) / max(mid, 0.01),
                                        -2.0), 20.0)
        t_back_yr = max(t_back_d / 365.0, 0.001)
        f["iv_proxy"] = min(max(bmid / (max(strike, 1.0)
                                        * math.sqrt(t_back_yr)), 0.0), 2.0)
        f["ff_x_tfwd"] = eff * math.sqrt(max(t_fwd_d, 1))
        m = entry_date.month
        f["month_sin"] = math.sin(2.0 * math.pi * m / 12.0)
        f["month_cos"] = math.cos(2.0 * math.pi * m / 12.0)
        f["quarter"] = float((m - 1) // 3 + 1)
        f["is_etf"] = 1.0 if ticker in cls.ML_ETF else 0.0
        return f

    @classmethod
    def _ml_score(cls, f):
        """Frozen LR: z=(x-mean)/std, linear=intercept+coefs.z, sigmoid."""
        lin = cls.ML_INTERCEPT
        for name in cls.ML_FEATS:
            lin += cls.ML_COEFS[name] * (f[name] - cls.ML_MEAN[name]) / cls.ML_STD[name]
        return 1.0 / (1.0 + math.exp(-lin))

    def _ml_feat_str(self, t, today, ff, mid, qf, qb, strike,
                     t_front_d, t_back_d):
        """ aid-keyed payload: 19 model inputs (6dp) + score. On ANY failure
        returns None (caller logs CSFFFEATGAP with the reason)."""
        try:
            f = self._ml_inputs(ff, mid, qf, qb, strike, t_front_d,
                                t_back_d, today, t)
            s = self._ml_score(f)
            vals = "|".join(f"{f[n]:.6f}" for n in self.ML_FEATS)
            return f"{vals}|score={s:.6f}"
        except Exception as e:
            aid = f"{today:%Y%m%d}-{t}"
            self.Log(f"CSFFFEATGAP|{aid}|{type(e).__name__}:{(str(e))[:100]}")
            return None

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
                             st["spot_entry"], st.get("feat") or "")
        if self.verbose:
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
                                 st["fsym"], st["bsym"], st["spot_entry"],
                                 st.get("feat") or "")
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
                                      st["fsym"], st["bsym"], st["spot_entry"],
                                      st.get("feat") or "")
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
        # R6.1 prereq (b) spirit: never initiate sequenced exits outside
        # regular hours (after-hours option orders invite MOO-type rejects).
        sec = self.Securities.get(t)
        if sec is not None and not sec.Exchange.ExchangeOpen:
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
        if self.verbose:
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
        # quote-complete row: exit quotes captured at flatten time so the
        # trade is harvestable from logs alone (ObjectStore unreadable via
        # API, gotcha 25). Missing quotes -> "" (row flagged, post-analysis
        # drops it like the R1b harness drops quoteless rows).
        qf = self._sec_quote(st["fsym"])
        qb = self._sec_quote(st["bsym"])
        st["x_f_bid"] = qf[0] if qf else ""
        st["x_f_ask"] = qf[1] if qf else ""
        st["x_b_bid"] = qb[0] if qb else ""
        st["x_b_ask"] = qb[1] if qb else ""
        self.trade_rows.append({
            "aid": st["aid"], "ticker": t,
            "entry_date": st["aid"].split("-")[0],
            "exit_date": str(self.Time.date()),
            "exit_reason": st.get("exit_reason", ""),
            "path": st.get("path", ""),
            "ff": st["ff"], "strike": st["strike"],
            "fill_px": round(st.get("fill_px") or 0, 4),
            "mid_debit": round(st.get("fill_px") or 0, 4),
            "k_realized": round(self._k_of(st.get("fill_px"), st["mid"],
                                           st["agg"]), 4)
            if st.get("fill_px") else "",
            "e_f_bid": st["e_f_bid"], "e_f_ask": st["e_f_ask"],
            "e_b_bid": st["e_b_bid"], "e_b_ask": st["e_b_ask"],
            "x_f_bid": st["x_f_bid"], "x_f_ask": st["x_f_ask"],
            "x_b_bid": st["x_b_bid"], "x_b_ask": st["x_b_ask"],
            "width_entry": round(st["agg"] - st["mid"], 4),
            "spot_entry": st.get("spot_entry", ""),
        })
        self.Log(f"CSFFCLOSE|{st['aid']}|{st.get('exit_reason', '')}|"
                 f"fill={st.get('fill_px') or 'NA'}|path={st.get('path', '')}|"
                 f"xfb={st['x_f_bid']}|xfa={st['x_f_ask']}|"
                 f"xbb={st['x_b_bid']}|xba={st['x_b_ask']}")
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
                        qf, qb, fill_px, outcome, fsym, bsym, spot,
                        feat=""):
        k = round(self._k_of(fill_px, mid, agg), 4) if fill_px else ""
        try:
            strike_px = float(fsym.ID.StrikePrice)
        except Exception:
            strike_px = ""
        spr = round((qf[1] - qf[0]) + (qb[1] - qb[0]), 4)
        self.attempt_rows.append({
            "date": str(date), "ticker": t, "ff": ff, "path": path,
            "requested_lim": round(requested_lim, 4),
            "fill_px": round(fill_px, 4) if fill_px else "",
            "spr": spr,
            "mid": round(mid, 4), "agg": round(agg, 4),
            "k_realized": k, "outcome": outcome,
            "strike": strike_px, "spot": spot if spot else "",
        })
        # Slim harvest contract: FULL quote text lines only for FILLED
        # attempts (the k-spectrum inputs); skips aggregate into summary
        # counters + per-ticker SKIPTOT lines (cloud ~100KB log cap).
        # R7b: every recorded attempt also carries its frozen-score FEAT line
        # (aid-keyed; ~200 chars) -- ranking inputs without touching the cap.
        if feat:
            aid = f"{str(date)[:10].replace('-', '')}-{t}"
            self.Log(f"CSFFFEAT|{aid}|{feat}")
        if outcome == "filled":
            self.Log(f"CSFFLOG|{t}|{date}|path={path}|lim={requested_lim:.2f}|"
                     f"fill={fill_px:.2f}|"
                     f"spr={spr:.2f}|"
                     f"mid={mid:.2f}|agg={agg:.2f}|k={k}|ff={ff:.1f}|"
                     f"efb={qf[0]:.2f}|efa={qf[1]:.2f}|"
                     f"ebb={qb[0]:.2f}|eba={qb[1]:.2f}|"
                     f"K={strike_px}|spot={spot if spot else ''}")
            return
        if outcome == "skipped":
            self.n_skipped += 1
            self.skip_by_ticker[t] = self.skip_by_ticker.get(t, 0) + 1
            return
        if outcome == "skip_badspread":
            self.n_badspread += 1
        if outcome.startswith("fallback"):
            self.n_fallback += 1
        self.Log(f"CSFFLOG|{t}|{date}|path={path}|lim={requested_lim:.2f}|"
                 f"fill=NA|spr={spr:.2f}|mid={mid:.2f}|agg={agg:.2f}|"
                 f"k=|{outcome}|ff={ff:.1f}|K={strike_px}")

    # ---------- assignment + teardown ----------
    # R6.1 prereq (b): assignment flatten via REGULAR-HOURS Limit orders,
    # buys first -- never after-hours Liquidate. R5.2 fired Liquidate at
    # 00:00, LEAN converted to MarketOnOpen, Tastytrade rejected it
    # (MarketOnOpen unsupported). The assigned state parks in phase
    # "assign_pending"; _assign_poll flattens at the next regular session.

    def OnAssignmentOrderEvent(self, event):
        for t, st in list(self.state.items()):
            if st.get("phase") == "held" and event.Symbol == st.get("fsym"):
                st["phase"] = "assign_pending"
                st["assign_t0"] = self.Time
                st["exit_reason"] = "assigned"
                self.n_assigned += 1
                self.Log(f"CSFFASSIGN|{st['aid']}|deferred_limit_flatten")

    def _open_order_symbols(self):
        try:
            return {o.Symbol for o in self.Transactions.GetOpenOrders()
                    if getattr(o, "Symbol", None) is not None}
        except Exception:
            return set()

    def _assign_poll(self, t, st):
        sec = self.Securities.get(t)
        if sec is None or not sec.Exchange.ExchangeOpen:
            return  # regular-hours only: no after-hours Liquidate/MOO
        try:
            fq = self.Portfolio[st["fsym"]].Quantity
        except Exception:
            fq = 0
        try:
            bq = self.Portfolio[st["bsym"]].Quantity
        except Exception:
            bq = 0
        try:
            eq = self.Portfolio[t].Quantity
        except Exception:
            eq = 0
        if fq == 0 and bq == 0 and eq == 0:
            self._finish_trade(t, st)  # exit_reason="assigned", x_* captured
            return
        busy = self._open_order_symbols()
        # step 1: BUYS first -- cover any short option quantity at the ask.
        # Orders carry the aid tag so any Tastytrade rejection is attributable
        # (R6.1 first full run: 2 empty-tag MOC rejects, unattributed).
        if fq < 0 and st["fsym"] not in busy:
            q = self._sec_quote(st["fsym"])
            if q is None:
                return
            try:
                self.LimitOrder(st["fsym"], -fq, q[1],
                                tag=f"CSFFASSIGNFLAT|{st['aid']}|buy_short")
                self.Log(f"CSFFASSIGNFLAT|{st['aid']}|buy_short|qty={-fq}|"
                         f"lim={q[1]:.3f}")
            except Exception as e:
                self.Log(f"CSFFREJECT|{st['aid']}|assign_buy_fail|{e}")
            return
        if fq != 0:
            return  # wait for the cover to fill before selling anything
        # step 2: sell longs + assignment equity at the bid.
        if bq > 0 and st["bsym"] not in busy:
            q = self._sec_quote(st["bsym"])
            if q is None:
                return
            try:
                self.LimitOrder(st["bsym"], -bq, q[0],
                                tag=f"CSFFASSIGNFLAT|{st['aid']}|sell_long")
                self.Log(f"CSFFASSIGNFLAT|{st['aid']}|sell_long|qty={bq}|"
                         f"lim={q[0]:.3f}")
            except Exception as e:
                self.Log(f"CSFFREJECT|{st['aid']}|assign_sell_fail|{e}")
            return
        if eq != 0 and t not in busy:
            b = sec.BidPrice
            a = sec.AskPrice
            if b is None or a is None or b <= 0 or a <= 0 or a < b:
                return
            lim = float(b) if eq > 0 else float(a)
            try:
                self.LimitOrder(t, -eq, lim,
                                tag=f"CSFFASSIGNFLAT|{st['aid']}|flat_equity")
                self.Log(f"CSFFASSIGNFLAT|{st['aid']}|flat_equity|qty={eq}|"
                         f"lim={lim:.3f}")
            except Exception as e:
                self.Log(f"CSFFREJECT|{st['aid']}|assign_equity_fail|{e}")

    def OnEndOfAlgorithm(self):
        for t in list(self.state):
            st = self.state[t]
            if st.get("phase") in ("held", "assign_pending"):
                # assign_pending at EOD: position marked at last quotes with
                # exit_reason "assigned_eod" (flatten may not have completed)
                if st.get("phase") == "assign_pending" \
                        and st.get("exit_reason") == "assigned":
                    st["exit_reason"] = "assigned_eod"
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
                                         st["fsym"], st["bsym"], st.get("spot_entry"),
                                         st.get("feat") or "")
        self.Log(f"CSFFSUMMARY|attempts={len(self.attempt_rows)}|"
                 f"trades={len(self.trade_rows)}|"
                 f"bp_rejections={self.bp_rejections}|"
                 f"other_rejections={self.other_rejections}|"
                 f"skipped={self.n_skipped}|badspread={self.n_badspread}|"
                 f"fallback={self.n_fallback}|assigned={self.n_assigned}")
        for t, n in sorted(self.skip_by_ticker.items()):
            self.Log(f"CSFFSKIPTOT|{t}|skips={n}")
        # gotcha 25: ObjectStore is org-level -- shard-suffixed keys so the 4
        # shard backtests never overwrite each other; harvest merges by key.
        suffix = f"s{self.shard}of{self.nshards}"
        if self.attempt_rows:
            cols = list(self.attempt_rows[0].keys())
            csv = ",".join(cols) + "\n" + "\n".join(
                ",".join(str(r.get(c, "")) for c in cols)
                for r in self.attempt_rows)
            self.ObjectStore.Save(f"ff_r7b_attempts_{suffix}", csv)
        if self.trade_rows:
            cols = list(self.trade_rows[0].keys())
            csv = ",".join(cols) + "\n" + "\n".join(
                ",".join(str(r.get(c, "")) for c in cols)
                for r in self.trade_rows)
            self.ObjectStore.Save(f"ff_r7b_trades_{suffix}", csv)
