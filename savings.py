# What is the forecast worth? Simulates a year of EV charging with four strategies, at the real consumer price (spot × 1.25 + add).
# usage: python3 elprisprognos/savings.py   (uses cache/ from run.py backtest)
import datetime as dt, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "custom_components", "elprisprognos"))
import prognos as P
sys.path.insert(0, HERE); import run

KWH_YEAR, CAP, RATE = 6160, 99.8, 6.7           # example car: yearly kWh, battery size, charging power – edit to match yours
E_MIN, E_MAX, TARGET = 0.25 * CAP, 0.9 * CAP, 0.6 * CAP
VAT, ADD = 1.25, 0.71
DAILY = KWH_YEAR / 365                          # ponytail: driving as a steady daily draw at 08:00; real trips are lumpier

def plan(hours, price, e0, use):
    """Cheapest-first charging over `hours` so the battery never drops below E_MIN or exceeds E_MAX. Returns {hour: kWh}."""
    ch = {h: 0.0 for h in hours}
    def ok():
        e = e0
        for h in hours:
            e += ch[h] - use.get(h, 0)
            if e > E_MAX + 1e-6: return False   # only the ceiling here; the floor is what need() keeps buying for
        return True
    def need():   # below the floor at some point, or not back at 60 % at the end of the horizon (keeps the buffer)
        lv = list(_levels(hours, ch, e0, use))
        return any(e < E_MIN - 1e-6 for e in lv) or (lv and lv[-1] < TARGET - 1e-6)
    for h in sorted(hours, key=lambda h: price[h]):
        if not need(): break
        ch[h] = RATE
        if not ok():   # too much for the cap: charge only what fits
            ch[h] = 0.0; lo, hi = 0.0, RATE
            for _ in range(20):
                ch[h] = (lo + hi) / 2
                lo, hi = (ch[h], hi) if ok() else (lo, ch[h])
            ch[h] = lo
    return ch

def _levels(hours, ch, e0, use):
    e = e0
    for h in hours:
        e += ch[h] - use.get(h, 0); yield e

def simulate(px, fcs, strategy, days):
    e, cost, kwh = TARGET, 0.0, 0.0
    for d in days:   # decide at 13:00 local (≈ 12 UTC) on day d; tomorrow's prices are known then
        now = f"{d}T12"
        h0 = dt.datetime.strptime(now, "%Y-%m-%dT%H")
        H = [f"{h0 + dt.timedelta(hours=i):%Y-%m-%dT%H}" for i in range(24 * 7)]
        known = [h for h in H if h in px and h[:10] <= f"{d + dt.timedelta(days=1)}"]
        use = {h: DAILY for h in H if h.endswith("T07")}
        if strategy == "deadline 07:30":   # tonight only: charge back to 60 % between 17:00 and 06:00 UTC
            win = [h for h in known if f"{d}T16" <= h <= f"{d + dt.timedelta(days=1)}T05"]
            ch = {h: 0.0 for h in H}; need = TARGET - e
            for h in sorted(win, key=lambda h: px[h]):
                if need <= 0: break
                ch[h] = min(RATE, need); need -= ch[h]
        elif strategy == "utan deadline, kända priser":   # keep ≥ E_MIN over what we can see, top up to 60 % cheapest within it
            ch = plan(known, px, e, use)
            extra = TARGET - (e + sum(ch.values()) - sum(use.get(h, 0) for h in known))
            for h in sorted(known, key=lambda h: px[h]):
                if extra <= 0: break
                add = min(RATE - ch[h], extra); ch[h] += add; extra -= add
        else:   # with a 7-day horizon: actual where known, forecast (or perfect foresight) for the rest
            fc = fcs.get(d, {})
            pr = {h: px[h] if h in known else (fc.get(h) if strategy == "med prognos" else px.get(h)) for h in H}
            hs = [h for h in H if pr.get(h) is not None]
            ch = plan(hs, pr, e, use)
        for h in H[:24]:   # execute the first 24 h, then re-plan
            if h in px:
                q = ch.get(h, 0.0); cost += q * (px[h] * VAT + ADD); kwh += q; e += q
            e -= use.get(h, 0)
    return cost, kwh, e

def forecasts(px, wx, days):
    """What the model would have said each day (refit weekly), for days d+2 … d+7."""
    fc, model = {}, None
    for i, d in enumerate(days):
        known = f"{d + dt.timedelta(days=1)}T23"
        tr = {h: v for h, v in px.items() if h <= known}
        if i % 7 == 0: model = P.fit(tr, wx)
        hrs = [h for h in wx if f"{d + dt.timedelta(days=2)}" <= h[:10] <= f"{d + dt.timedelta(days=7)}"]
        for h, v in P.predict(model, hrs, wx, P._level(tr, known)).items(): fc.setdefault(d, {})[h] = v
    return fc

if __name__ == "__main__":
    px, wx = run.data(dt.date.today() - dt.timedelta(days=1))
    end = dt.date.today() - dt.timedelta(days=9)   # leave a week of real prices after the last plan
    days = [end - dt.timedelta(days=i) for i in range(365, 0, -1)]
    fcf = os.path.join(run.CACHE, f"savings_fc_{days[0]}.json")   # the weekly refits take minutes: reuse them
    if os.path.exists(fcf): fcs = {dt.date.fromisoformat(k): v for k, v in json.load(open(fcf)).items()}
    else: fcs = forecasts(px, wx, days); json.dump({str(k): v for k, v in fcs.items()}, open(fcf, "w"))
    base = None
    print(f"EV-laddning {days[0]} – {days[-1]}, {KWH_YEAR} kWh/år, {RATE} kW, batteri {CAP} kWh (25–90 %):")
    for s in ("deadline 07:30", "utan deadline, kända priser", "med prognos", "perfekt framförhållning"):
        cost, kwh, e = simulate(px, fcs, s, days)
        cost += (TARGET - e) * 1.2   # settle a different end level at ~1.20 kr/kWh so strategies compare fairly
        base = base or cost
        print(f"  {s:30} {cost:7.0f} kr  {100 * cost / KWH_YEAR:4.0f} öre/kWh  sparar {base - cost:6.0f} kr/år")
