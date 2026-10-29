# What is the forecast worth? Simulates a year of EV charging with six strategies, at the real consumer price (spot × 1.25 + add).
# usage: python3 elprisprognos/savings.py   (uses cache/ from run.py backtest)
import datetime as dt, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "custom_components", "elprisprognos"))
import prognos as P
sys.path.insert(0, HERE); import run

KWH_YEAR, CAP, RATE = 6160, 99.8, 6.7           # example car: yearly kWh, battery size, charging power – edit to match yours
E_MIN, E_MAX, TARGET = 0.25 * CAP, 0.9 * CAP, 0.6 * CAP
VAT, ADD = 1.25, 0.71
LOW = 1e9
AWAY = 1e3                                       # price that marks an hour the car is not plugged in
DAILY = KWH_YEAR / 365                          # ponytail: driving as a steady daily draw at 08:00; real trips are lumpier

def plan(hours, price, e0, use):
    """Charging over `hours` so the battery stays ≥ E_MIN, ≤ E_MAX and ends at TARGET: fix the earliest shortfall with the
    cheapest earlier hour that still has room under the cap. Returns {hour: kWh}."""
    ch = {h: 0.0 for h in hours}
    while hours:
        lv = list(_levels(hours, ch, e0, use))
        bad = next((i for i, e in enumerate(lv) if e < E_MIN - 1e-6), None)
        if bad is not None: short = E_MIN - lv[bad]
        elif lv[-1] < TARGET - 1e-6: bad, short = len(lv) - 1, TARGET - lv[-1]
        else: break
        room = lambda i: E_MAX - max(lv[i:])
        cand = [i for i in range(bad + 1) if ch[hours[i]] < RATE - 1e-6 and room(i) > 1e-6]
        if not cand: break   # nothing left that helps: accept the shortfall
        i = min(cand, key=lambda i: price[hours[i]])
        ch[hours[i]] += min(RATE - ch[hours[i]], short, room(i))
    return ch

def _levels(hours, ch, e0, use):
    e = e0
    for h in hours:
        e += ch[h] - use.get(h, 0); yield e

RESERVE = 25   # % the rule wants left when the cheaper day comes (plus a 10 % margin)

def rule_limit(soc, known, fc, d, trip=60, daily=None):
    """Rule for a charge-limit automation (spot prices): charge enough now to last until the first forecast day
    whose 4 cheapest hours are ≥ 20 % below the 4 cheapest known ones (RESERVE + 10 % margin), never below the trip level, max 90 %."""
    daily = daily or 100 * DAILY / CAP
    cheap = lambda v: sum(sorted(v)[:4]) / min(4, len(v))
    days = {}
    for h, v in fc.items(): days.setdefault(h[:10], []).append(v)
    if len(known) < 4 or not days: return trip
    k = cheap(known)
    n = next((n for n, v in sorted(((dt.date.fromisoformat(x) - d).days, cheap(v)) for x, v in days.items()) if n <= 6 and v < 0.8 * k), 7)
    return max(trip, min(90, math.ceil((RESERVE + (n - 1) * daily + 10) / 10) * 10))

def simulate(px, fcs, strategy, days, real=None):
    """`px` drives the decisions (hours the car is away can be priced out), `real` (default px) is what gets paid."""
    real = real or px
    e, cost, kwh = TARGET, 0.0, 0.0
    for d in days:   # decide at 13:00 local (≈ 12 UTC) on day d; tomorrow's prices are known then
        now = f"{d}T12"
        h0 = dt.datetime.strptime(now, "%Y-%m-%dT%H")
        H = [f"{h0 + dt.timedelta(hours=i):%Y-%m-%dT%H}" for i in range(24 * 7)]
        known = [h for h in H if h in px and h[:10] <= f"{d + dt.timedelta(days=1)}"]
        use = {h: DAILY for h in H if h.endswith("T07")}
        if strategy == "dum (direkt 17:00)":   # plug in at ~17:00 local, charge at full power straight back to 60 %
            ch = {h: 0.0 for h in H}; need = TARGET - e
            for h in [h for h in known if h >= f"{d}T16" and px[h] < AWAY]:
                if need <= 0: break
                ch[h] = min(RATE, need); need -= ch[h]
        elif strategy == "deadline 07:30":   # tonight only: charge back to 60 % between 17:00 and 06:00 UTC
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
        elif strategy == "regel (automationen)":   # what the EV9 automation does: pick a charge limit, then cheapest known hours
            kn = [h for h in known if h >= now]
            if e < 0.35 * CAP: kn = [h for h in kn if h <= f"{d + dt.timedelta(days=1)}T05"]   # < 35 %: 07:30 deadline, no deferring
            ch = {h: 0.0 for h in H}; need = rule_limit(100 * e / CAP, [px[h] for h in kn], fcs.get(d, {}), d) / 100 * CAP - e
            for h in sorted(kn, key=lambda h: px[h]):
                if need <= 0: break
                ch[h] = min(RATE, need); need -= ch[h]
        else:   # with a 7-day horizon: actual where known, forecast (or perfect foresight) for the rest
            fc = fcs.get(d, {})
            pr = {h: px[h] if h in known else (fc.get(h) if strategy == "med prognos" else px.get(h)) for h in H}
            hs = [h for h in H if pr.get(h) is not None]
            ch = plan(hs, pr, e, use)
        for h in H[:24]:   # execute the first 24 h, then re-plan
            if h in px:
                q = ch.get(h, 0.0); cost += q * (real[h] * VAT + ADD); kwh += q; e += q
            e -= use.get(h, 0); global LOW; LOW = min(LOW, e)
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
    px, wx, _ = run.data(dt.date.today() - dt.timedelta(days=1))
    end = dt.date.today() - dt.timedelta(days=9)   # leave a week of real prices after the last plan
    days = [end - dt.timedelta(days=i) for i in range(365, 0, -1)]
    fcf = os.path.join(run.CACHE, f"savings_fc_{days[0]}.json")   # the weekly refits take minutes: reuse them
    if os.path.exists(fcf): fcs = {dt.date.fromisoformat(k): v for k, v in json.load(open(fcf)).items()}
    else: fcs = forecasts(px, wx, days); json.dump({str(k): v for k, v in fcs.items()}, open(fcf, "w"))
    base = None
    print(f"EV-laddning {days[0]} – {days[-1]}, {KWH_YEAR} kWh/år, {RATE} kW, batteri {CAP} kWh (25–90 %):")
    for s in ("dum (direkt 17:00)", "deadline 07:30", "utan deadline, kända priser", "regel (automationen)", "med prognos", "perfekt framförhållning"):
        LOW = 1e9
        cost, kwh, e = simulate(px, fcs, s, days)
        cost += (TARGET - e) * 1.2   # settle a different end level at ~1.20 kr/kWh so strategies compare fairly
        base = base or cost
        print(f"  {s:30} {cost:7.0f} kr  {100 * cost / KWH_YEAR:4.0f} öre/kWh  sparar {base - cost:6.0f} kr/år  lägst {100 * LOW / CAP:3.0f} %")
