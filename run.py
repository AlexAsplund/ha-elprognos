# usage: python3 run.py backtest        – how good is it (days 2–7 ahead): v2 vs. the ridge model vs. naive guesses
#        (WORKERS=n: parallel cuts; OPEN_METEO_HISTORY=…: other weather history source, see prognos.py)
#        HA_URL=… HA_ACCESS_TOKEN=… python3 run.py publish   – train + forecast → sensor.elpris_prognos in HA
import datetime as dt, json, multiprocessing, os, statistics as S, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "custom_components", "elprisprognos"))
import prognos as P
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
START = dt.date(2023, 1, 1)

def data(end, history=False, nb_end=None):
    """Prices, weather with reactor outages (final versions; `history` also keeps older versions for as-of replays), reactor
    UMM messages, neighbour prices and every version of the transmission / Finnish reactor messages."""
    px = P.prices(START, end, CACHE)
    wf = os.path.join(CACHE, f"weather2_{START}_{end}.json")   # weather2_: with the v2 stage-1 points
    if not os.path.exists(wf):
        wx = P.weather(START, end)   # download first: if Open-Meteo is down, the old cache survives
        for old in [f for f in os.listdir(CACHE) if f.startswith("weather2_")]: os.remove(os.path.join(CACHE, old))
        json.dump(wx, open(wf, "w"))
    msgs = [] if os.environ.get("NO_NUCLEAR") else P.umm(history)
    nb = P.neighbour_prices(START, nb_end or end, CACHE)
    return px, P.add_nuclear(json.load(open(wf)), msgs), msgs, nb, P.umm(True, P.UMM_TRANS) + P.umm(True, P.UMM_FI)

def _cut(cut):
    """One backtest forecast made at the end of `cut` for cut+2 … cut+7, with only what was known then: (hours, ridge, v2)."""
    px, wx, msgs, nb, m4 = D
    known = f"{cut:%Y-%m-%d}T23"
    tr = {h: v for h, v in px.items() if h <= known}; nbk = {z: {h: v for h, v in d.items() if h <= known} for z, d in nb.items()}
    hrs = [h for h in px if f"{cut + dt.timedelta(days=2):%Y-%m-%d}" <= h[:10] <= f"{cut + dt.timedelta(days=7):%Y-%m-%d}" and h in wx]
    # reactor outages as they were known when the forecast was made, not as they turned out
    nuc = P.nuclear(msgs, min(hrs), max(hrs), as_of=known) if hrs else {}
    wt = {h: {**wx[h], "nuc": nuc.get(h, 0)} for h in hrs}
    ridge = P.predict(P.fit(tr, wx), hrs, wt, P._level(tr, known))
    v2 = P.predict2(P.fit2(tr, wx, nbk, m4), hrs, {**wx, **wt}, tr, nbk, m4, cut.toordinal())
    return hrs, ridge, v2

def backtest():
    global D
    end = dt.date.today() - dt.timedelta(days=1); D = data(end, history=True); px = D[0]
    cuts = [end - dt.timedelta(days=d) for d in range(360, 0, -9)]   # forecast made at the end of `cut`, for cut+2 … cut+7
    with multiprocessing.get_context("fork").Pool(int(os.environ.get("WORKERS", os.cpu_count()))) as pool: res = pool.map(_cut, cuts)
    err = {k: [] for k in ("v2", "ridge (v1)", "samma timme förra veckan", "7-dagars medel")}; rank = {"v2": [], "ridge (v1)": []}
    for cut, (hrs, ridge, v2) in zip(cuts, res):
        tr = {h: v for h, v in px.items() if h <= f"{cut:%Y-%m-%d}T23"}; lvl = P._level(tr, f"{cut:%Y-%m-%d}T23")
        for h in hrs:
            wk = f"{dt.datetime.strptime(h, '%Y-%m-%dT%H') - dt.timedelta(days=7 * (1 + (dt.datetime.strptime(h, '%Y-%m-%dT%H').date() - cut).days // 7)):%Y-%m-%dT%H}"
            err["v2"].append(abs(v2[h] - px[h])); err["ridge (v1)"].append(abs(ridge[h] - px[h])); err["7-dagars medel"].append(abs(lvl - px[h]))
            if wk in tr: err["samma timme förra veckan"].append(abs(tr[wk] - px[h]))
        days = sorted({h[:10] for h in hrs})   # does it rank the days right? (what charging cares about)
        a = [S.mean(px[h] for h in hrs if h[:10] == d) for d in days]
        for k, pr in (("v2", v2), ("ridge (v1)", ridge)):
            if len(days) > 2: rank[k].append(S.correlation(a, [S.mean(pr[h] for h in hrs if h[:10] == d) for d in days]))
    for k, v in err.items(): print(f"{k:26} medelfel {100 * S.mean(v):5.1f} öre/kWh (n={len(v)})")
    for k, v in rank.items(): print(f"dygnsrangordning {k}: korrelation dagsmedel prognos vs verklig {S.mean(v):.2f} (1 = perfekt)")

def publish():
    sys.path.insert(0, os.path.dirname(HERE)); from ws import call
    end = dt.date.today() + dt.timedelta(days=1); px, wx, msgs, nb, m4 = data(dt.date.today() - dt.timedelta(days=1), nb_end=end)
    px.update(P.prices(dt.date.today(), end))   # today + tomorrow (if published) – not cached
    model = P.fit(px, wx)
    try: v2 = P.fit2(px, wx, nb, m4)
    except Exception as e: print("v2 fit failed, ridge only:", e); v2 = None
    fc = P.forecast(model, px, msgs, (v2, wx, nb, m4))
    nps = [s for s in call({"type": "get_states"})[0]["result"] if s["entity_id"] == "sensor.nordpool_kwh_se3_sek_3_10_025"]
    add = float(nps[0]["attributes"].get("additional_costs_current_hour", 0.71)) if nps else 0.71
    vat = 1.25
    loc = lambda h: dt.datetime.strptime(h, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc).astimezone(P.TZ)
    rows = [{"start": loc(h).isoformat(), "value": round(v * vat + add, 3), "spot": round(v, 4), "forecast": f} for h, v, f in fc]
    first = next((r for r in rows if r["forecast"]), None)
    attrs = {"unit_of_measurement": "SEK/kWh", "friendly_name": "Elpris prognos", "icon": "mdi:crystal-ball", "device_class": "monetary",
             "forecast": rows, "forecast_from": first and first["start"], "model_hours": model["n"], "vat": vat, "additional_costs": add,
             "updated": dt.datetime.now(P.TZ).isoformat(timespec="seconds")}
    fut = [r["value"] for r in rows if r["forecast"]]
    state = round(sum(fut[:24]) / len(fut[:24]), 3) if fut else None   # state: mean of the first forecast day
    r = subprocess.run(["curl", "-sf", "-A", "curl/8", "-X", "POST", "-H", f"Authorization: Bearer {os.environ['HA_ACCESS_TOKEN']}",
                        "-H", "Content-Type: application/json", "-d", json.dumps({"state": state, "attributes": attrs}),
                        os.environ["HA_URL"].rstrip("/") + "/api/states/sensor.elpris_prognos"], capture_output=True, text=True)
    print("published" if r.returncode == 0 else f"failed {r.returncode}", len(rows), "hours, forecast from", attrs["forecast_from"], "state", state)

if __name__ == "__main__":
    {"backtest": backtest, "publish": publish}[sys.argv[1] if len(sys.argv) > 1 else "backtest"]()
