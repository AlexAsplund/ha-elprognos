"""SE3 spot price forecast, 7 days ahead. No Home Assistant dependency (stdlib + numpy) so it can become an integration.

Data: hourly spot prices from elprisetjustnu.se (SEK/kWh excl. VAT; 15-min prices averaged per hour), weather from Open-Meteo
(wind at 100 m where the Nordic/German wind farms are, temperature in the big cities, sun over Germany), Swedish reactor
outages from Nord Pool's UMM (REMIT) messages.
Model: ridge regression where every feature also gets a copy scaled by the recent price level, so "windy night" means
−X % rather than −X öre (the level carries gas prices, hydro balance and season).
"""
import datetime as dt, json, math, os, time, urllib.request
from zoneinfo import ZoneInfo
import numpy as np
TZ = ZoneInfo("Europe/Stockholm")

ZONE = "SE3"
# (name, lat, lon): wind sites that move SE3 prices + a few temperature / solar points
WIND = [("SE1", 65.3, 20.7), ("SE2n", 63.2, 14.6), ("SE2s", 62.4, 17.3), ("SE3", 59.6, 16.5), ("SE4", 55.6, 13.2),
        ("FI", 63.0, 21.6), ("NO2", 58.5, 6.5), ("DK1", 56.0, 8.4), ("DEn", 54.0, 9.0)]
TEMP = [("Stockholm", 59.33, 18.06), ("Oslo", 59.91, 10.75), ("Helsinki", 60.17, 24.94)]
SUN = [("DE", 51.0, 10.0), ("SE3sun", 58.5, 15.0)]
UA = {"User-Agent": "elprisprognos/0.1 (Home Assistant)"}

def _get(url, tries=3):
    for i in range(tries):   # Open-Meteo sometimes answers with an empty body
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r: return json.load(r)
        except Exception:
            if i == tries - 1: raise
            time.sleep(10 * (i + 1))

def prices(start, end, cache=None):
    """{'YYYY-MM-DDTHH' (UTC): SEK/kWh} for start..end (dates, inclusive). Past days are cached as files in `cache`."""
    out, d = {}, start
    while d <= end:
        f = cache and os.path.join(cache, f"{ZONE}_{d}.json")
        if f and os.path.exists(f): day = json.load(open(f))
        else:
            try: day = _get(f"https://www.elprisetjustnu.se/api/v1/prices/{d:%Y}/{d:%m-%d}_{ZONE}.json", tries=1)
            except Exception: day = None   # tomorrow before ~13:00, or a gap
            if f and day and d < dt.date.today(): json.dump(day, open(f, "w"))
        acc = {}
        for p in day or []:
            h = dt.datetime.fromisoformat(p["time_start"]).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H")
            acc.setdefault(h, []).append(p["SEK_per_kWh"])
        out.update({h: sum(v) / len(v) for h, v in acc.items()})
        d += dt.timedelta(days=1)
    return out

def weather(start, end, forecast=False):
    """{'YYYY-MM-DDTHH': {feature: value}} from Open-Meteo (history = archived forecasts, so it looks like what we predict from)."""
    if not forecast and (end - start).days > 200:   # keep requests small
        out, a = {}, start
        while a <= end:
            b = min(end, a + dt.timedelta(days=180)); out.update(weather(a, b)); a = b + dt.timedelta(days=1)
        return out
    pts = WIND + TEMP + SUN
    base = "https://api.open-meteo.com/v1/forecast?forecast_days=8&past_days=2" if forecast else \
           f"https://historical-forecast-api.open-meteo.com/v1/forecast?start_date={start}&end_date={end}"
    url = (base + "&timezone=GMT&wind_speed_unit=ms&hourly=wind_speed_100m,temperature_2m,shortwave_radiation"
           + "&latitude=" + ",".join(str(p[1]) for p in pts) + "&longitude=" + ",".join(str(p[2]) for p in pts))
    res, out = _get(url), {}
    for (name, *_), loc in zip(pts, res if isinstance(res, list) else [res]):
        h = loc["hourly"]
        for i, t in enumerate(h["time"]):
            r = out.setdefault(t[:13], {})
            if any(name == w[0] for w in WIND): r["w_" + name] = h["wind_speed_100m"][i]
            if any(name == w[0] for w in TEMP): r["t_" + name] = h["temperature_2m"][i]
            if any(name == w[0] for w in SUN): r["s_" + name] = h["shortwave_radiation"][i]
    return out

SE_NUCLEAR = "10Y1001A1001A46L"   # all Swedish reactors are in SE3; their outages move prices in every Swedish area

def umm(history=False):
    """Nord Pool UMM messages about nuclear units in SE3 (back to 2012): the current version of each, or every version."""
    out = []
    while True:
        page = _get(f"https://ummapi.nordpoolgroup.com/messages?areas={SE_NUCLEAR}&fuelTypes=14&limit=500&skip={len(out)}"
                    + "&includeOutdated=true" * history)["items"]
        out += page
        if len(page) < 500: return out

def nuclear(msgs, start, end, as_of=None):
    """{'YYYY-MM-DDTHH': MW of reactor capacity out} for UTC hours start..end, from the newest version of each message
    published by `as_of` (an hour string; None = newest). Overlapping messages for one unit count once (the largest)."""
    last = {}
    for m in msgs:
        if (as_of is None or m["publicationDate"][:13] <= as_of) and m["version"] > last.get(m["messageId"], {"version": -1})["version"]:
            last[m["messageId"]] = m
    unit, a0, b0 = {}, dt.datetime.strptime(start, "%Y-%m-%dT%H"), dt.datetime.strptime(end, "%Y-%m-%dT%H")
    for m in last.values():
        if m.get("eventStatus") == 3: continue   # dismissed
        for u in (m.get("productionUnits") or []) + (m.get("generationUnits") or []):
            key = u.get("productionUnitEic") or u["eic"]
            for tp in u["timePeriods"]:
                stop = (tp.get("eventStop") or "9999")[:16]
                h = max(a0, dt.datetime.strptime(tp["eventStart"][:13], "%Y-%m-%dT%H"))
                while h <= b0 and f"{h:%Y-%m-%dT%H:%M}" < stop:
                    k = (key, f"{h:%Y-%m-%dT%H}"); unit[k] = max(unit.get(k, 0), min(tp["unavailableCapacity"], u["installedCapacity"]))
                    h += dt.timedelta(hours=1)
    out = {}
    for (_, h), mw in unit.items(): out[h] = out.get(h, 0) + mw
    return out

def add_nuclear(wx, msgs, as_of=None):
    """Put MW of reactors out into every hour of `wx` (key 'nuc')."""
    n = nuclear(msgs, min(wx), max(wx), as_of) if wx else {}
    for h, r in wx.items(): r["nuc"] = n.get(h, 0)
    return wx

def _power(v):   # rough wind turbine power curve, 0..1 (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s)
    return 0.0 if v is None or v < 3 or v > 25 else min(1.0, (v ** 3 - 27) / (12 ** 3 - 27))

def features(hour, wx, level):
    """Feature vector for one UTC hour. `level` = mean spot price of the last 7 known days."""
    t = dt.datetime.strptime(hour, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc).astimezone(TZ)
    wk = t.weekday() >= 5
    x = [1.0] + [float(t.hour == h) for h in range(24)] + [float(t.weekday() == d) for d in range(7)] + [float(wk and t.hour == h) for h in range(24)]
    w = [_power(wx.get("w_" + n)) for n, *_ in WIND]
    temp = [wx.get("t_" + n, 5.0) or 0.0 for n, *_ in TEMP]
    sun = [(wx.get("s_" + n) or 0.0) / 800 for n, *_ in SUN]
    x += w + [sum(w) / len(w), (sum(w) / len(w)) ** 2] + [min(c, 15) / 10 for c in temp] + [max(0, -c) / 10 for c in temp]
    x += sun + [(wx.get("nuc") or 0) / 1000] + [math.cos(2 * math.pi * t.timetuple().tm_yday / 365.25), math.sin(2 * math.pi * t.timetuple().tm_yday / 365.25)]
    x = np.array(x)
    return np.concatenate([x, x * level])   # every effect both in öre and relative to the price level

def _level(px, until, days=7):
    end = dt.datetime.strptime(until, "%Y-%m-%dT%H")
    v = [px[k] for k in (f"{end - dt.timedelta(hours=i):%Y-%m-%dT%H}" for i in range(24 * days)) if k in px]
    return sum(v) / len(v) if v else 0.8

def fit(px, wx, lead_days=2, lam=3.0):
    """Ridge fit. Each training hour sees the price level as it was `lead_days` before (like a real forecast would)."""
    X, y = [], []
    for h, p in px.items():
        if h in wx:
            known = f"{dt.datetime.strptime(h, '%Y-%m-%dT%H') - dt.timedelta(days=lead_days):%Y-%m-%dT%H}"
            X.append(features(h, wx[h], _level(px, known))); y.append(p)
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9; sd[0] = 1; mu[0] = 0
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + lam * len(y) / 1000 * np.eye(Z.shape[1]), Z.T @ y)
    return {"beta": beta.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "n": len(y)}

def predict(model, hours, wx, level):
    b, mu, sd = (np.array(model[k]) for k in ("beta", "mu", "sd"))
    return {h: float(((features(h, wx[h], level) - mu) / sd) @ b) for h in hours if h in wx}

def forecast(model, px, msgs=()):
    """Next 7 days: real prices where published, the model for the rest. Returns [(utc hour, SEK/kWh spot, is_forecast)]."""
    wx = add_nuclear(weather(None, None, forecast=True), msgs)
    last = max(px)
    hours = sorted(h for h in wx if h > last)
    pred = predict(model, hours, wx, _level(px, last))
    return [(h, px[h], False) for h in sorted(px) if h >= f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H}"] + [(h, pred[h], True) for h in hours if h in pred]
