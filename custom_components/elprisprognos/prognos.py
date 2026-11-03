"""SE3 spot price forecast, 7 days ahead. No Home Assistant dependency (stdlib + numpy) so it can become an integration.

Data: hourly spot prices from elprisetjustnu.se (SEK/kWh excl. VAT; 15-min prices averaged per hour), weather from Open-Meteo
(wind at 100 m where the Nordic/German wind farms are, temperature in the big cities, sun over Germany), Swedish reactor
outages from Nord Pool's UMM (REMIT) messages.
v2 adds neighbour day-ahead prices (energy-charts.info or ENTSO-E), transmission outages on the SE3 links and Finnish reactor
outages (UMM), and weather at points in Germany, Finland, Norway and Denmark.
Model v2: two stages of gradient-boosted trees (gbm.py). Stage 1 forecasts the DE-LU/DK1/FI/NO1/NO2 prices from their own
weather; stage 2 forecasts the Swedish price from those forecasts, the weather, price levels and outages.
Model v1 (the fallback when neighbour prices are unavailable): ridge regression where every feature also gets a copy scaled by
the recent price level, so "windy night" means −X % rather than −X öre (the level carries gas prices, hydro balance and season).
"""
import datetime as dt, json, logging, math, os, time, urllib.request, warnings, xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv
try: from . import gbm
except ImportError: import gbm   # run.py imports this file outside the package
TZ = ZoneInfo("Europe/Stockholm")

ZONE = "SE3"
# (name, lat, lon): wind sites that move SE3 prices + a few temperature / solar points
WIND = [("SE1", 65.3, 20.7), ("SE2n", 63.2, 14.6), ("SE2s", 62.4, 17.3), ("SE3", 59.6, 16.5), ("SE4", 55.6, 13.2),
        ("FI", 63.0, 21.6), ("NO2", 58.5, 6.5), ("DK1", 56.0, 8.4), ("DEn", 54.0, 9.0)]
TEMP = [("Stockholm", 59.33, 18.06), ("Oslo", 59.91, 10.75), ("Helsinki", 60.17, 24.94)]
SUN = [("DE", 51.0, 10.0), ("SE3sun", 58.5, 15.0)]
# v2 stage 1: neighbour bidding zones (ENTSO-E EIC) and their own weather points (wind, temperature and sun at each)
NEIGHBOURS = {"DE-LU": "10Y1001A1001A82H", "DK1": "10YDK-1--------W", "FI": "10YFI-1--------U", "NO1": "10YNO-1--------2", "NO2": "10YNO-2--------T"}
REGIONS = {"DE-LU": [(54.2, 9.0), (52.6, 13.0), (49.5, 9.0)],   # North Sea coast wind, Brandenburg wind/solar, south (solar/load)
           "DK1": [(55.5, 7.6), (56.4, 8.5), (56.2, 9.9)],      # Horns Rev offshore, West Jutland onshore, Aarhus
           "FI": [(64.7, 25.2), (62.5, 22.0), (60.2, 24.9)],    # Oulu coast wind, Ostrobothnia wind, Helsinki load
           "NO1": [(59.9, 10.8), (61.1, 10.5)],                 # Oslo, Gudbrandsdalen
           "NO2": [(58.9, 5.7), (58.2, 8.0), (59.6, 7.0)]}      # Stavanger coast, Kristiansand, inland hydro
RPTS = [(f"{z}{i + 1}", la, lo) for z, p in REGIONS.items() for i, (la, lo) in enumerate(p)]
POINTS = [(p, k) for k, ps in (("w", WIND), ("t", TEMP), ("s", SUN), ("wts", RPTS)) for p in ps]   # (point, which of w/t/s it gives)
# OPEN_METEO_HISTORY: other history source for backtests, e.g. a self-hosted "http://127.0.0.1:8081/v1/forecast?models=ecmwf_ifs&"
UA = {"User-Agent": "elprisprognos/0.1 github.com/AlexAsplund/ha-elprognos"}   # MET Norway wants a contact in the User-Agent
_LOG = logging.getLogger(__name__)

def _get(url, tries=3):
    for i in range(tries):   # Open-Meteo sometimes answers with an empty body
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r: return json.load(r)
        except Exception as e:
            if i == tries - 1: raise
            ra = str((getattr(e, "headers", None) or {}).get("Retry-After", ""))   # energy-charts: 429 + Retry-After ~20 s
            time.sleep(min(int(ra), 60) + 1 if ra.isdigit() else 10 * (i + 1))

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
    pts = [p for p, _ in POINTS]
    base = "https://api.open-meteo.com/v1/forecast?forecast_days=8&past_days=2" if forecast else \
           f"{os.environ.get('OPEN_METEO_HISTORY', 'https://historical-forecast-api.open-meteo.com/v1/forecast?')}start_date={start}&end_date={end}"
    url = (base + "&timezone=GMT&wind_speed_unit=ms&hourly=wind_speed_100m,temperature_2m,shortwave_radiation"
           + "&latitude=" + ",".join(str(p[1]) for p in pts) + "&longitude=" + ",".join(str(p[2]) for p in pts))
    res, out = _get(url), {}
    var = {"w": "wind_speed_100m", "t": "temperature_2m", "s": "shortwave_radiation"}
    for ((name, *_), kinds), loc in zip(POINTS, res if isinstance(res, list) else [res]):
        h = loc["hourly"]
        for i, t in enumerate(h["time"]):
            r = out.setdefault(t[:13], {})
            for k in kinds: r[k + "_" + name] = h[var[k]][i]
    return out

SE_NUCLEAR = "10Y1001A1001A46L"   # all Swedish reactors are in SE3; their outages move prices in every Swedish area

UMM_TRANS = "messageTypes=3&areas=10Y1001A1001A46L"    # transmission (SE3 and its links)
UMM_FI = "areas=10YFI-1--------U&fuelTypes=14"           # Finnish reactors

def umm(history=False, q=f"areas={SE_NUCLEAR}&fuelTypes=14"):
    """Nord Pool UMM messages (default: nuclear units in SE3, back to 2012): the current version of each, or every version."""
    out = []
    while True:
        page = _get(f"https://ummapi.nordpoolgroup.com/messages?{q}&limit=500&skip={len(out)}" + "&includeOutdated=true" * history)["items"]
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

def _raw(t, wx):
    """The 22 weather/outage inputs for local time t: wind power per site, mean, mean², temperature, cold, sun, reactors out, season."""
    w = [_power(wx.get("w_" + n)) for n, *_ in WIND]
    temp = [wx.get("t_" + n, 5.0) or 0.0 for n, *_ in TEMP]
    sun = [(wx.get("s_" + n) or 0.0) / 800 for n, *_ in SUN]
    return (w + [sum(w) / len(w), (sum(w) / len(w)) ** 2] + [min(c, 15) / 10 for c in temp] + [max(0, -c) / 10 for c in temp]
            + sun + [(wx.get("nuc") or 0) / 1000] + [math.cos(2 * math.pi * t.timetuple().tm_yday / 365.25), math.sin(2 * math.pi * t.timetuple().tm_yday / 365.25)])

def features(hour, wx, level):
    """Feature vector for one UTC hour. `level` = mean spot price of the last 7 known days."""
    t = dt.datetime.strptime(hour, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc).astimezone(TZ)
    wk = t.weekday() >= 5
    x = [1.0] + [float(t.hour == h) for h in range(24)] + [float(t.weekday() == d) for d in range(7)] + [float(wk and t.hour == h) for h in range(24)]
    x = np.array(x + _raw(t, wx))
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

def _sun_clear(t, lat, lon):
    """Clear-sky global irradiance (W/m², Haurwitz) at UTC time t."""
    n = t.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + n) / 365)
    ha = math.radians(15 * (t.hour + t.minute / 60 + lon / 15 - 12))
    se = math.sin(math.radians(lat)) * math.sin(decl) + math.cos(math.radians(lat)) * math.cos(decl) * math.cos(ha)
    return 1098 * se * math.exp(-0.057 / se) if se > 0.01 else 0.0

def _sunpos(t, lat=59.33, lon=18.07):
    """Sun at Stockholm for the hour starting at t (aware): sin(elevation) at its middle, hours since sunrise, hours to sunset, day length."""
    u = t.astimezone(dt.timezone.utc); n = u.timetuple().tm_yday; h = u.hour + 0.5
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + n) / 365); la = math.radians(lat)
    se = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(math.radians(15 * (h + lon / 15 - 12)))
    h0 = math.degrees(math.acos(max(-1.0, min(1.0, -math.tan(la) * math.tan(decl))))) / 15   # half the day length
    noon = 12 - lon / 15
    return [se, h - (noon - h0), (noon + h0) - h, 2 * h0]

def weather_metno(pts=None):
    """Live forecast from MET Norway (api.met.no), the fallback when Open-Meteo is down. Same keys as weather(forecast=True).
    It has no 100 m wind or irradiance, so: wind 10 m × 1.8 (gives the same 7-day forecast as Open-Meteo within 12 öre, 2026-11), sun = clear sky × (1 − 0.75·cloud³·⁴) (Kasten–Czeplak).
    Hourly for ~2.5 days, then 6-hourly: interpolated linearly."""
    # ponytail: one factor for all sites, tuned on one week; refit per site if the fallback gets used a lot
    out = {}
    for (name, la, lo), kinds in pts or POINTS:
        ts = _get(f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={la}&lon={lo}")["properties"]["timeseries"]
        ts = [(dt.datetime.fromisoformat(x["time"].replace("Z", "+00:00")), x["data"]["instant"]["details"]) for x in ts]
        for (t0, a), (t1, b) in zip(ts, ts[1:]):
            n = int((t1 - t0).total_seconds() // 3600)
            for k in range(n):
                t, f = t0 + dt.timedelta(hours=k), k / n
                v = lambda key: a.get(key, 0) + f * (b.get(key, 0) - a.get(key, 0))
                r = out.setdefault(f"{t:%Y-%m-%dT%H}", {})
                if "w" in kinds: r["w_" + name] = v("wind_speed") * 1.8
                if "t" in kinds: r["t_" + name] = v("air_temperature")
                if "s" in kinds:   # Open-Meteo's value is the mean of the hour before: use mid-hour
                    r["s_" + name] = _sun_clear(t - dt.timedelta(minutes=30), la, lo) * (1 - 0.75 * (v("cloud_area_fraction") / 100) ** 3.4)
    return out

# ---------------- v2 ----------------
# Refit cost with 4 years of history: ~2 min and ~0.5 GB peak (with the weather history) on one desktop CPU core; a Raspberry Pi is
# several times slower. Time scales with ROUNDS × len(LEADS) and memory with len(LEADS): on weak hardware try ROUNDS = 400
# and LEADS = (2, 4, 6): 1 min instead of 2 here (stage 1 and the inputs are a fixed ~35 s), less memory; not backtested.
ROUNDS, ROUNDS1, LEADS = 800, 200, range(2, 8)   # stage-2 rounds, stage-1 rounds, lead days with one training copy each
T0 = dt.datetime(2023, 1, 1)                      # hour 0 of the trend feature
TRANS = ["SE2-SE3", "SE4-SE3", "SE3-SE4", "NO1-SE3", "SE3-NO1", "FI-SE3", "SE3-FI", "DK1-SE3", "SE3-DK1"]   # UMM in-out area pairs
# stage-2 monotone constraints by column (see _x2): less wind and warmth → higher price, more cold / reactors / lines out → higher
MONO = [0] * 8 + [-1] * 14 + [1] * 3 + [0, 0, 1, 0, 0] + [0] * 8 + [0] * 5 + [1] * (len(TRANS) + 1) + [0] * 5
V2 = 3   # stored-model format: bump when the features change, a saved model of another version is refitted at once

def _T(h): return dt.datetime.strptime(h, "%Y-%m-%dT%H")

def _dump(f, obj):   # atomic: several price areas share the cache
    with open(f + ".tmp", "w") as fh: json.dump(obj, fh, separators=(",", ":"))
    os.replace(f + ".tmp", f)

def _hourly(pairs):   # [(UTC datetime, value)] → {'YYYY-MM-DDTHH': mean} (15-min prices averaged per hour)
    acc = {}
    for t, v in pairs:
        if v is not None: acc.setdefault(f"{t:%Y-%m-%dT%H}", []).append(v)
    return {h: sum(v) / len(v) for h, v in acc.items()}

def _energy_charts(z, a, b, token=None):
    d = _get(f"https://api.energy-charts.info/price?bzn={z}&start={a}&end={b}", tries=5)
    return _hourly((dt.datetime.fromtimestamp(t, dt.timezone.utc), p) for t, p in zip(d["unix_seconds"], d["price"]))

def _entsoe(z, a, b, token):
    tag = lambda e, n: [c for c in e.iter() if c.tag.rsplit("}", 1)[-1] == n]
    out = []
    while a <= b:   # at most a year per request
        e = min(b, a + dt.timedelta(days=364))
        url = (f"https://web-api.tp.entsoe.eu/api?securityToken={token}&documentType=A44&in_Domain={NEIGHBOURS[z]}&out_Domain={NEIGHBOURS[z]}"
               f"&periodStart={a:%Y%m%d}0000&periodEnd={e + dt.timedelta(days=1):%Y%m%d}0000")
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r: root = ET.parse(r).getroot()
        for per in tag(root, "Period"):
            t0, t1 = (dt.datetime.strptime(tag(per, k)[0].text, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc) for k in ("start", "end"))
            step = dt.timedelta(minutes={"PT15M": 15, "PT30M": 30, "PT60M": 60}[tag(per, "resolution")[0].text])
            pts, v = {int(tag(p, "position")[0].text): float(tag(p, "price.amount")[0].text) for p in tag(per, "Point")}, None
            for i in range(int((t1 - t0) / step)):   # curve type A03: a missing position repeats the previous price
                v = pts.get(i + 1, v); out.append((t0 + i * step, v))
        a = e + dt.timedelta(days=1)
    if not out: raise ValueError("no prices in the ENTSO-E answer")
    return _hourly(out)

def neighbour_prices(start, end, cache=None, token=None):
    """{zone: {'YYYY-MM-DDTHH' (UTC): EUR/MWh}} for the stage-1 zones, dates start..end. From ENTSO-E when a token is given, else
    (or when that fails) energy-charts.info. Everything before this month is cached in `cache` (one file per zone), so a
    refresh asks for this month only (energy-charts answers 429 to many requests in a row: one per zone and call)."""
    out, month = {}, f"{dt.date.today():%Y-%m}"
    for z in NEIGHBOURS:
        f = cache and os.path.join(cache, f"{z}.json")
        try: acc = json.load(open(f)) if f else {}
        except (OSError, ValueError): acc = {}
        a = dt.date.fromisoformat(max(acc)[:10]) - dt.timedelta(days=1) if acc else start   # a day of overlap: zones end their day at 21/22/23 UTC
        if a <= end:
            for src in ([_entsoe] if token else []) + [_energy_charts]:
                try: got = src(z, a, end, token); break
                except Exception as e:
                    if src is _energy_charts: raise
                    _LOG.warning("ENTSO-E unavailable for %s (%s), trying energy-charts", z, e)
            old = len(acc); acc.update({h: v for h, v in got.items() if h < month})
            if f and len(acc) > old: _dump(f, acc)
            acc.update(got)
        out[z] = {h: v for h, v in acc.items() if f"{start}" <= h[:10] <= f"{end}"}
    return out

def outages(msgs, t0, n, asof):
    """MW out for the n UTC hours from t0: columns TRANS (transmission, in>out area) + Finnish reactors, from the newest version of
    each message published by T23 UTC of day `asof` (date ordinal, per hour or one for all; inf = newest). Like nuclear():
    dismissed messages count 0, overlapping messages for one line or unit count once (the largest)."""
    asof = np.broadcast_to(np.minimum(asof, 4e6), (n,)); vers, unit = {}, {}   # 4e6: after any date, before inf
    for m in msgs: vers.setdefault(m["messageId"], []).append(m)
    for vs in vers.values():
        vs.sort(key=lambda m: m["version"])
        for m, nx in zip(vs, vs[1:] + [None]):   # a version is what's known from its publication day until the next one's
            if m.get("eventStatus") == 3: continue
            a = dt.date.fromisoformat(m["publicationDate"][:10]).toordinal()
            b = dt.date.fromisoformat(nx["publicationDate"][:10]).toordinal() if nx else math.inf
            for kind in ("transmissionUnits", "productionUnits", "generationUnits"):
                for u in m.get(kind) or []:
                    if kind[0] == "t": key, cap = (f"{u.get('inAreaName')}-{u.get('outAreaName')}", None), math.inf
                    else:   # Loviisa is often reported per generator: those add up within the plant
                        key, cap = (u.get("productionUnitEic") or u["eic"], u["eic"] if kind[0] == "g" and u.get("productionUnitEic") else None), u["installedCapacity"]
                    if kind[0] == "t" and key[0] not in TRANS: continue
                    for tp in u["timePeriods"]:
                        i0 = max(0, int((_T(tp["eventStart"][:13]) - t0).total_seconds() // 3600))
                        stop = tp.get("eventStop")
                        i1 = n if not stop else min(n, math.ceil((dt.datetime.strptime(stop[:16], "%Y-%m-%dT%H:%M") - t0).total_seconds() / 3600))
                        if i1 <= i0: continue
                        arr = unit.setdefault((kind[0] == "t",) + key, np.zeros(n))
                        vis = (asof[i0:i1] >= a) & (asof[i0:i1] < b)
                        arr[i0:i1] = np.maximum(arr[i0:i1], np.where(vis, min(tp["unavailableCapacity"], cap), 0))
    out, plant, gens = np.zeros((n, len(TRANS) + 1)), {}, {}
    for (tr, k, g), arr in unit.items():
        if tr: out[:, TRANS.index(k)] = arr
        elif g is None: plant[k] = np.maximum(plant.get(k, 0), arr)
        else: gens[k] = gens.get(k, 0) + arr
    for k in plant.keys() | gens.keys(): out[:, -1] += np.maximum(plant.get(k, 0), gens.get(k, 0))
    return out

def _easter(y):   # anonymous Gregorian algorithm
    a, b, c = y % 19, y // 100, y % 100; d, e = b // 4, b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i, k = c // 4, c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7; m = (a + 11 * h + 22 * l) // 451
    return dt.date(y, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)

def _holidays(y):   # Swedish public holidays and the eves that behave like them
    e, D = _easter(y), dt.timedelta
    mid = next(dt.date(y, 6, d) for d in range(19, 26) if dt.date(y, 6, d).weekday() == 4)
    return {dt.date(y, 1, 1), dt.date(y, 1, 6), e - D(2), e + D(1), dt.date(y, 5, 1), e + D(39), dt.date(y, 6, 6), mid,
            dt.date(y, 12, 24), dt.date(y, 12, 25), dt.date(y, 12, 26), dt.date(y, 12, 31)}

def _mean(a, w, need=1):   # mean of the non-NaN values in the w hours ending at each index (NaN with fewer than `need`)
    c, s = np.r_[0, np.cumsum(~np.isnan(a))], np.r_[0, np.cumsum(np.nan_to_num(a))]
    j = np.maximum(np.arange(1, len(a) + 1) - w, 0); k = c[1:] - c[j]
    return np.where(k >= need, (s[1:] - s[j]) / np.maximum(k, 1), np.nan)

def _lag(a, d):   # the value d days earlier
    return np.concatenate([np.repeat(a[:1], 24 * d, 0), a[:-24 * d]])

def _frame(px, wx, nb, a, b):
    """Every v2 input on the UTC hour grid a..b as arrays."""
    t0 = _T(a); n = int((_T(b) - t0).total_seconds() // 3600) + 1
    G = [f"{t0 + dt.timedelta(hours=i):%Y-%m-%dT%H}" for i in range(n)]
    loc = [(t0 + dt.timedelta(hours=i)).replace(tzinfo=dt.timezone.utc).astimezone(TZ) for i in range(n)]
    hol = set().union(*(_holidays(y) for y in range(loc[0].year, loc[-1].year + 1)))
    p = np.array([px.get(h, np.nan) for h in G])
    raw = np.array([_raw(t, wx[h]) if h in wx else [np.nan] * 22 for h, t in zip(G, loc)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # all-NaN windows
        win = lambda x, w: _swv(np.r_[np.full(w - 1, np.nan), x], w)
        L7, L28 = (np.nan_to_num(_mean(p, w), nan=0.8) for w in (168, 672))
        M7 = np.nan_to_num(np.nanmedian(win(p, 168), 1), nan=0.8)
        W, Pw = win(raw[:, 9], 168), win(p, 168)   # mean of the 7 days' calmer half: the price when the wind doesn't help
        calm = (W < np.nanmedian(W, 1)[:, None]) & ~np.isnan(Pw)
        C7 = np.where(calm.any(1), np.where(calm, Pw, 0).sum(1) / np.maximum(calm.sum(1), 1), L7)
        p24 = win(p, 24); R24 = np.nanmax(p24, 1) - np.nanmin(p24, 1)                 # range of the last 24 h
        S7 = np.nan_to_num(_mean(R24, 168), nan=0.5)                                   # mean daily swing, last 7 days
        H7 = np.nan_to_num(np.nanmean(np.stack([_lag(p, d) for d in range(1, 8)], 1), 1), nan=0.8) - L7   # same hour, last 7 days
    nbp = np.array([[nb.get(z, {}).get(h, np.nan) for z in NEIGHBOURS] for h in G])
    return {"t0": t0, "n": n, "G": G, "price": p, "raw": raw, "lev": np.stack([L7, L28, M7, C7, S7, H7], 1),
            "cal": np.array([[t.hour, t.weekday(), t.weekday() >= 5, t.date() in hol] + _sunpos(t) for t in loc], float),
            "ts": (np.arange(n) + (t0 - T0).total_seconds() / 3600)[:, None], "hod": (np.arange(n) + t0.hour) % 24,
            "day": t0.toordinal() + (np.arange(n) + t0.hour) // 24,
            "nbp": nbp, "nbl": np.stack([_mean(nbp[:, c], 168, 43) for c in range(len(NEIGHBOURS))], 1),
            "regw": {z: np.array([[wx.get(h, {}).get(f"{k}_{z}{i + 1}") for i in range(len(pts)) for k in "wts"] for h in G], float)
                     for z, pts in REGIONS.items()}}

def _x1(f, i, z, lev):   # stage 1, zone z: calendar, Swedish weather, the zone's own weather, its price level, trend
    return np.concatenate([f["cal"][i], f["raw"][i], f["regw"][z][i], lev[:, None], f["ts"][i]], 1)

def _x2(f, i, lead, lev, nbl, out, s1):   # stage 2: calendar, weather, SE price levels, trend, lead, neighbour levels, outages, stage 1
    return np.concatenate([f["cal"][i], f["raw"][i], lev, f["ts"][i], lead[:, None], nbl, out, s1], 1)

def fit2(px, wx, nb, msgs, rounds=None, leads=None):
    """v2 fit on every hour with a price and weather. `wx` has the reactor outages (final versions), `nb` = neighbour_prices(),
    `msgs` = every version of the UMM messages about transmission and Finnish reactors (replayed as known at forecast time)."""
    f, rounds, leads = _frame(px, wx, nb, min(px), max(px)), rounds or ROUNDS, leads or LEADS; n = f["n"]
    tr = np.flatnonzero(~np.isnan(f["price"]) & ~np.isnan(f["raw"][:, 0]))
    m1, s1 = {}, []
    for c, z in enumerate(NEIGHBOURS):   # stage 1, in-sample: the same model makes the training and the live inputs of stage 2
        y = f["nbp"][:, c]; ok = tr[~np.isnan(y[tr])]
        if len(ok) < 5000: raise ValueError(f"only {len(ok)} hours of {z} prices")
        fill = float(np.nanmean(f["nbl"][:, c])); lev = _lag(np.nan_to_num(f["nbl"][:, c], nan=fill), 2)[ok]
        m1[z] = gbm.train(_x1(f, ok, z, lev), y[ok], lr=0.1, rounds=ROUNDS1)
        o = np.full(n, np.nan); o[ok] = gbm.predict(m1[z], _x1(f, ok, z, lev)); s1.append(o[tr])
    s1, X = np.stack(s1, 1), []
    for d in leads:   # one copy of the training hours per lead day, with levels and outages as known d days before
        ki = np.maximum(tr - f["hod"][tr] - 24 * (d - 1) - 1, 0)   # T23 UTC d days before
        X.append(_x2(f, tr, np.full(len(tr), float(d)), _lag(f["lev"], d)[tr], f["nbl"][ki], outages(msgs, f["t0"], n, f["day"] - d)[tr], s1))
    X = np.concatenate(X)
    return {"v": V2, "n": len(tr), "leads": [min(leads), max(leads)], "s1": m1,
            "s2": gbm.train(X, np.tile(f["price"][tr], len(leads)), rounds=rounds, huber=0.3, mono=MONO)}

def predict2(model, hours, wx, px, nb, msgs, asof=math.inf):
    """v2 forecast for `hours` (those in wx): levels as of the last hour in px, UMM messages as known on day `asof` (ordinal)."""
    k_h, hours = max(px), [h for h in hours if h in wx]
    f = _frame(px, wx, nb, f"{_T(k_h) - dt.timedelta(hours=700):%Y-%m-%dT%H}", max(hours + [k_h]))
    ix = {h: j for j, h in enumerate(f["G"])}; k, i = ix[k_h], np.array([ix[h] for h in hours])
    nbl = f["nbl"][k]
    if np.isnan(nbl).any(): raise ValueError("no recent prices for " + ", ".join(z for z, v in zip(NEIGHBOURS, nbl) if np.isnan(v)))
    s1 = np.stack([gbm.predict(model["s1"][z], _x1(f, i, z, np.full(len(i), nbl[c]))) for c, z in enumerate(NEIGHBOURS)], 1)
    lead = np.clip(f["day"][i] - f["day"][k], *model["leads"]).astype(float)
    lev = np.tile(f["lev"][k], (len(i), 1))
    lev[:, -1] = f["lev"][k - (f["hod"][k] - f["hod"][i]) % 24, -1]   # the latest known same-hour profile
    X = _x2(f, i, lead, lev, np.tile(nbl, (len(i), 1)), outages(msgs, f["t0"], f["n"], asof)[i], s1)
    return dict(zip(hours, gbm.predict(model["s2"], X).tolist()))

def forecast(model, px, msgs=(), v2=None, info=None):
    """Next 7 days: real prices where published, the model for the rest. Returns [(utc hour, SEK/kWh spot, is_forecast)].
    v2 = (v2 model, weather of the last 7+ days, neighbour prices, transmission/Finnish UMM messages); the ridge `model` is
    used when v2 is missing or fails. `info` (a dict) gets which model and weather source were used, and why not v2."""
    info = {} if info is None else info
    try: wx = weather(None, None, forecast=True); info["weather"] = "Open-Meteo"
    except Exception as e:
        _LOG.warning("Open-Meteo unavailable (%s), using MET Norway's forecast", e)
        wx = weather_metno(); info["weather"] = "MET Norway"
    wx = add_nuclear(wx, msgs)
    last = max(px)
    hours = sorted(h for h in wx if h > last)
    pred = None
    if v2 and v2[0]:
        try: pred = predict2(v2[0], hours, {**(v2[1] or {}), **wx}, px, v2[2] or {}, v2[3] or ())
        except Exception as e: _LOG.warning("v2 model unavailable (%s), using the ridge model", e); info["problem"] = f"v2: {e}"
    else: info["problem"] = "v2 model not fitted yet"
    info["model"] = "v2" if pred is not None else "ridge"
    if pred is None: pred = predict(model, hours, wx, _level(px, last))
    return [(h, px[h], False) for h in sorted(px) if h >= f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H}"] + [(h, pred[h], True) for h in hours if h in pred]
