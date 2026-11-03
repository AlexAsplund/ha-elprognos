# HA wiring check with synthetic data (no network): uv run --python 3.13 --with pytest-homeassistant-custom-component pytest elprisprognos/tests
import datetime as dt, logging, math, os, shutil
from unittest.mock import patch
import numpy as np, pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from custom_components.elprisprognos import gbm, prognos as P

@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations): yield

@pytest.fixture(autouse=True)
def small_v2():   # a few rounds keep the v2 fits fast
    with patch.object(P, "ROUNDS", 30), patch.object(P, "ROUNDS1", 20), patch.object(P, "LEADS", (2, 5)): yield

def fake_prices(start, end, cache=None):
    out, d = {}, start
    while d <= end:
        for h in range(24): out[f"{d}T{h:02d}"] = 1 + 0.5 * math.sin(h / 24 * 6.28) + 0.1 * (d.toordinal() % 7)
        d += dt.timedelta(days=1)
    return out

def fake_weather(start, end, forecast=False):
    if forecast: start, end = dt.date.today() - dt.timedelta(days=2), dt.date.today() + dt.timedelta(days=7)
    out, d = {}, start
    while d <= end:
        for h in range(24):
            out[f"{d}T{h:02d}"] = {k + "_" + n: (5 + (d.toordinal() + h) % 9 if k == "w" else 3.0 if k == "t" else 100.0) for (n, *_), ks in P.POINTS for k in ks}
        d += dt.timedelta(days=1)
    return out

def fake_nb(start, end, cache=None, token=None):   # neighbour prices in EUR/MWh
    return {z: {h: 40 + 30 * v + i for h, v in fake_prices(start, end).items()} for i, z in enumerate(P.NEIGHBOURS)}

def fake_umm(history=False, q=None):   # one reactor half out from yesterday for 10 days, plus a dismissed message; a line out
    d = dt.date.today() - dt.timedelta(days=1)
    if q == P.UMM_FI: return []
    if q == P.UMM_TRANS:
        return [{"messageId": "t", "version": 1, "publicationDate": f"{d}T08:00:00Z", "transmissionUnits": [{"inAreaName": "SE3", "outAreaName": "SE4",
                 "timePeriods": [{"unavailableCapacity": 900, "eventStart": f"{d}T00:00:00Z", "eventStop": f"{d + dt.timedelta(days=5)}T00:00:00Z"}]}]}]
    tp = lambda mw: [{"unavailableCapacity": mw, "eventStart": f"{d}T00:00:00.0000000Z", "eventStop": f"{d + dt.timedelta(days=10)}T00:00:00.0000000Z"}]
    return [{"messageId": "a", "version": 1, "publicationDate": f"{d}T00:00:00Z", "eventStatus": 1,
             "productionUnits": [{"eic": "R3", "installedCapacity": 1081, "timePeriods": tp(540)}]},
            {"messageId": "b", "version": 2, "publicationDate": f"{d}T00:00:00Z", "eventStatus": 3,
             "productionUnits": [{"eic": "R4", "installedCapacity": 1134, "timePeriods": tp(1134)}]}]

def test_nuclear():
    d = dt.date.today() - dt.timedelta(days=1)
    n = P.nuclear(fake_umm(), f"{d - dt.timedelta(days=1)}T00", f"{d + dt.timedelta(days=11)}T00")
    assert n.get(f"{d - dt.timedelta(days=1)}T23", 0) == 0 and n[f"{d}T00"] == 540 and f"{d + dt.timedelta(days=10)}T00" not in n
    assert P.nuclear(fake_umm(), f"{d}T00", f"{d}T05", as_of=f"{d - dt.timedelta(days=1)}T12") == {}   # not published yet

def test_outages():   # versions: what was known at T23 UTC of a day; dismissed = 0; lines outside TRANS ignored
    t0 = dt.datetime(2026, 3, 1)
    tp = lambda mw, a="2026-03-05T00:00:00Z", b="2026-03-06T00:00:00Z": [{"unavailableCapacity": mw, "eventStart": a, "eventStop": b}]
    line = lambda mw, **k: [{"inAreaName": "SE3", "outAreaName": "SE4", "timePeriods": tp(mw, **k)}]
    msgs = [{"messageId": "a", "version": 1, "publicationDate": "2026-03-02T10:00:00Z", "transmissionUnits": line(500)},
            {"messageId": "a", "version": 2, "publicationDate": "2026-03-03T10:00:00Z", "transmissionUnits": line(700, b="2026-03-05T12:30:00Z")},
            {"messageId": "b", "version": 1, "publicationDate": "2026-03-02T10:00:00Z", "eventStatus": 3, "transmissionUnits": line(2000)},
            {"messageId": "c", "version": 1, "publicationDate": "2026-03-02T10:00:00Z", "transmissionUnits": [{"inAreaName": "SE1", "outAreaName": "SE2", "timePeriods": tp(99)}]},
            {"messageId": "d", "version": 1, "publicationDate": "2026-03-02T10:00:00Z", "generationUnits": [
                {"productionUnitEic": "L1", "eic": g, "installedCapacity": 500, "timePeriods": tp(600)} for g in ("G1", "G2")]}]
    col = P.TRANS.index("SE3-SE4")
    o = P.outages(msgs, t0, 24 * 10, dt.date(2026, 3, 2).toordinal())
    assert o[4 * 24 - 1, col] == 0 and o[4 * 24, col] == 500 and o[5 * 24 - 1, col] == 500 and o[5 * 24, col] == 0 and o.sum(0)[col] == 500 * 24
    assert o[4 * 24, -1] == 1000 and o[:, :col].sum() == 0   # two generators, each capped at its capacity
    o = P.outages(msgs, t0, 24 * 10, math.inf)
    assert o[4 * 24 + 12, col] == 700 and o[4 * 24 + 13, col] == 0   # the newer version: shorter, to 12:30
    assert P.outages(msgs, t0, 24 * 10, dt.date(2026, 3, 1).toordinal()).sum() == 0   # nothing published yet

def test_gbm():   # numpy GBM self-check: huber fits a toy problem despite outliers, the monotone constraint holds
    rng = np.random.default_rng(0); X = rng.random((3000, 4)); y = np.sin(6 * X[:, 0]) + 2 * X[:, 1] + 0.1 * rng.standard_normal(3000)
    y[:30] += 20
    a, b = X[:300].copy(), X[:300].copy(); a[:, 1] = 0.2; b[:, 1] = 0.8
    m = gbm.train(X[:2500], y[:2500], rounds=150, lr=0.05, huber=0.3, mono=[0, 1, 0, 0])
    assert np.sqrt(((gbm.predict(m, X[2500:]) - y[2500:]) ** 2).mean()) < 0.15 and (gbm.predict(m, b) >= gbm.predict(m, a) - 1e-12).all()
    m = gbm.train(X[:2500], y[:2500], rounds=50, mono=[0, -1, 0, 0])
    assert (gbm.predict(m, b) <= gbm.predict(m, a) + 1e-12).all()   # non-increasing in feature 1 although y rises with it
    X[0, 2] = np.nan; assert np.isfinite(gbm.predict(m, X[:1])).all()

def test_v2_and_ridge_fallback(caplog):
    d0, today = dt.date.today() - dt.timedelta(days=300), dt.date.today()
    px, wx, nb = fake_prices(d0, today), fake_weather(d0, today), fake_nb(d0, today)
    ridge, m2 = P.fit(px, wx), P.fit2(px, wx, nb, fake_umm(True, P.UMM_TRANS))
    with patch.object(P, "weather", fake_weather):
        rows = P.forecast(ridge, px, (), (m2, wx, nb, fake_umm(True, P.UMM_TRANS)))
        assert any(f for *_, f in rows) and "ridge" not in caplog.text
        assert all(abs(v - 1.1) < 0.8 for _, v, f in rows if f)   # synthetic prices are 0.5..1.7
        rows = P.forecast(ridge, px, (), (m2, wx, {}, []))   # no neighbour prices → ridge, with a warning
    assert "using the ridge model" in caplog.text and any(f for *_, f in rows)
    with pytest.raises(ValueError): P.fit2(px, wx, {}, [])

async def test_setup_and_forecast(hass, caplog):
    shutil.rmtree(hass.config.path(".storage", "elprisprognos"), ignore_errors=True)   # the test config dir outlives a run
    entry = MockConfigEntry(domain="elprisprognos", data={"area": "SE3", "vat": 25, "additional_costs": 0.71}, unique_id="SE3")
    entry.add_to_hass(hass)
    with patch.object(P, "prices", fake_prices), patch.object(P, "weather", fake_weather), patch.object(P, "umm", fake_umm), \
         patch.object(P, "neighbour_prices", fake_nb), \
         patch("custom_components.elprisprognos.const.HISTORY_START", "2026-01-01"), patch("custom_components.elprisprognos.HISTORY_START", "2026-01-01"):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert os.path.exists(hass.config.path(".storage", "elprisprognos", "model2_SE3.json")) and "ridge" not in caplog.text
    s = hass.states.get("sensor.elpris_prognos_se3")
    assert s is not None and s.state not in ("unavailable", "unknown"), s
    rows = s.attributes["forecast"]
    assert any(r["forecast"] for r in rows) and any(not r["forecast"] for r in rows)
    r = rows[0]; assert abs(r["value"] - (r["spot"] * 1.25 + 0.71)) < 0.002
    st = hass.states.get("sensor.elpris_prognos_se3_status")
    assert st.state == "v2" and st.attributes["weather"] == "Open-Meteo" and st.attributes["fitted"] and st.attributes["fit_seconds"] is not None, st
    assert await hass.config_entries.async_unload(entry.entry_id)
    # a restart reuses the saved models (refit at night) and shows the saved forecast before the first refresh
    with patch.object(P, "prices", fake_prices), patch.object(P, "weather", fake_weather), patch.object(P, "umm", fake_umm), \
         patch.object(P, "neighbour_prices", fake_nb), patch.object(P, "fit", side_effect=AssertionError("refit after restart")), \
         patch.object(P, "fit2", side_effect=AssertionError("refit after restart")):
        assert await hass.config_entries.async_setup(entry.entry_id)
        s = hass.states.get("sensor.elpris_prognos_se3")
        assert s.state not in ("unavailable", "unknown") and s.attributes["forecast"], s
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get("sensor.elpris_prognos_se3").state not in ("unavailable", "unknown") and "ridge" not in caplog.text
    assert await hass.config_entries.async_unload(entry.entry_id)

async def test_config_flow(hass):
    r = await hass.config_entries.flow.async_init("elprisprognos", context={"source": "user"})
    assert r["type"] == "form"
    with patch("custom_components.elprisprognos.async_setup_entry", return_value=True):
        r = await hass.config_entries.flow.async_configure(r["flow_id"], {"area": "SE3", "vat": 25, "additional_costs": 0.71})
    assert r["type"] == "create_entry" and r["title"] == "Elprisprognos SE3"

def test_metno_fallback():   # Open-Meteo down → the live forecast comes from MET Norway instead
    px = fake_prices(dt.date.today() - dt.timedelta(days=60), dt.date.today())
    model = P.fit(px, fake_weather(dt.date.today() - dt.timedelta(days=60), dt.date.today()))
    def down(start, end, forecast=False): raise OSError("503")
    with patch.object(P, "weather", down), patch.object(P, "weather_metno", lambda pts=None: fake_weather(None, None, forecast=True)):
        rows = P.forecast(model, px)
    assert any(f for _, _, f in rows)
    assert P._sun_clear(dt.datetime(2026, 6, 21, 11, tzinfo=dt.timezone.utc), 51, 10) > 750 and P._sun_clear(dt.datetime(2026, 12, 21, 0, tzinfo=dt.timezone.utc), 59, 15) == 0

def test_sunpos():   # Stockholm: ~18.5 h day at midsummer, ~6 h at midwinter; sun up at noon, down at midnight
    s = P._sunpos(dt.datetime(2026, 6, 21, 11, tzinfo=dt.timezone.utc)); w = P._sunpos(dt.datetime(2026, 12, 21, 23, tzinfo=dt.timezone.utc))
    assert 18 < s[2] + s[1] < 19.5 and s[0] > 0.8 and 5.5 < w[3] < 7 and w[0] < 0
