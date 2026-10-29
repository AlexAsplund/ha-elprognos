# HA wiring check with synthetic data (no network): uv run --python 3.13 --with pytest-homeassistant-custom-component pytest elprisprognos/tests
import datetime as dt, math
from unittest.mock import patch
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from custom_components.elprisprognos import prognos as P

@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations): yield

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
            out[f"{d}T{h:02d}"] = {**{"w_" + n: 5 + (d.toordinal() + h) % 9 for n, *_ in P.WIND}, **{"t_" + n: 3.0 for n, *_ in P.TEMP}, **{"s_" + n: 100.0 for n, *_ in P.SUN}}
        d += dt.timedelta(days=1)
    return out

async def test_setup_and_forecast(hass):
    entry = MockConfigEntry(domain="elprisprognos", data={"area": "SE3", "vat": 25, "additional_costs": 0.71}, unique_id="SE3")
    entry.add_to_hass(hass)
    with patch.object(P, "prices", fake_prices), patch.object(P, "weather", fake_weather), \
         patch("custom_components.elprisprognos.const.HISTORY_START", "2026-01-01"), patch("custom_components.elprisprognos.HISTORY_START", "2026-01-01"):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
    s = hass.states.get("sensor.elpris_prognos_se3")
    assert s is not None and s.state not in ("unavailable", "unknown"), s
    rows = s.attributes["forecast"]
    assert any(r["forecast"] for r in rows) and any(not r["forecast"] for r in rows)
    r = rows[0]; assert abs(r["value"] - (r["spot"] * 1.25 + 0.71)) < 0.002
    assert await hass.config_entries.async_unload(entry.entry_id)

async def test_config_flow(hass):
    r = await hass.config_entries.flow.async_init("elprisprognos", context={"source": "user"})
    assert r["type"] == "form"
    with patch("custom_components.elprisprognos.async_setup_entry", return_value=True):
        r = await hass.config_entries.flow.async_configure(r["flow_id"], {"area": "SE3", "vat": 25, "additional_costs": 0.71})
    assert r["type"] == "create_entry" and r["title"] == "Elprisprognos SE3"
