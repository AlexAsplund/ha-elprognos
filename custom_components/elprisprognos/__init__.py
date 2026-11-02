"""Elprisprognos: spot price forecast 7 days ahead for the Swedish price areas."""
from __future__ import annotations
import datetime as dt, json, logging, os, time
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from . import prognos as P
from .const import CONF_AREA, CONF_TOKEN, DOMAIN, HISTORY_START

_LOG = logging.getLogger(__name__)
PLATFORMS = ["sensor"]

class Coordinator(DataUpdateCoordinator):
    """Hourly: refresh prices + weather forecast and predict with the last fit. Daily, on the first update after 02:00 local time
    (at once without a fit): refit both models in the executor (v2 takes minutes of CPU, see prognos.ROUNDS)."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(hass, _LOG, name=DOMAIN, update_interval=dt.timedelta(hours=1))
        self.entry, self.model, self.model2, self.fitted, self.wx_hist, self.umm = entry, None, None, None, {}, []
        self.umm4, self.umm4_day, self.nb, self.fit_seconds = [], None, None, None
        self.cache = hass.config.path(".storage", DOMAIN); os.makedirs(self.cache, exist_ok=True)
        area = entry.data[CONF_AREA]
        self.state_file, self.model_file = (os.path.join(self.cache, f"{n}_{area}.json") for n in ("state", "model2"))
        self.wx_file = os.path.join(self.cache, "weather_v2.json")   # weather history, shared by all areas

    def load_state(self):
        """Last models + forecast from before a restart: the sensor has values at once and nothing is refitted before 02:00."""
        try: s = json.load(open(self.state_file))
        except (OSError, ValueError): return
        self.model, self.data, self.fit_seconds = s["model"], s["data"], s.get("fit_seconds")
        try: self.model2, self.fitted = json.load(open(self.model_file)), dt.date.fromisoformat(s["fitted"])
        except (OSError, ValueError): return   # state from 0.1 (ridge only): refit now
        self.wx_hist = self._recent(self._load_wx())

    def _load_wx(self):
        try: return json.load(open(self.wx_file))
        except (OSError, ValueError): return {}

    def _recent(self, wx):   # predicting needs the last week's weather (calm-hours price level); the rest stays on disk
        cut = f"{dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=9):%Y-%m-%dT%H}"
        return {h: v for h, v in wx.items() if h >= cut}

    def _save_state(self, data):
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f: json.dump({"model": self.model, "fitted": self.fitted.isoformat(), "fit_seconds": self.fit_seconds, "data": data}, f)
        os.replace(tmp, self.state_file)

    def _work(self):
        P.ZONE = self.entry.data[CONF_AREA]
        token = self.entry.options.get(CONF_TOKEN, self.entry.data.get(CONF_TOKEN)) or None
        today, start = dt.date.today(), dt.date.fromisoformat(HISTORY_START)
        px = P.prices(start, today - dt.timedelta(days=1), self.cache)   # cached per day after the first run
        px.update(P.prices(today, today + dt.timedelta(days=1)))
        try: self.umm = P.umm()   # reactor outages: news within the hour matters
        except Exception as e: _LOG.warning("Nord Pool UMM unavailable, keeping the last %d messages: %s", len(self.umm), e)
        if self.umm4_day != today:   # transmission + Finnish reactors, every version (training replays them as known back then)
            try: self.umm4, self.umm4_day = P.umm(True, P.UMM_TRANS) + P.umm(True, P.UMM_FI), today
            except Exception as e: _LOG.warning("Nord Pool UMM (transmission, Finland) unavailable, keeping the last %d messages: %s", len(self.umm4), e)
        if not self.nb or min(max(v, default="")[:10] for v in self.nb.values()) < max(px)[:10]:   # new day-ahead prices
            try: self.nb = P.neighbour_prices(start, today + dt.timedelta(days=1), self.cache, token)
            except Exception as e: _LOG.warning("Neighbour prices unavailable (%s); the ridge model is used without them", e)
        if self.fitted is None or self.fitted < today and dt.datetime.now().hour >= 2:
            wx = self._load_wx()
            try: wx.update(P.weather(dt.date.fromisoformat(max(k[:10] for k in wx)) if wx else start, today - dt.timedelta(days=1)))
            except Exception as e:   # a few missing recent days don't matter for a refit; an empty or stale history does
                fresh = wx and max(wx) >= f"{today - dt.timedelta(days=4)}"
                _LOG.warning("Weather history unavailable (%s), %s", e, "refitting on what is stored" if fresh else "refit postponed")
                if not fresh: wx = None
            if wx is not None:
                P._dump(self.wx_file, wx)
                wx = P.add_nuclear(wx, self.umm)
                self.model, t0 = P.fit(px, wx), time.monotonic()
                try:
                    self.model2 = P.fit2(px, wx, self.nb or {}, self.umm4)
                    P._dump(self.model_file, self.model2); self.fit_seconds = round(time.monotonic() - t0)
                except Exception as e: _LOG.warning("v2 fit failed (%s), keeping the previous one%s", e, "" if self.model2 else " (ridge until then)")
                self.fitted, self.wx_hist = today, self._recent(wx)
        if self.model is None: raise RuntimeError("no model yet (weather history unavailable)")
        info = {"fitted": self.fitted and self.fitted.isoformat(), "fit_seconds": self.fit_seconds,
                "neighbour_prices_until": min((max(v, default="") for v in (self.nb or {}).values()), default=None) or None}
        return P.forecast(self.model, px, self.umm, (self.model2, self.wx_hist, self.nb, self.umm4), info), self.model["n"], info

    async def _async_update_data(self):
        rows, n, info = await self.hass.async_add_executor_job(self._work)
        data = {"rows": rows, "model_hours": n, "updated": dt_util.now().isoformat(timespec="seconds"), "status": info}
        try: await self.hass.async_add_executor_job(self._save_state, data)
        except (OSError, TypeError) as e: _LOG.warning("Could not save the forecast state: %s", e)
        return data

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coord = Coordinator(hass, entry)
    await hass.async_add_executor_job(coord.load_state)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coord
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    # first run downloads ~4 years of prices and weather and fits the models (minutes): don't hold up HA's startup
    entry.async_create_background_task(hass, coord.async_refresh(), f"{DOMAIN}_first_refresh")
    entry.async_on_unload(entry.add_update_listener(lambda h, e: h.config_entries.async_reload(e.entry_id)))
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if ok: hass.data[DOMAIN].pop(entry.entry_id)
    return ok
