"""Elprisprognos: spot price forecast 7 days ahead for the Swedish price areas."""
from __future__ import annotations
import datetime as dt, logging, os
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from . import prognos as P
from .const import CONF_AREA, DOMAIN, HISTORY_START

_LOG = logging.getLogger(__name__)
PLATFORMS = ["sensor"]

class Coordinator(DataUpdateCoordinator):
    """Hourly: refresh prices + weather forecast and predict. Daily: refit the model (a few seconds in the executor)."""
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        super().__init__(hass, _LOG, name=DOMAIN, update_interval=dt.timedelta(hours=1))
        self.entry, self.model, self.fitted, self.wx_hist = entry, None, None, {}
        self.cache = hass.config.path(".storage", DOMAIN); os.makedirs(self.cache, exist_ok=True)

    def _work(self):
        P.ZONE = self.entry.data[CONF_AREA]
        today = dt.date.today()
        px = P.prices(dt.date.fromisoformat(HISTORY_START), today - dt.timedelta(days=1), self.cache)   # cached per day after the first run
        px.update(P.prices(today, today + dt.timedelta(days=1)))
        if self.model is None or self.fitted != today:
            start = dt.date.fromisoformat(max(HISTORY_START, *(k[:10] for k in self.wx_hist))) if self.wx_hist else dt.date.fromisoformat(HISTORY_START)
            self.wx_hist.update(P.weather(start, today - dt.timedelta(days=1)))
            self.model, self.fitted = P.fit(px, self.wx_hist), today
        return P.forecast(self.model, px), self.model["n"]

    async def _async_update_data(self):
        rows, n = await self.hass.async_add_executor_job(self._work)
        return {"rows": rows, "model_hours": n, "updated": dt_util.now().isoformat(timespec="seconds")}

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    coord = Coordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coord
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    # first run downloads ~4 years of prices and weather (minutes): don't hold up HA's startup
    entry.async_create_background_task(hass, coord.async_refresh(), f"{DOMAIN}_first_refresh")
    entry.async_on_unload(entry.add_update_listener(lambda h, e: h.config_entries.async_reload(e.entry_id)))
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if ok: hass.data[DOMAIN].pop(entry.entry_id)
    return ok
