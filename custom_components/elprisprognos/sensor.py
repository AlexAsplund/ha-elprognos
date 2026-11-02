from __future__ import annotations
import datetime as dt
from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.const import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from . import prognos as P
from .const import CONF_ADD, CONF_AREA, CONF_VAT, DOMAIN

async def async_setup_entry(hass, entry, add):
    coord = hass.data[DOMAIN][entry.entry_id]
    add([Forecast(coord, entry), Status(coord, entry)])

class Forecast(CoordinatorEntity, SensorEntity):
    """State: mean price of the first forecast day. Attribute `forecast`: every hour, published prices first, then the model."""
    _attr_has_entity_name, _attr_icon = True, "mdi:crystal-ball"
    _attr_native_unit_of_measurement, _attr_state_class = "SEK/kWh", SensorStateClass.MEASUREMENT

    def __init__(self, coord, entry):
        super().__init__(coord)
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_forecast"
        self._attr_name = f"Elpris prognos {entry.data[CONF_AREA]}"

    def _rows(self):
        vat = 1 + self.entry.options.get(CONF_VAT, self.entry.data[CONF_VAT]) / 100
        add = self.entry.options.get(CONF_ADD, self.entry.data[CONF_ADD])
        loc = lambda h: dt.datetime.strptime(h, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc).astimezone(P.TZ)
        return [{"start": loc(h).isoformat(), "value": round(v * vat + add, 3), "spot": round(v, 4), "forecast": f}
                for h, v, f in (self.coordinator.data or {}).get("rows", [])]

    @property
    def available(self):
        return bool(self.coordinator.data)

    @property
    def native_value(self):
        fut = [r["value"] for r in self._rows() if r["forecast"]][:24]
        return round(sum(fut) / len(fut), 3) if fut else None

    @property
    def extra_state_attributes(self):
        rows = self._rows(); first = next((r for r in rows if r["forecast"]), None)
        d = self.coordinator.data or {}
        return {"forecast": rows, "forecast_from": first and first["start"], "model_hours": d.get("model_hours"), "updated": d.get("updated")}

class Status(CoordinatorEntity, SensorEntity):
    """State: the model behind the forecast ("v2", or "ridge" when v2 is missing/failing) or "error" when the last update failed.
    Attributes: weather source, last fit (date, seconds), neighbour prices known until, the last problem, last update."""
    _attr_has_entity_name, _attr_icon, _attr_entity_category = True, "mdi:list-status", EntityCategory.DIAGNOSTIC

    def __init__(self, coord, entry):
        super().__init__(coord)
        self._attr_unique_id = f"{entry.entry_id}_status"
        self._attr_name = f"Elpris prognos {entry.data[CONF_AREA]} status"

    @property
    def available(self):
        return True

    @property
    def native_value(self):
        if not self.coordinator.last_update_success: return "error"
        return ((self.coordinator.data or {}).get("status") or {}).get("model", "starting")

    @property
    def extra_state_attributes(self):
        d = self.coordinator.data or {}; s = d.get("status") or {}
        err = self.coordinator.last_exception
        return {"weather": s.get("weather"), "fitted": s.get("fitted"), "fit_seconds": s.get("fit_seconds"),
                "neighbour_prices_until": s.get("neighbour_prices_until"), "problem": str(err) if err and not self.coordinator.last_update_success else s.get("problem"),
                "updated": d.get("updated")}
