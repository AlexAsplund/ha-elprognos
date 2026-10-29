from __future__ import annotations
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from .const import AREAS, CONF_ADD, CONF_AREA, CONF_VAT, DOMAIN

def _schema(d):
    return {vol.Required(CONF_VAT, default=d.get(CONF_VAT, 25)): vol.Coerce(float),
            vol.Required(CONF_ADD, default=d.get(CONF_ADD, 0.0)): vol.Coerce(float)}

class Flow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input:
            await self.async_set_unique_id(user_input[CONF_AREA]); self._abort_if_unique_id_configured()
            return self.async_create_entry(title=f"Elprisprognos {user_input[CONF_AREA]}", data=user_input)
        return self.async_show_form(step_id="user", data_schema=vol.Schema({vol.Required(CONF_AREA, default="SE3"): vol.In(AREAS), **_schema({})}))

    @staticmethod
    @callback
    def async_get_options_flow(entry):
        return Options()

class Options(config_entries.OptionsFlow):
    async def async_step_init(self, user_input=None):
        if user_input: return self.async_create_entry(data=user_input)
        return self.async_show_form(step_id="init", data_schema=vol.Schema(_schema({**self.config_entry.data, **self.config_entry.options})))
