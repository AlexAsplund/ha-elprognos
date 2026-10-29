# Elprisprognos

Home Assistant integration that forecasts Nordic spot electricity prices (SE1–SE4) up to 7 days ahead.

## Install

HACS → Custom repositories → `https://github.com/AlexAsplund/ha-elprognos` (Integration), or copy `custom_components/elprisprognos` to your `config/custom_components/`. Restart, then add **Elprisprognos** under Settings → Devices & services. You choose the price area, the VAT (default 25 %) and any additional costs per kWh (grid fee, energy tax, …).

## Sensor

`sensor.elpris_prognos_<area>`: the state is the mean price of the first forecast day.

The `forecast` attribute holds one row per hour: `{start, value, spot, forecast}`.
- `value` is spot × (1 + VAT) + additional costs.
- `forecast` is false for prices that are already published and true for model output.

It works with apexcharts-card `data_generator`, or in automations that plan around the cheapest coming days.

## How it works

`prognos.py` holds the whole model, using stdlib + numpy.
- Prices: elprisetjustnu.se (from 2022-11).
- Weather: Open-Meteo.
  - Training uses archived forecasts; prediction uses the live forecast.
  - Inputs: wind at 100 m at 9 wind-farm sites (SE1–SE4, FI, NO2, DK1, N. Germany), temperature in Stockholm/Oslo/Helsinki, and sun over Germany/SE3.
- Model: ridge regression on hour/weekday/season + weather. Every feature is also scaled by the last 7 days' price level.
- The model refits once a day and the forecast updates hourly.

Backtest (SE3, 40 forecasts over a year, days 2–7 ahead): mean error 26 öre/kWh, against 40–42 öre for "same hour last week" or "the 7-day mean". Daily averages correlate 0.88 with the real prices. Training uses archived short-lead weather forecasts, so days 5–7 come out a bit optimistic.

## Scripts

- `python3 run.py backtest`: reproduces the numbers above.
- `python3 savings.py`: simulates a year of EV charging: plain charging on arrival, fixed overnight deadline, known prices only, a simple forecast rule for a charge-limit automation, full planning with the forecast, and perfect foresight. Hours the car is away can be priced out (`AWAY`) to model a commuter.
- `python3 run.py publish`: alternative to the integration. It posts `sensor.elpris_prognos` over the REST API and needs `HA_URL` and `HA_ACCESS_TOKEN` in the environment.

## Tests

```
uv run --python 3.13 --with pytest-homeassistant-custom-component --with numpy pytest -q
```
