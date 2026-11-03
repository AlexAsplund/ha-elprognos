# Elprisprognos

Home Assistant integration that forecasts Nordic spot electricity prices (SE1–SE4) up to 7 days ahead.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/forecast_dark.png">
  <img alt="Published Nord Pool prices followed by the 7-day forecast" src="docs/forecast_light.png" width="760">
</picture>

## Install

HACS → Custom repositories → `https://github.com/AlexAsplund/ha-elprognos` (Integration), or copy `custom_components/elprisprognos` to your `config/custom_components/`. Restart, then add **Elprisprognos** under Settings → Devices & services. You choose the price area, the VAT (default 25 %) and any additional costs per kWh (grid fee, energy tax, …).

## Sensor

`sensor.elpris_prognos_<area>`: the state is the mean price of the first forecast day.

The `forecast` attribute holds one row per hour: `{start, value, spot, forecast}`.
- `value` is spot × (1 + VAT) + additional costs.
- `forecast` is false for prices that are already published and true for model output.

It works with apexcharts-card `data_generator`, or in automations that plan around the cheapest coming days.

The chart above as an [apexcharts-card](https://github.com/RomRider/apexcharts-card):

```yaml
type: custom:apexcharts-card
header: {show: true, title: Elpris prognos SE3, show_states: true}
graph_span: 8d
span: {start: hour}
now: {show: true, label: Now}
yaxis: [{min: 0, decimals: 1}]
series:
  - entity: sensor.elpris_prognos_se3
    name: Published (Nord Pool)
    type: line
    curve: stepline
    color: "#fb8c00"
    show: {in_header: false}
    data_generator: |
      return entity.attributes.forecast.filter(e => !e.forecast).map(e => [new Date(e.start).getTime(), e.value]);
  - entity: sensor.elpris_prognos_se3
    name: Forecast
    type: line
    curve: stepline
    color: "#8e24aa"
    stroke_dash: 5
    data_generator: |
      return entity.attributes.forecast.filter(e => e.forecast).map(e => [new Date(e.start).getTime(), e.value]);
```

## How it works

`prognos.py` holds the model and the data fetching, `gbm.py` a small gradient-boosted-trees implementation; stdlib + numpy only.

Data (all keyless by default):
- Prices: elprisetjustnu.se (from 2022-11).
- Neighbour day-ahead prices (DE-LU, DK1, FI, NO1, NO2): energy-charts.info. Optionally set an ENTSO-E security token in the integration's options to fetch them from the ENTSO-E Transparency Platform instead (energy-charts is then the fallback).
- Weather: Open-Meteo, at the Swedish/Nordic wind, temperature and sun points plus 14 points in the neighbour zones (wind, temperature, sun), in one request. If it is down, the live forecast falls back to MET Norway (api.met.no) for every point: 10 m wind × 1.8 and clear-sky sun × cloud cover, which gives the same 7-day forecast within ~12 öre.
  - Training uses archived forecasts; prediction uses the live forecast.
- Outages: Nord Pool UMM (REMIT messages, no API key): Swedish reactors, transmission limits on the SE3 links (SE2/SE4/NO1/FI/DK1) and Finnish reactors. Training replays every message version as it was known when each forecast would have been made.

Model (v2):
- Stage 1: one tree model per neighbour zone forecasts its price from its own weather, the calendar and its recent price level.
- Stage 2: a tree model (Huber loss, monotone in wind, temperature and outages) forecasts the Swedish price from the stage-1 forecasts, the weather, hour/weekday/holiday/season, the recent Swedish and neighbour price levels, and MW of reactors and lines out. Each training hour appears once per lead time 2–7 days, with the price levels and outages as they were known that many days before.
- If the neighbour prices are unavailable, the integration logs a warning and uses the v1 ridge regression (hour/weekday/season + weather + reactors out, every feature also scaled by the last 7 days' price level).
- The models refit daily, on the first update after 02:00 (the v2 fit takes ~2 min on one desktop CPU core and ~0.5 GB RAM at its peak; a Raspberry Pi is several times slower: lower `ROUNDS`/`LEADS` in `prognos.py` if that hurts). The forecast updates hourly with the last fit. Models, the weather history, the neighbour prices and the last forecast are kept in `.storage/elprisprognos/`, so a restart neither refits nor waits for data.
- Tuned on SE3; SE1, SE2 and SE4 use the same inputs (the same neighbours and SE3 links).

Backtest (SE3, the same 40 forecasts over a year, days 2–7 ahead): mean error 19.9 öre/kWh for v2 against 25.0 for the ridge model, and 40–41 öre for "same hour last week" or "the 7-day mean". Daily averages correlate 0.88 with the real prices (ridge 0.86). Training uses archived short-lead weather forecasts, so days 5–7 come out a bit optimistic. Outages and neighbour price levels are replayed as they were known when each forecast was made.

Licences: energy-charts.info data is for private and internal use only. The integration fetches it on each install for its owner's own use and doesn't redistribute it; use the ENTSO-E option if that doesn't fit your use. Open-Meteo (CC BY 4.0), MET Norway (CC BY 4.0), Nord Pool UMM and ENTSO-E data are under their providers' terms.

## Scripts

- `python3 run.py backtest`: reproduces the numbers above (v2 and ridge; `WORKERS=n` runs the cuts in parallel, each v2 fit takes a couple of minutes).
- `python3 savings.py`: simulates a year of EV charging: plain charging on arrival, fixed overnight deadline, known prices only, a simple forecast rule for a charge-limit automation, full planning with the forecast, and perfect foresight. Hours the car is away can be priced out (`AWAY`) to model a commuter.
- `python3 run.py publish`: alternative to the integration. It posts `sensor.elpris_prognos` over the REST API and needs `HA_URL` and `HA_ACCESS_TOKEN` in the environment.

## Tests

```
uv run --python 3.13 --with pytest-homeassistant-custom-component --with numpy pytest -q
```
