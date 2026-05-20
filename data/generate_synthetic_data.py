"""Generate non-IID synthetic meteorological data for 5 Indian coastal cities.

Each node receives 3 years (2021-2023) of daily observations with seasonal
monsoon rainfall, Oct-Nov / Apr-May cyclone events, and per-city distribution
shifts to simulate realistic federated heterogeneity.

Output schema matches ``data/fetch_data.py``::

    date, rainfall_mm, temperature_c, wind_speed_kmh, humidity_percent,
    pressure_hpa, flood_occurred, cyclone_occurred

Run from project root::

    python -m data.generate_synthetic_data
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

START = date(2021, 1, 1)
END = date(2023, 12, 31)

# Per-city parameters. Offsets create non-IID distributions across nodes.
CITIES = {
    "chennai": dict(
        seed=11,
        base_temp=29.0, temp_amp=4.0, temp_peak_doy=140,   # hottest in May
        monsoon_peak_doy=320, monsoon_width=45, rain_scale=18.0,  # NE monsoon (Oct-Dec)
        cyclone_months=(10, 11, 4, 5), cyclone_rate=0.60,
        base_pressure=1009.0, base_wind=11.0,
        flood_rain_3d=70.0,
    ),
    "mumbai": dict(
        seed=22,
        base_temp=27.5, temp_amp=3.5, temp_peak_doy=135,
        monsoon_peak_doy=210, monsoon_width=55, rain_scale=28.0,  # heavy SW monsoon
        cyclone_months=(5, 6, 10), cyclone_rate=0.25,
        base_pressure=1011.0, base_wind=10.0,
        flood_rain_3d=110.0,
    ),
    "kolkata": dict(
        seed=33,
        base_temp=26.5, temp_amp=6.5, temp_peak_doy=140,
        monsoon_peak_doy=200, monsoon_width=50, rain_scale=22.0,
        cyclone_months=(5, 10, 11), cyclone_rate=0.45,
        base_pressure=1010.0, base_wind=9.0,
        flood_rain_3d=90.0,
    ),
    "bhubaneswar": dict(
        seed=44,
        base_temp=27.0, temp_amp=5.5, temp_peak_doy=140,
        monsoon_peak_doy=210, monsoon_width=50, rain_scale=18.0,
        cyclone_months=(10, 11, 5), cyclone_rate=0.75,
        base_pressure=1008.0, base_wind=12.0,
        flood_rain_3d=80.0,
    ),
    "visakhapatnam": dict(
        seed=55,
        base_temp=28.0, temp_amp=4.5, temp_peak_doy=140,
        monsoon_peak_doy=240, monsoon_width=60, rain_scale=16.0,
        cyclone_months=(10, 11, 4, 5), cyclone_rate=0.90,
        base_pressure=1008.5, base_wind=12.5,
        flood_rain_3d=75.0,
    ),
}


def _seasonal_rain_mean(doy: int, peak: int, width: float, scale: float) -> float:
    # Gaussian bump on day-of-year, wraps via min distance.
    diff = min(abs(doy - peak), 365 - abs(doy - peak))
    return scale * math.exp(-(diff ** 2) / (2 * width ** 2))


def _sample_cyclone_days(rng: random.Random, params: dict) -> set[date]:
    """Place 2-5 day cyclone events inside the city's cyclone-prone months."""
    days: set[date] = set()
    d = START
    while d <= END:
        # cyclone_rate ≈ expected events per cyclone-month; ~30 days per month.
        if d.month in params["cyclone_months"] and rng.random() < params["cyclone_rate"] / 30.0:
            duration = rng.randint(2, 5)
            for k in range(duration):
                day = d + timedelta(days=k)
                if day <= END:
                    days.add(day)
            d += timedelta(days=duration + rng.randint(3, 10))
        else:
            d += timedelta(days=1)
    return days


def generate_city(name: str, params: dict, out_path: Path) -> dict:
    rng = random.Random(params["seed"])
    cyclone_days = _sample_cyclone_days(rng, params)

    rows: list[list] = []
    rain_hist: list[float] = []
    prev_temp_noise = 0.0
    prev_pres_noise = 0.0

    d = START
    while d <= END:
        doy = d.timetuple().tm_yday
        is_cyclone = d in cyclone_days

        # Rainfall: seasonal mean + gamma noise; amplified during cyclones.
        mean_rain = _seasonal_rain_mean(doy, params["monsoon_peak_doy"],
                                        params["monsoon_width"], params["rain_scale"])
        if mean_rain > 0.5:
            rain = rng.gammavariate(2.0, max(0.5, mean_rain / 2))
        else:
            rain = max(0.0, rng.gauss(0.3, 0.6))
        if is_cyclone:
            rain += rng.gammavariate(2.5, 25)

        # Temperature: annual cosine + AR(1) noise, lowered during cyclones.
        seasonal_t = params["base_temp"] + params["temp_amp"] * math.cos(
            2 * math.pi * (doy - params["temp_peak_doy"]) / 365
        )
        prev_temp_noise = 0.7 * prev_temp_noise + rng.gauss(0, 0.8)
        temp = seasonal_t + prev_temp_noise - (3.0 if is_cyclone else 0.0)

        # Pressure: baseline + AR(1) noise, sharp dip during cyclones.
        prev_pres_noise = 0.6 * prev_pres_noise + rng.gauss(0, 1.2)
        pressure = params["base_pressure"] + prev_pres_noise
        if is_cyclone:
            pressure -= rng.uniform(8, 20)

        # Wind: base + monsoon boost + cyclone spike.
        wind = params["base_wind"] + 0.05 * mean_rain + abs(rng.gauss(0, 3))
        if is_cyclone:
            wind += rng.uniform(40, 90)

        # Humidity: correlates with rainfall and inversely with temp anomaly.
        humidity = 60 + 0.6 * min(rain, 50) - 0.8 * (temp - params["base_temp"]) + rng.gauss(0, 4)
        humidity = max(20.0, min(100.0, humidity))

        # 3-day rolling rainfall for flood label.
        rain_hist.append(rain)
        if len(rain_hist) > 3:
            rain_hist.pop(0)
        rain_3d = sum(rain_hist)
        flood = int(rain_3d > params["flood_rain_3d"] or (is_cyclone and rain > 60))

        rows.append([
            d.isoformat(),
            round(rain, 1),
            round(temp, 1),
            round(wind, 1),
            round(humidity, 1),
            round(pressure, 1),
            flood,
            int(is_cyclone),
        ])
        d += timedelta(days=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "date", "rainfall_mm", "temperature_c", "wind_speed_kmh",
            "humidity_percent", "pressure_hpa", "flood_occurred", "cyclone_occurred",
        ])
        w.writerows(rows)

    n = len(rows)
    floods = sum(r[6] for r in rows)
    cyclones = sum(r[7] for r in rows)
    return dict(city=name, rows=n, floods=floods, cyclones=cyclones, path=str(out_path))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    clients_dir = project_root / "clients"
    print(f"Generating synthetic data for {len(CITIES)} nodes -> {clients_dir}")
    summary = []
    for city, params in CITIES.items():
        out = clients_dir / f"node_{city}.csv"
        summary.append(generate_city(city, params, out))

    print(f"\n{'city':<15} {'rows':>6} {'floods':>8} {'cyclones':>10}  file")
    for s in summary:
        print(
            f"{s['city']:<15} {s['rows']:>6} "
            f"{s['floods']:>4} ({100*s['floods']/s['rows']:>4.1f}%) "
            f"{s['cyclones']:>4} ({100*s['cyclones']/s['rows']:>4.1f}%)  {s['path']}"
        )


if __name__ == "__main__":
    main()
