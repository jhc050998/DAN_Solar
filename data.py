"""NSRDB download, parsing, and paper-style phase-space reconstruction."""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np


NSRDB_CSV_ENDPOINT = (
    "https://developer.nlr.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-aggregated-v4-0-0-download.csv"
)


@dataclass(frozen=True)
class MinMaxScaler:
    minimum: float
    maximum: float

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.minimum) / self.span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.span + self.minimum


def download_nsrdb_csv(
    destination: str | Path,
    *,
    api_key: str,
    email: str,
    latitude: float,
    longitude: float,
    year: int,
    full_name: str = "DAN replication",
    affiliation: str = "Independent research",
) -> Path:
    """Download one location/year of hourly GHI from the current NSRDB API."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "api_key": api_key,
        "wkt": f"POINT({longitude} {latitude})",
        "attributes": "ghi",
        "names": str(year),
        "utc": "false",
        "leap_day": "true",
        "interval": "60",
        "email": email,
        "full_name": full_name,
        "affiliation": affiliation,
        "reason": "Academic research",
        "mailing_list": "false",
    }
    url = f"{NSRDB_CSV_ENDPOINT}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "dan-nsrdb-replication/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NSRDB download failed ({exc.code}): {details[:500]}") from exc

    destination.write_bytes(payload)
    return destination


def read_nsrdb_ghi(path: str | Path) -> list[tuple[datetime, float]]:
    """Read NSRDB CSV despite its metadata line(s) before the table header."""
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if {"Year", "Month", "Day", "Hour", "Minute", "GHI"}.issubset(row)
        ),
        None,
    )
    if header_index is None:
        preview = text[:300].replace("\n", " ")
        raise ValueError(f"No NSRDB data header found in {path}. Response starts with: {preview}")

    header = rows[header_index]
    column = {name: header.index(name) for name in ("Year", "Month", "Day", "Hour", "Minute", "GHI")}
    result: list[tuple[datetime, float]] = []
    for row in rows[header_index + 1 :]:
        if not row or len(row) <= max(column.values()):
            continue
        try:
            timestamp = datetime(
                int(row[column["Year"]]),
                int(row[column["Month"]]),
                int(row[column["Day"]]),
                int(row[column["Hour"]]),
                int(row[column["Minute"]]),
            )
            ghi = float(row[column["GHI"]])
        except ValueError:
            continue
        result.append((timestamp, ghi))
    if not result:
        raise ValueError(f"No GHI records found in {path}")
    return result


def select_paper_window(
    records: list[tuple[datetime, float]],
    *,
    start_date: date,
    days: int = 122,
    first_hour: int = 6,
    last_hour: int = 19,
) -> np.ndarray:
    """Select consecutive local dates and retain 06:00--19:00 as in the paper."""
    end_date = start_date + timedelta(days=days)
    hourly: dict[tuple[date, int], list[float]] = {}
    for timestamp, ghi in records:
        if start_date <= timestamp.date() < end_date and first_hour <= timestamp.hour <= last_hour:
            hourly.setdefault((timestamp.date(), timestamp.hour), []).append(ghi)

    values: list[float] = []
    missing: list[str] = []
    for day_offset in range(days):
        current = start_date + timedelta(days=day_offset)
        for hour in range(first_hour, last_hour + 1):
            samples = hourly.get((current, hour))
            if samples:
                values.append(float(np.mean(samples)))
            else:
                missing.append(f"{current.isoformat()} {hour:02d}:00")
    if missing:
        raise ValueError(
            f"The selected window has {len(missing)} missing hourly values; first missing: {missing[0]}"
        )
    return np.asarray(values, dtype=np.float64)


def make_supervised(
    series: np.ndarray, *, delay: int, embedding_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create mu delayed inputs and a one-step-ahead target.

    This interpretation gives N-(mu-1)*delay-1 samples, exactly matching the
    sample counts in Table 1 of the paper.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    first_target_source = (embedding_dim - 1) * delay
    x, target = [], []
    for index in range(first_target_source, len(series) - 1):
        x.append([series[index - offset * delay] for offset in range(embedding_dim)])
        target.append(series[index + 1])
    return np.asarray(x, dtype=np.float64), np.asarray(target, dtype=np.float64)


def fit_scaler(x_train: np.ndarray, y_train: np.ndarray) -> MinMaxScaler:
    minimum = float(min(np.min(x_train), np.min(y_train)))
    maximum = float(max(np.max(x_train), np.max(y_train)))
    if maximum <= minimum:
        raise ValueError("Cannot scale a constant series")
    return MinMaxScaler(minimum=minimum, maximum=maximum)


def synthetic_irradiance(days: int = 122, seed: int = 42) -> np.ndarray:
    """Generate 14 daylight values/day for an offline end-to-end smoke test."""
    rng = np.random.default_rng(seed)
    hour = np.arange(14)
    daylight = np.maximum(0.0, np.sin(np.pi * (hour + 0.5) / 14.0))
    result = []
    cloud = 0.85
    for day in range(days):
        cloud = np.clip(0.75 * cloud + 0.25 * rng.uniform(0.35, 1.1), 0.25, 1.05)
        seasonal = 0.9 + 0.1 * np.sin(2.0 * np.pi * day / days)
        noise = rng.normal(0.0, 35.0, size=14)
        result.extend(np.maximum(0.0, 1050.0 * seasonal * cloud * daylight + noise))
    return np.asarray(result, dtype=np.float64)

