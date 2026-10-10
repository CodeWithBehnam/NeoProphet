"""Temperature for any location as a forecasting regressor, from Open-Meteo.

``fetch_temperature`` returns one temperature per day (or per hour) for a
date range, ready to be merged onto a Prophet dataframe and used with
``add_regressor``. Past dates come from the archive (reanalysis) API, recent
days and the next 16 days from the forecast API, and anything further ahead
is filled with a climatology: the mean for that calendar day (and hour) over
the previous five full years. Each row says which of the three it is in its
``source`` column, so the caller can tell observations from projections.

Example
-------
::

    from prophet.weather import fetch_temperature

    result = fetch_temperature("London", "2025-01-01", "2026-12-31")
    print(result.location)                # Location(name='London, ...', ...)
    print(result.frame.groupby("source").size())
    df = df.merge(result.frame[["ds", "temperature"]], on="ds", how="left")

Terms of use
------------
The Open-Meteo free API is for non-commercial use and subject to fair use
limits (see https://open-meteo.com/en/terms). Commercial users need one of
the paid API plans. Responses are cached on disk to keep the request count
low: set ``NEOPROPHET_CACHE_DIR`` or pass ``cache_dir`` to choose where.

Design notes
------------
The module follows a functional core, imperative shell split. URL building,
response parsing, source planning, merging, gap detection and climatology are
pure functions. HTTP and the disk cache live only in ``_default_fetch``,
``_get_json`` and ``fetch_temperature``. Tests inject ``fetch`` and never
touch the network.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("prophet")

__all__ = ["Location", "Source", "WeatherError", "WeatherResult", "fetch_temperature"]

Freq = Literal["daily", "hourly"]
Source = Literal["archive", "forecast", "climatology"]

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_DIR_ENV = "NEOPROPHET_CACHE_DIR"

# The forecast API accepts 0 to 16 forecast days (checked against the live API).
FORECAST_DAYS = 16
# The archive lags a few days behind today; a week of forecast history bridges it.
PAST_DAYS = 7
CLIMATOLOGY_YEARS = 5
# Archive data this close to today can still be revised or contain nulls.
ARCHIVE_SETTLED_DAYS = 45
FORECAST_TTL_S = 3 * 3600
REQUEST_TIMEOUT_S = 30.0
GEOCODING_CANDIDATES = 10
USER_AGENT = (
    "neoprophet-weather/1.0 (Prophet fork; https://pypi.org/project/neoprophet/)"
)

_VARIABLE: dict[str, str] = {"daily": "temperature_2m_mean", "hourly": "temperature_2m"}


@dataclasses.dataclass(frozen=True)
class Location:
    """A resolved place: display name, coordinates and IANA timezone."""

    name: str
    latitude: float
    longitude: float
    timezone: str


@dataclasses.dataclass(frozen=True)
class WeatherResult:
    """Temperatures for a range plus the location they belong to.

    ``frame`` has columns ``ds`` (tz-naive local timestamps), ``temperature``
    (degrees Celsius) and ``source`` (``"archive"``, ``"forecast"`` or
    ``"climatology"``), with one row per day or hour and no gaps.
    """

    location: Location
    frame: pd.DataFrame


class WeatherError(RuntimeError):
    """Network, HTTP or response-format failure while fetching weather data."""


@dataclasses.dataclass(frozen=True)
class SourcePlan:
    """Which API calls a range needs. Ranges are inclusive local dates."""

    archive: tuple[date, date] | None
    forecast: bool


# --- URL building (pure) -----------------------------------------------------


def _query(params: dict[str, str | int | float]) -> str:
    return urllib.parse.urlencode(params)


def build_geocoding_url(name: str, count: int = 1) -> str:
    return f"{GEOCODING_URL}?{_query({'name': name, 'count': count, 'language': 'en', 'format': 'json'})}"


def build_archive_url(
    latitude: float, longitude: float, start: date, end: date, freq: Freq
) -> str:
    params: dict[str, str | int | float] = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        freq: _VARIABLE[freq],
        "timezone": "auto",
    }
    return f"{ARCHIVE_URL}?{_query(params)}"


def build_forecast_url(
    latitude: float,
    longitude: float,
    freq: Freq,
    past_days: int = PAST_DAYS,
    forecast_days: int = FORECAST_DAYS,
) -> str:
    params: dict[str, str | int | float] = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        freq: _VARIABLE[freq],
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    return f"{FORECAST_URL}?{_query(params)}"


# --- Range handling and planning (pure) --------------------------------------


def _naive(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _floor_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, ts.day, ts.hour)


def expected_index(
    start: str | pd.Timestamp, end: str | pd.Timestamp, freq: Freq
) -> pd.DatetimeIndex:
    """The timestamps the result must contain, tz-naive and inclusive.

    Hourly ranges whose ``end`` is a bare date (midnight) run to 23:00 of that
    day, because "until 5 May" almost always means the whole of 5 May.
    """
    if freq not in _VARIABLE:
        raise ValueError(f"freq must be 'daily' or 'hourly', got {freq!r}")
    first, last = _naive(start), _naive(end)
    if last < first:
        raise ValueError(f"end ({last}) is before start ({first})")
    if freq == "daily":
        index = pd.date_range(first.normalize(), last.normalize(), freq="D")
    else:
        if last == last.normalize():
            last = last + pd.Timedelta(hours=23)
        index = pd.date_range(
            _floor_hour(first), _floor_hour(last), freq=pd.Timedelta(hours=1)
        )
    return pd.DatetimeIndex(index.astype("datetime64[ns]"))


def plan_sources(grid: pd.DatetimeIndex, today: date) -> SourcePlan:
    """Decide which APIs to call for ``grid``, given today's date.

    The archive is asked only up to yesterday, since it rejects dates past
    today and its newest days are often null. The forecast call always spans
    ``PAST_DAYS`` back and ``FORECAST_DAYS`` ahead, so it is requested
    whenever the range touches that window.
    """
    first, last = grid[0].date(), grid[-1].date()
    archive_limit = today - timedelta(days=1)
    archive = (first, min(last, archive_limit)) if first <= archive_limit else None
    forecast = last >= today - timedelta(days=PAST_DAYS) and first <= today + timedelta(
        days=FORECAST_DAYS
    )
    return SourcePlan(archive=archive, forecast=forecast)


def archive_ttl(end: date, today: date) -> float | None:
    """Cache lifetime in seconds for an archive request; ``None`` is forever."""
    settled = end <= today - timedelta(days=ARCHIVE_SETTLED_DAYS)
    return None if settled else float(FORECAST_TTL_S)


def climatology_history_range(today: date) -> tuple[date, date]:
    """The previous ``CLIMATOLOGY_YEARS`` full calendar years."""
    return (
        date(today.year - CLIMATOLOGY_YEARS, 1, 1),
        date(today.year - 1, 12, 31),
    )


# --- Response parsing (pure) -------------------------------------------------


def describe_http_error(status: int, body: bytes) -> str:
    """An actionable message for a failed Open-Meteo HTTP response."""
    reason = ""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("reason"), str):
            reason = f": {parsed['reason']}"
    except (ValueError, UnicodeDecodeError):
        pass
    if status == 429:
        return (
            "Open-Meteo rate limit reached (HTTP 429). Wait a few minutes and "
            "retry; cached responses are reused, so repeated runs get cheaper. "
            "Heavy or commercial use needs an API plan: https://open-meteo.com/en/pricing"
        )
    if status in (400, 422):
        return f"Open-Meteo rejected the request (HTTP {status}){reason}"
    return f"Open-Meteo returned HTTP {status}{reason}. Try again later."


def parse_json_object(body: bytes, url: str) -> dict[str, Any]:
    """Decode a response body, turning API error payloads into WeatherError."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise WeatherError(f"Open-Meteo returned invalid JSON for {url}") from exc
    if not isinstance(payload, dict):
        raise WeatherError(f"Open-Meteo returned an unexpected payload for {url}")
    if payload.get("error"):
        raise WeatherError(
            f"Open-Meteo reported an error: {payload.get('reason', 'unknown reason')}"
        )
    return payload


def parse_geocoding(
    payload: dict[str, Any], query: str, qualifier: str | None = None
) -> Location:
    """Pick the best geocoding hit. ``qualifier`` narrows by country or region."""
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise WeatherError(
            f"No location found for {query!r}. Try the city name alone, another "
            "spelling, a larger nearby town, or pass (latitude, longitude)."
        )
    candidates = [r for r in results if isinstance(r, dict)]
    if qualifier:
        wanted = qualifier.strip().lower()
        matching = [
            r
            for r in candidates
            if wanted
            in {
                str(r.get(key, "")).lower()
                for key in ("country", "country_code", "admin1", "admin2")
            }
        ]
        if not matching:
            raise WeatherError(
                f"No location found for {query!r}: {len(candidates)} places match "
                f"{query.split(',')[0].strip()!r} but none in {qualifier!r}. Try the "
                "city name alone or pass (latitude, longitude)."
            )
        candidates = matching
    hit = candidates[0]
    try:
        label = str(hit["name"])
        country = hit.get("country")
        return Location(
            name=f"{label}, {country}" if country else label,
            latitude=float(hit["latitude"]),
            longitude=float(hit["longitude"]),
            timezone=str(hit.get("timezone") or "UTC"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError(f"Unexpected geocoding response for {query!r}") from exc


def parse_timezone(payload: dict[str, Any]) -> str:
    tz = payload.get("timezone")
    if not isinstance(tz, str) or not tz:
        raise WeatherError("Open-Meteo response is missing the 'timezone' field")
    return tz


def parse_series(payload: dict[str, Any], freq: Freq) -> pd.DataFrame:
    """Turn a daily or hourly block into ``ds, temperature``, dropping nulls."""
    block = payload.get(freq)
    variable = _VARIABLE[freq]
    if not isinstance(block, dict) or "time" not in block or variable not in block:
        raise WeatherError(
            f"Open-Meteo response has no {freq!r} block with 'time' and {variable!r}"
        )
    times, values = block["time"], block[variable]
    if not isinstance(times, list) or not isinstance(values, list):
        raise WeatherError(f"Malformed {freq!r} block in Open-Meteo response")
    if len(times) != len(values):
        raise WeatherError(
            f"Open-Meteo returned {len(times)} timestamps but {len(values)} values"
        )
    try:
        ds = pd.to_datetime(pd.Series(times, dtype="object"))
    except (ValueError, TypeError) as exc:
        raise WeatherError("Open-Meteo returned unparseable timestamps") from exc
    temperature = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
    frame = pd.DataFrame(
        {"ds": ds.astype("datetime64[ns]"), "temperature": temperature.astype(float)}
    )
    return frame.dropna(subset=["temperature"]).reset_index(drop=True)


# --- Merging, climatology and gaps (pure) ------------------------------------


def merge_sources(
    archive: pd.DataFrame | None, forecast: pd.DataFrame | None
) -> pd.DataFrame:
    """Combine observations, preferring the archive where both cover a time."""
    parts = [
        frame.assign(source=name)
        for name, frame in (("archive", archive), ("forecast", forecast))
        if frame is not None and not frame.empty
    ]
    if not parts:
        return pd.DataFrame(
            {
                "ds": pd.Series(dtype="datetime64[ns]"),
                "temperature": pd.Series(dtype=float),
                "source": pd.Series(dtype=object),
            }
        )
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset="ds", keep="first")
    return merged.sort_values("ds").reset_index(drop=True)


def split_missing(
    grid: pd.DatetimeIndex, observed_ds: pd.Series, today: date
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split timestamps not yet covered into (climatology targets, true gaps).

    Climatology may only extend the series forwards, past the last observed
    or forecast time and beyond today. Anything else missing is a real gap.
    """
    missing = grid.difference(pd.DatetimeIndex(observed_ds))
    if missing.empty:
        return missing, missing
    last_known = observed_ds.max() if len(observed_ds) else None
    beyond_today = pd.Timestamp(today + timedelta(days=1))
    extendable = missing >= beyond_today
    if last_known is not None:
        extendable &= missing > last_known
    return missing[extendable], missing[~extendable]


def _profile_keys(ds: pd.DatetimeIndex, freq: Freq) -> list[np.ndarray]:
    keys = [np.asarray(ds.month), np.asarray(ds.day)]
    if freq == "hourly":
        keys.append(np.asarray(ds.hour))
    return keys


def climatology_profile(history: pd.DataFrame, freq: Freq) -> pd.Series:
    """Mean temperature per (month, day[, hour]) over ``history``.

    Month and day are used instead of day of year so that a leap year does
    not shift every later date by one.
    """
    ds = pd.DatetimeIndex(history["ds"])
    names = ["month", "day"] + (["hour"] if freq == "hourly" else [])
    keyed = history["temperature"].groupby(_profile_keys(ds, freq))
    profile = keyed.mean()
    profile.index.names = names
    return profile


def climatology_rows(
    profile: pd.Series, targets: pd.DatetimeIndex, freq: Freq
) -> pd.DataFrame:
    """Look up each target timestamp in ``profile``; 29 February falls back to 28."""
    if targets.empty:
        return pd.DataFrame({"ds": targets, "temperature": [], "source": []})

    def lookup(day_shift_leap: bool) -> pd.Series:
        keys = _profile_keys(targets, freq)
        if day_shift_leap:
            leap = (keys[0] == 2) & (keys[1] == 29)
            keys[1] = np.where(leap, 28, keys[1])
        index = pd.MultiIndex.from_arrays(keys, names=profile.index.names)
        return pd.Series(profile.reindex(index).to_numpy(), index=range(len(targets)))

    values = lookup(False).fillna(lookup(True))
    if values.isna().any():
        raise WeatherError(
            "Climatology history does not cover every calendar day needed; "
            "the archive returned too little data for the previous years."
        )
    return pd.DataFrame(
        {
            "ds": targets.astype("datetime64[ns]"),
            "temperature": values.to_numpy(dtype=float),
            "source": "climatology",
        }
    )


def assemble(
    grid: pd.DatetimeIndex, observed: pd.DataFrame, climatology: pd.DataFrame | None
) -> pd.DataFrame:
    """Final frame: exactly the grid, sorted, with no gaps or duplicates."""
    parts = [observed] + ([climatology] if climatology is not None else [])
    combined = pd.concat(parts, ignore_index=True).drop_duplicates("ds", keep="first")
    combined = combined[combined["ds"].isin(grid)].sort_values("ds")
    result = combined.reset_index(drop=True)[["ds", "temperature", "source"]]
    if len(result) != len(grid):
        missing = grid.difference(pd.DatetimeIndex(result["ds"]))
        raise WeatherError(_gap_message(missing))
    return result


def _gap_message(gaps: pd.DatetimeIndex) -> str:
    shown = ", ".join(str(ts) for ts in gaps[:3])
    more = f" and {len(gaps) - 3} more" if len(gaps) > 3 else ""
    return (
        f"Open-Meteo has no temperature for {len(gaps)} requested timestamp(s) "
        f"({shown}{more}) and they are too recent to fill with climatology. "
        "Narrow the end date, or retry later when the archive has caught up."
    )


# --- Disk cache and HTTP (imperative shell) ----------------------------------


def resolve_cache_dir(cache_dir: Path | None, injected_fetch: bool) -> Path | None:
    """Where to cache, or ``None`` for no caching.

    An injected fetcher with no explicit directory is never cached, so test
    doubles cannot pollute the real cache under the URLs of the real API.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    if injected_fetch:
        return None
    env = os.environ.get(CACHE_DIR_ENV)
    return Path(env) if env else Path.home() / ".cache" / "neoprophet" / "weather"


def _cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:40]}.json"


def _read_cache(cache_dir: Path, url: str, ttl: float | None) -> bytes | None:
    path = _cache_path(cache_dir, url)
    try:
        if ttl is not None and time.time() - path.stat().st_mtime > ttl:
            return None
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Weather cache read failed for %s: %s", path, exc)
        return None


def _write_cache(cache_dir: Path, url: str, body: bytes) -> None:
    path = _cache_path(cache_dir, url)
    tmp_name: str | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write then rename so a concurrent reader never sees a partial file.
        with tempfile.NamedTemporaryFile(
            dir=cache_dir, delete=False, suffix=".tmp"
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(body)
        os.replace(tmp_name, path)
    except OSError as exc:
        logger.warning("Weather cache write failed for %s: %s", path, exc)
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _default_fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        raise WeatherError(f"Refusing to fetch non-HTTPS URL: {url}")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        raise WeatherError(describe_http_error(exc.code, body)) from exc
    except OSError as exc:
        raise WeatherError(
            f"Could not reach Open-Meteo ({exc}). Check your internet connection; "
            "cached data is used when available."
        ) from exc


def _get_json(
    url: str,
    ttl: float | None,
    *,
    fetch: Callable[[str], bytes],
    cache_dir: Path | None,
) -> dict[str, Any]:
    """Fetch ``url`` through the cache. ``ttl`` of ``None`` means keep forever."""
    if cache_dir is not None:
        cached = _read_cache(cache_dir, url, ttl)
        if cached is not None:
            try:
                return parse_json_object(cached, url)
            except WeatherError:
                logger.warning("Discarding unreadable weather cache entry for %s", url)
    logger.debug("Fetching %s", url)
    body = fetch(url)
    payload = parse_json_object(body, url)
    if cache_dir is not None:
        _write_cache(cache_dir, url, body)
    return payload


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


# --- Public entry point ------------------------------------------------------


def _split_place(text: str) -> tuple[str, str | None]:
    head, _, tail = text.partition(",")
    return head.strip(), (tail.strip() or None)


def _resolve_location(
    location: str | tuple[float, float],
    get_json: Callable[[str, float | None], dict[str, Any]],
) -> Location | tuple[float, float]:
    """A geocoded Location, or validated coordinates whose timezone is still unknown."""
    if isinstance(location, str):
        if not location.strip():
            raise ValueError("location must not be empty")
        name, qualifier = _split_place(location)
        count = GEOCODING_CANDIDATES if qualifier else 1
        payload = get_json(build_geocoding_url(name, count), None)
        return parse_geocoding(payload, location, qualifier)
    latitude, longitude = (float(location[0]), float(location[1]))
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(
            f"Coordinates out of range: latitude {latitude}, longitude {longitude}"
        )
    return (latitude, longitude)


def fetch_temperature(
    location: str | tuple[float, float],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    freq: Literal["daily", "hourly"] = "daily",
    cache_dir: Path | None = None,
    fetch: Callable[[str], bytes] | None = None,
) -> WeatherResult:
    """Temperature for ``location`` over ``[start, end]``, one row per day or hour.

    Parameters
    ----------
    location:
        A place name such as ``"London"`` or ``"Paris, France"`` (geocoded with
        Open-Meteo), or a ``(latitude, longitude)`` pair.
    start, end:
        Inclusive local date range. Daily rows are the mean 2 m temperature of
        each local day; hourly rows are the 2 m temperature at each hour.
    freq:
        ``"daily"`` or ``"hourly"``.
    cache_dir:
        Directory for cached API responses. Defaults to the environment
        variable ``NEOPROPHET_CACHE_DIR``, then ``~/.cache/neoprophet/weather``.
        Archive and geocoding responses are kept; forecasts and archive data
        from the last 45 days expire after three hours.
    fetch:
        Replacement for the HTTP layer, mapping a URL to response bytes. When
        given without ``cache_dir`` nothing is cached.

    Raises
    ------
    WeatherError
        On network, HTTP or response-format failures, an unknown place, or a
        gap that no source can fill.
    ValueError
        On invalid arguments (bad frequency, reversed range, bad coordinates).
    """
    grid = expected_index(start, end, freq)
    today = _utc_today()
    get_json = functools.partial(
        _get_json,
        fetch=fetch or _default_fetch,
        cache_dir=resolve_cache_dir(cache_dir, fetch is not None),
    )
    resolved = _resolve_location(location, get_json)
    if isinstance(resolved, Location):
        name, latitude, longitude = resolved.name, resolved.latitude, resolved.longitude
        timezone_name: str | None = resolved.timezone
    else:
        latitude, longitude = resolved
        name, timezone_name = f"{latitude:.2f}, {longitude:.2f}", None

    plan = plan_sources(grid, today)
    archive_frame: pd.DataFrame | None = None
    forecast_frame: pd.DataFrame | None = None
    if plan.archive is not None:
        first, last = plan.archive
        payload = get_json(
            build_archive_url(latitude, longitude, first, last, freq),
            archive_ttl(last, today),
        )
        archive_frame = parse_series(payload, freq)
        timezone_name = timezone_name or parse_timezone(payload)
    if plan.forecast:
        payload = get_json(
            build_forecast_url(latitude, longitude, freq), float(FORECAST_TTL_S)
        )
        forecast_frame = parse_series(payload, freq)
        timezone_name = timezone_name or parse_timezone(payload)

    observed = merge_sources(archive_frame, forecast_frame)
    targets, gaps = split_missing(grid, observed["ds"], today)
    if len(gaps):
        raise WeatherError(_gap_message(gaps))

    climatology: pd.DataFrame | None = None
    if len(targets):
        history_start, history_end = climatology_history_range(today)
        payload = get_json(
            build_archive_url(latitude, longitude, history_start, history_end, freq),
            None,
        )
        timezone_name = timezone_name or parse_timezone(payload)
        profile = climatology_profile(parse_series(payload, freq), freq)
        climatology = climatology_rows(profile, targets, freq)

    frame = assemble(grid, observed, climatology)
    counts = frame["source"].value_counts().to_dict()
    logger.info("Fetched %s temperature for %s: %s", freq, name, counts)
    return WeatherResult(
        location=Location(
            name=name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone_name or "UTC",
        ),
        frame=frame,
    )
