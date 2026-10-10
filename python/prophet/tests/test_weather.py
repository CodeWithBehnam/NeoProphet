"""Tests for prophet.weather. No test touches the network."""

import dataclasses
import io
import json
import os
import urllib.error
import urllib.parse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from prophet import weather
from prophet.weather import (
    Location,
    WeatherError,
    WeatherResult,
    archive_ttl,
    assemble,
    build_archive_url,
    build_forecast_url,
    build_geocoding_url,
    climatology_history_range,
    climatology_profile,
    climatology_rows,
    describe_http_error,
    expected_index,
    fetch_temperature,
    merge_sources,
    parse_geocoding,
    parse_json_object,
    parse_series,
    parse_timezone,
    plan_sources,
    resolve_cache_dir,
    split_missing,
)

TODAY = date(2026, 6, 15)


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather, "_utc_today", lambda: TODAY)


# --- fake Open-Meteo ---------------------------------------------------------

PLACES = {
    "london": {
        "name": "London",
        "latitude": 51.50853,
        "longitude": -0.12574,
        "country": "United Kingdom",
        "country_code": "GB",
        "admin1": "England",
        "timezone": "Europe/London",
    },
}


def observed_temp(ts: pd.Timestamp) -> float:
    """A smooth seasonal value that also drifts by year, so averaging is visible."""
    return round(10 + ts.dayofyear / 20 + (ts.year - 2000) / 10 + ts.hour / 100, 3)


class FakeApi:
    """Serves canned JSON shaped like the real Open-Meteo responses."""

    def __init__(self, timezone: str = "Europe/London", archive_lag_days: int = 3):
        self.timezone = timezone
        self.archive_lag_days = archive_lag_days
        self.urls: list[str] = []

    def calls(self, host_prefix: str) -> list[str]:
        return [
            u
            for u in self.urls
            if urllib.parse.urlsplit(u).netloc.startswith(host_prefix)
        ]

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        parts = urllib.parse.urlsplit(url)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parts.query).items()}
        if parts.netloc.startswith("geocoding"):
            hits = [p for key, p in PLACES.items() if key == q["name"].strip().lower()]
            return json.dumps({"results": hits} if hits else {}).encode()
        freq = "daily" if "daily" in q else "hourly"
        if parts.netloc.startswith("archive"):
            start, end = pd.Timestamp(q["start_date"]), pd.Timestamp(q["end_date"])
            if end.date() > TODAY:
                return self._error(
                    f"Parameter 'end_date' is out of allowed range to {TODAY}"
                )
            last_valid = pd.Timestamp(TODAY - timedelta(days=self.archive_lag_days))
            return self._block(start, end, freq, last_valid, offset=0.0)
        if parts.netloc.startswith("api"):
            days = int(q["forecast_days"])
            if days > 16:
                return self._error("Forecast days is invalid. Allowed range 0 to 16.")
            start = pd.Timestamp(TODAY - timedelta(days=int(q["past_days"])))
            end = pd.Timestamp(TODAY + timedelta(days=days - 1))
            return self._block(start, end, freq, end, offset=0.5)
        raise AssertionError(f"unexpected URL {url}")

    @staticmethod
    def _error(reason: str) -> bytes:
        return json.dumps({"error": True, "reason": reason}).encode()

    def _block(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str,
        last_valid: pd.Timestamp,
        offset: float,
    ) -> bytes:
        if freq == "daily":
            times = pd.date_range(start, end, freq="D")
            fmt, key = "%Y-%m-%d", "temperature_2m_mean"
        else:
            times = pd.date_range(
                start, end + pd.Timedelta(hours=23), freq=pd.Timedelta(hours=1)
            )
            fmt, key = "%Y-%m-%dT%H:%M", "temperature_2m"
        values = [
            None if ts.normalize() > last_valid else observed_temp(ts) + offset
            for ts in times
        ]
        return json.dumps(
            {
                "latitude": 51.5,
                "longitude": -0.12,
                "timezone": self.timezone,
                f"{freq}_units": {"time": "iso8601", key: "°C"},
                freq: {"time": [t.strftime(fmt) for t in times], key: values},
            }
        ).encode()


# --- URL building ------------------------------------------------------------


def test_geocoding_url_encodes_name():
    url = build_geocoding_url("São Paulo", count=1)
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert url.startswith("https://geocoding-api.open-meteo.com/v1/search?")
    assert q["name"] == ["São Paulo"]
    assert q["count"] == ["1"]


def test_archive_url_daily_and_hourly():
    daily = build_archive_url(
        51.50853, -0.12574, date(2025, 1, 1), date(2025, 1, 31), "daily"
    )
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(daily).query)
    assert daily.startswith("https://archive-api.open-meteo.com/v1/archive?")
    assert q["daily"] == ["temperature_2m_mean"]
    assert q["start_date"] == ["2025-01-01"]
    assert q["end_date"] == ["2025-01-31"]
    assert q["timezone"] == ["auto"]
    assert q["latitude"] == ["51.5085"]
    hourly = build_archive_url(1.0, 2.0, date(2025, 1, 1), date(2025, 1, 2), "hourly")
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(hourly).query)["hourly"] == [
        "temperature_2m"
    ]


def test_forecast_url_bridges_archive_lag_and_reaches_16_days():
    url = build_forecast_url(40.71, -74.01, "hourly")
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert url.startswith("https://api.open-meteo.com/v1/forecast?")
    assert q["past_days"] == [str(weather.PAST_DAYS)]
    assert q["forecast_days"] == ["16"]
    assert q["hourly"] == ["temperature_2m"]
    assert q["timezone"] == ["auto"]


# --- parsing -----------------------------------------------------------------


def _daily_payload(times: list[str], values: list[Any]) -> dict[str, Any]:
    return {
        "timezone": "Europe/London",
        "daily": {"time": times, "temperature_2m_mean": values},
    }


def test_parse_series_drops_nulls_and_returns_naive_floats():
    frame = parse_series(
        _daily_payload(["2026-06-01", "2026-06-02", "2026-06-03"], [10, None, 12.5]),
        "daily",
    )
    assert list(frame.columns) == ["ds", "temperature"]
    assert frame["ds"].dt.tz is None
    assert frame["temperature"].tolist() == [10.0, 12.5]
    assert frame["ds"].dt.day.tolist() == [1, 3]


def test_parse_series_hourly():
    payload = {
        "hourly": {
            "time": ["2026-06-01T00:00", "2026-06-01T01:00"],
            "temperature_2m": [1.5, 2.5],
        }
    }
    frame = parse_series(payload, "hourly")
    assert frame["ds"].dt.hour.tolist() == [0, 1]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"daily": {"time": ["2026-06-01"]}},
        {"daily": {"time": ["2026-06-01", "2026-06-02"], "temperature_2m_mean": [1.0]}},
        {"daily": {"time": "x", "temperature_2m_mean": "y"}},
    ],
)
def test_parse_series_rejects_malformed_payloads(payload: dict[str, Any]):
    with pytest.raises(WeatherError):
        parse_series(payload, "daily")


def test_parse_json_object_surfaces_api_error_reason():
    body = json.dumps({"error": True, "reason": "Forecast days is invalid"}).encode()
    with pytest.raises(WeatherError, match="Forecast days is invalid"):
        parse_json_object(body, "u")
    with pytest.raises(WeatherError, match="invalid JSON"):
        parse_json_object(b"<html>", "u")
    with pytest.raises(WeatherError, match="unexpected payload"):
        parse_json_object(b"[1]", "u")


def test_parse_timezone_requires_field():
    assert parse_timezone({"timezone": "Asia/Tokyo"}) == "Asia/Tokyo"
    with pytest.raises(WeatherError, match="timezone"):
        parse_timezone({})


def test_parse_geocoding_builds_location():
    loc = parse_geocoding({"results": [PLACES["london"]]}, "London")
    assert loc == Location(
        "London, United Kingdom", 51.50853, -0.12574, "Europe/London"
    )


def test_parse_geocoding_no_match_lists_what_to_try():
    with pytest.raises(WeatherError, match="Try the city name alone"):
        parse_geocoding({}, "Zzzqqxx")


def test_parse_geocoding_qualifier_filters_by_country_or_region():
    paris_fr = {
        "name": "Paris",
        "latitude": 48.85,
        "longitude": 2.35,
        "country": "France",
        "country_code": "FR",
        "timezone": "Europe/Paris",
    }
    paris_tx = {
        "name": "Paris",
        "latitude": 33.66,
        "longitude": -95.55,
        "country": "United States",
        "country_code": "US",
        "admin1": "Texas",
        "timezone": "America/Chicago",
    }
    payload = {"results": [paris_tx, paris_fr]}
    assert (
        parse_geocoding(payload, "Paris, France", "France").timezone == "Europe/Paris"
    )
    assert (
        parse_geocoding(payload, "Paris, Texas", "texas").timezone == "America/Chicago"
    )
    with pytest.raises(WeatherError, match="none in 'Peru'"):
        parse_geocoding(payload, "Paris, Peru", "Peru")


def test_describe_http_error():
    assert "rate limit" in describe_http_error(429, b"")
    assert "bad thing" in describe_http_error(400, b'{"reason": "bad thing"}')
    assert "HTTP 503" in describe_http_error(503, b"not json")


# --- ranges and planning -----------------------------------------------------


def test_expected_index_daily_normalises_and_drops_timezone():
    idx = expected_index(
        pd.Timestamp("2026-01-01 15:00", tz="UTC"), "2026-01-04 08:00", "daily"
    )
    assert idx.tz is None
    assert idx.tolist() == list(pd.date_range("2026-01-01", "2026-01-04"))


def test_expected_index_hourly_bare_end_date_covers_whole_day():
    idx = expected_index("2026-01-01", "2026-01-02", "hourly")
    assert len(idx) == 48
    assert idx[-1] == pd.Timestamp("2026-01-02 23:00")


def test_expected_index_hourly_with_explicit_end_time_is_exact():
    idx = expected_index("2026-01-01 03:30", "2026-01-01 06:10", "hourly")
    assert idx.tolist() == list(
        pd.date_range("2026-01-01 03:00", "2026-01-01 06:00", freq="h")
    )


def test_expected_index_validates_arguments():
    with pytest.raises(ValueError, match="before start"):
        expected_index("2026-01-02", "2026-01-01", "daily")
    with pytest.raises(ValueError, match="freq"):
        expected_index("2026-01-01", "2026-01-02", "weekly")  # type: ignore[arg-type]


def test_plan_sources_past_only_skips_forecast():
    plan = plan_sources(expected_index("2025-01-01", "2025-12-31", "daily"), TODAY)
    assert plan.archive == (date(2025, 1, 1), date(2025, 12, 31))
    assert plan.forecast is False


def test_plan_sources_spanning_range_uses_both_and_clamps_archive():
    plan = plan_sources(expected_index("2026-01-01", "2026-12-31", "daily"), TODAY)
    assert plan.archive == (date(2026, 1, 1), TODAY - timedelta(days=1))
    assert plan.forecast is True


def test_plan_sources_far_future_needs_neither():
    plan = plan_sources(expected_index("2027-01-01", "2027-01-31", "daily"), TODAY)
    assert plan.archive is None
    assert plan.forecast is False


def test_archive_ttl_is_forever_only_when_settled():
    assert archive_ttl(TODAY - timedelta(days=60), TODAY) is None
    assert archive_ttl(TODAY - timedelta(days=5), TODAY) == weather.FORECAST_TTL_S


def test_climatology_history_range_is_five_full_years():
    assert climatology_history_range(TODAY) == (date(2021, 1, 1), date(2025, 12, 31))


# --- merging, gaps, climatology ----------------------------------------------


def _frame(days: list[str], temps: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ds": pd.to_datetime(days).astype("datetime64[ns]"), "temperature": temps}
    )


def test_merge_prefers_archive_over_forecast_and_sorts():
    archive = _frame(["2026-06-01", "2026-06-02"], [1.0, 2.0])
    forecast = _frame(["2026-06-02", "2026-06-03"], [99.0, 3.0])
    merged = merge_sources(archive, forecast)
    assert merged["ds"].is_unique
    assert merged["source"].tolist() == ["archive", "archive", "forecast"]
    assert merged["temperature"].tolist() == [1.0, 2.0, 3.0]


def test_merge_handles_missing_inputs():
    assert merge_sources(None, None).empty
    only = merge_sources(None, _frame(["2026-06-01"], [1.0]))
    assert only["source"].tolist() == ["forecast"]


def test_split_missing_extends_only_forwards_beyond_today():
    grid = expected_index("2026-06-10", "2026-07-20", "daily")
    observed = _frame(["2026-06-10", "2026-06-11", "2026-07-01"], [1, 2, 3])["ds"]
    targets, gaps = split_missing(grid, observed, TODAY)
    # 12 to 30 June fall between known points: real gaps. Only after 1 July is climatology.
    assert targets.min() == pd.Timestamp("2026-07-02")
    assert targets.max() == pd.Timestamp("2026-07-20")
    assert gaps.min() == pd.Timestamp("2026-06-12")
    assert gaps.max() == pd.Timestamp("2026-06-30")


def test_split_missing_past_tail_is_a_gap_not_climatology():
    grid = expected_index("2026-06-01", "2026-06-14", "daily")
    observed = _frame(["2026-06-01", "2026-06-02"], [1, 2])["ds"]
    targets, gaps = split_missing(grid, observed, TODAY)
    assert len(targets) == 0
    assert len(gaps) == 12


def test_split_missing_with_nothing_observed_uses_climatology_for_the_future():
    grid = expected_index("2027-01-01", "2027-01-03", "daily")
    targets, gaps = split_missing(grid, pd.Series([], dtype="datetime64[ns]"), TODAY)
    assert len(targets) == 3 and len(gaps) == 0


def test_climatology_daily_is_mean_over_years_by_month_and_day():
    history = _frame(["2024-03-05", "2025-03-05", "2025-03-06"], [10.0, 20.0, 7.0])
    profile = climatology_profile(history, "daily")
    rows = climatology_rows(
        profile, pd.DatetimeIndex(["2027-03-05", "2027-03-06"]), "daily"
    )
    assert rows["temperature"].tolist() == [15.0, 7.0]
    assert set(rows["source"]) == {"climatology"}


def test_climatology_hourly_keys_include_the_hour():
    history = _frame(
        ["2025-03-05 00:00", "2025-03-05 12:00", "2024-03-05 12:00"], [1.0, 10.0, 20.0]
    )
    rows = climatology_rows(
        climatology_profile(history, "hourly"),
        pd.DatetimeIndex(["2027-03-05 00:00", "2027-03-05 12:00"]),
        "hourly",
    )
    assert rows["temperature"].tolist() == [1.0, 15.0]


def test_climatology_leap_day_falls_back_to_28_february():
    history = _frame(["2025-02-28", "2025-03-01"], [5.0, 9.0])
    rows = climatology_rows(
        climatology_profile(history, "daily"), pd.DatetimeIndex(["2028-02-29"]), "daily"
    )
    assert rows["temperature"].tolist() == [5.0]


def test_climatology_uses_real_leap_day_when_history_has_one():
    history = _frame(["2024-02-29", "2024-02-28"], [1.0, 2.0])
    rows = climatology_rows(
        climatology_profile(history, "daily"), pd.DatetimeIndex(["2028-02-29"]), "daily"
    )
    assert rows["temperature"].tolist() == [1.0]


def test_climatology_raises_when_history_is_too_thin():
    profile = climatology_profile(_frame(["2025-03-05"], [1.0]), "daily")
    with pytest.raises(WeatherError, match="Climatology history"):
        climatology_rows(profile, pd.DatetimeIndex(["2027-07-01"]), "daily")


def test_assemble_detects_gaps_and_duplicates():
    grid = expected_index("2026-06-01", "2026-06-03", "daily")
    observed = merge_sources(_frame(["2026-06-01", "2026-06-03"], [1.0, 3.0]), None)
    with pytest.raises(WeatherError, match="1 requested timestamp"):
        assemble(grid, observed, None)
    clim = pd.DataFrame(
        {
            "ds": pd.DatetimeIndex(["2026-06-02", "2026-06-03"]).astype(
                "datetime64[ns]"
            ),
            "temperature": [2.0, 77.0],
            "source": "climatology",
        }
    )
    out = assemble(grid, observed, clim)
    assert out["temperature"].tolist() == [1.0, 2.0, 3.0]
    assert out["ds"].is_unique


# --- end to end with a fake API ----------------------------------------------


def test_spanning_range_uses_all_three_sources_without_gaps_or_duplicates():
    api = FakeApi()
    result = fetch_temperature("London", "2026-03-01", "2026-08-15", fetch=api)
    frame = result.frame
    assert isinstance(result, WeatherResult)
    assert result.location == Location(
        "London, United Kingdom", 51.50853, -0.12574, "Europe/London"
    )
    assert list(frame.columns) == ["ds", "temperature", "source"]
    assert frame["ds"].tolist() == list(pd.date_range("2026-03-01", "2026-08-15"))
    assert frame["ds"].is_unique
    assert frame["temperature"].notna().all()
    assert frame["ds"].dt.tz is None
    by_source = frame.groupby("source")["ds"].agg(["min", "max", "size"])
    # Archive is asked up to yesterday but only has data up to today - 3 days.
    assert by_source.loc["archive", "max"] == pd.Timestamp(TODAY - timedelta(days=3))
    assert by_source.loc["forecast", "min"] == pd.Timestamp(TODAY - timedelta(days=2))
    assert by_source.loc["forecast", "max"] == pd.Timestamp(TODAY + timedelta(days=15))
    assert by_source.loc["climatology", "min"] == pd.Timestamp(
        TODAY + timedelta(days=16)
    )
    assert by_source.loc["climatology", "max"] == pd.Timestamp("2026-08-15")
    # Sources are contiguous blocks in time order.
    order = frame["source"].ne(frame["source"].shift()).sum()
    assert order == 3


def test_archive_values_win_where_forecast_overlaps():
    result = fetch_temperature("London", "2026-06-05", "2026-06-12", fetch=FakeApi())
    frame = result.frame.set_index("ds")
    day = pd.Timestamp("2026-06-10")
    assert frame.loc[day, "source"] == "archive"
    assert frame.loc[day, "temperature"] == observed_temp(day)


def test_climatology_values_average_previous_five_years():
    result = fetch_temperature("London", "2027-04-10", "2027-04-12", fetch=FakeApi())
    assert set(result.frame["source"]) == {"climatology"}
    expected = (
        sum(
            observed_temp(pd.Timestamp(year=y, month=4, day=11))
            for y in range(2021, 2026)
        )
        / 5
    )
    row = result.frame.set_index("ds").loc[pd.Timestamp("2027-04-11")]
    assert row["temperature"] == pytest.approx(expected)


def test_far_future_range_never_calls_forecast():
    api = FakeApi()
    fetch_temperature("London", "2027-04-10", "2027-04-12", fetch=api)
    assert api.calls("api.") == []
    history = api.calls("archive")
    assert len(history) == 1
    assert "start_date=2021-01-01" in history[0] and "end_date=2025-12-31" in history[0]


def test_past_only_range_calls_archive_once_and_not_forecast():
    api = FakeApi()
    result = fetch_temperature("London", "2025-01-01", "2025-01-31", fetch=api)
    assert set(result.frame["source"]) == {"archive"}
    assert len(result.frame) == 31
    assert len(api.calls("archive")) == 1
    assert api.calls("api.") == []


def test_coordinates_hourly_name_and_timezone_from_api():
    api = FakeApi(timezone="America/New_York")
    result = fetch_temperature(
        (40.71, -74.01), "2026-06-10", "2026-06-16", freq="hourly", fetch=api
    )
    assert result.location == Location(
        "40.71, -74.01", 40.71, -74.01, "America/New_York"
    )
    frame = result.frame
    assert len(frame) == 7 * 24
    assert frame["ds"].tolist() == list(
        pd.date_range("2026-06-10", "2026-06-16 23:00", freq="h")
    )
    assert api.calls("geocoding") == []
    assert set(frame["source"]) == {"archive", "forecast"}


def test_coordinate_name_uses_two_decimals():
    result = fetch_temperature(
        (51.5074, -0.1278), "2026-06-10", "2026-06-10", fetch=FakeApi()
    )
    assert result.location.name == "51.51, -0.13"


def test_hourly_climatology_is_per_hour():
    result = fetch_temperature(
        "London", "2027-04-11 00:00", "2027-04-11 02:00", freq="hourly", fetch=FakeApi()
    )
    expected = [
        sum(
            observed_temp(pd.Timestamp(year=y, month=4, day=11, hour=h))
            for y in range(2021, 2026)
        )
        / 5
        for h in range(3)
    ]
    assert result.frame["temperature"].tolist() == pytest.approx(expected)


def test_unknown_place_raises_weather_error():
    with pytest.raises(WeatherError, match="No location found"):
        fetch_temperature("Zzzqqxx", "2026-06-01", "2026-06-02", fetch=FakeApi())


def test_unfillable_gap_raises_weather_error():
    # The archive lags 3 days and a forecast that starts at today leaves a hole.
    api = FakeApi()

    def no_history_forecast(url: str) -> bytes:
        if url.startswith("https://api.open-meteo.com"):
            return json.dumps(
                {"timezone": "UTC", "daily": {"time": [], "temperature_2m_mean": []}}
            ).encode()
        return api(url)

    with pytest.raises(WeatherError, match="no temperature for"):
        fetch_temperature(
            "London", "2026-06-10", "2026-06-14", fetch=no_history_forecast
        )


def test_fetch_errors_propagate_as_weather_error():
    def broken(url: str) -> bytes:
        raise WeatherError("network down")

    with pytest.raises(WeatherError, match="network down"):
        fetch_temperature((1.0, 2.0), "2026-06-01", "2026-06-02", fetch=broken)


def test_invalid_arguments_raise_value_error():
    with pytest.raises(ValueError, match="out of range"):
        fetch_temperature((91.0, 0.0), "2026-06-01", "2026-06-02", fetch=FakeApi())
    with pytest.raises(ValueError, match="empty"):
        fetch_temperature("  ", "2026-06-01", "2026-06-02", fetch=FakeApi())


def test_place_with_country_qualifier_queries_name_part_only():
    api = FakeApi()
    result = fetch_temperature(
        "London, United Kingdom", "2025-01-01", "2025-01-02", fetch=api
    )
    assert result.location.timezone == "Europe/London"
    geocode = urllib.parse.parse_qs(
        urllib.parse.urlsplit(api.calls("geocoding")[0]).query
    )
    assert geocode["name"] == ["London"]


def test_result_is_frozen():
    result = fetch_temperature("London", "2025-01-01", "2025-01-02", fetch=FakeApi())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.location = result.location  # type: ignore[misc]


# --- cache -------------------------------------------------------------------


def test_cache_hit_avoids_network_for_settled_data(tmp_path: Path):
    first = FakeApi()
    a = fetch_temperature(
        "London", "2025-01-01", "2025-01-10", fetch=first, cache_dir=tmp_path
    )
    assert len(first.urls) == 2  # geocoding + archive
    second = FakeApi()
    b = fetch_temperature(
        "London", "2025-01-01", "2025-01-10", fetch=second, cache_dir=tmp_path
    )
    assert second.urls == []
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.location == b.location


def test_settled_archive_and_geocoding_never_expire(tmp_path: Path):
    fetch_temperature(
        "London", "2025-01-01", "2025-01-10", fetch=FakeApi(), cache_dir=tmp_path
    )
    year_ago = 400 * 86400
    for f in tmp_path.glob("*.json"):
        old = f.stat().st_mtime - year_ago
        os.utime(f, (old, old))
    again = FakeApi()
    fetch_temperature(
        "London", "2025-01-01", "2025-01-10", fetch=again, cache_dir=tmp_path
    )
    assert again.urls == []


def test_forecast_cache_expires_after_three_hours(tmp_path: Path):
    fetch_temperature(
        "London", "2026-06-10", "2026-06-20", fetch=FakeApi(), cache_dir=tmp_path
    )
    fresh = FakeApi()
    fetch_temperature(
        "London", "2026-06-10", "2026-06-20", fetch=fresh, cache_dir=tmp_path
    )
    assert fresh.urls == []  # forecast and recent archive still within 3 hours

    for f in tmp_path.glob("*.json"):
        old = f.stat().st_mtime - (weather.FORECAST_TTL_S + 60)
        os.utime(f, (old, old))
    stale = FakeApi()
    fetch_temperature(
        "London", "2026-06-10", "2026-06-20", fetch=stale, cache_dir=tmp_path
    )
    assert len(stale.calls("api.")) == 1
    assert len(stale.calls("archive")) == 1  # recent archive data also expires
    assert stale.calls("geocoding") == []  # geocoding is permanent


def test_cache_is_keyed_by_url(tmp_path: Path):
    fetch_temperature(
        "London", "2025-01-01", "2025-01-10", fetch=FakeApi(), cache_dir=tmp_path
    )
    other = FakeApi()
    fetch_temperature(
        "London", "2025-02-01", "2025-02-10", fetch=other, cache_dir=tmp_path
    )
    assert len(other.calls("archive")) == 1
    assert other.calls("geocoding") == []


def test_corrupt_cache_entry_is_refetched(tmp_path: Path):
    fetch_temperature(
        "London", "2025-01-01", "2025-01-10", fetch=FakeApi(), cache_dir=tmp_path
    )
    for f in tmp_path.glob("*.json"):
        f.write_bytes(b"not json")
    again = FakeApi()
    result = fetch_temperature(
        "London", "2025-01-01", "2025-01-10", fetch=again, cache_dir=tmp_path
    )
    assert len(again.urls) == 2
    assert len(result.frame) == 10


def test_cache_write_failure_logs_warning_and_still_works(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with caplog.at_level("WARNING", logger="prophet"):
        result = fetch_temperature(
            "London", "2025-01-01", "2025-01-03", fetch=FakeApi(), cache_dir=blocker
        )
    assert len(result.frame) == 3
    assert any("cache" in r.getMessage().lower() for r in caplog.records)


def test_error_responses_are_not_cached(tmp_path: Path):
    def failing(url: str) -> bytes:
        return json.dumps({"error": True, "reason": "nope"}).encode()

    with pytest.raises(WeatherError):
        fetch_temperature(
            (1.0, 2.0), "2025-01-01", "2025-01-02", fetch=failing, cache_dir=tmp_path
        )
    assert list(tmp_path.glob("*.json")) == []


def test_resolve_cache_dir_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv(weather.CACHE_DIR_ENV, raising=False)
    assert (
        resolve_cache_dir(None, False)
        == Path.home() / ".cache" / "neoprophet" / "weather"
    )
    monkeypatch.setenv(weather.CACHE_DIR_ENV, str(tmp_path / "env"))
    assert resolve_cache_dir(None, False) == tmp_path / "env"
    assert resolve_cache_dir(tmp_path / "arg", False) == tmp_path / "arg"


def test_injected_fetch_without_cache_dir_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(weather.CACHE_DIR_ENV, "/should/never/be/used")
    assert resolve_cache_dir(None, True) is None


# --- default HTTP layer ------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_default_fetch_sends_user_agent_and_timeout(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        seen["ua"] = request.get_header("User-agent")
        seen["timeout"] = timeout
        return _Response(b'{"ok": 1}')

    monkeypatch.setattr(weather.urllib.request, "urlopen", fake_urlopen)
    assert weather._default_fetch("https://example.test/x") == b'{"ok": 1}'
    assert "neoprophet" in seen["ua"]
    assert seen["timeout"] == weather.REQUEST_TIMEOUT_S


def test_default_fetch_converts_http_errors(monkeypatch: pytest.MonkeyPatch):
    def raise_http(request: Any, timeout: float) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many",
            {},
            io.BytesIO(b'{"reason": "slow down"}'),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(weather.urllib.request, "urlopen", raise_http)
    with pytest.raises(WeatherError, match="rate limit"):
        weather._default_fetch("https://example.test/x")


def test_default_fetch_converts_url_errors_and_timeouts(
    monkeypatch: pytest.MonkeyPatch,
):
    for exc in (urllib.error.URLError("dns failure"), TimeoutError("timed out")):

        def raise_exc(request: Any, timeout: float, _exc: Exception = exc) -> None:
            raise _exc

        monkeypatch.setattr(weather.urllib.request, "urlopen", raise_exc)
        with pytest.raises(WeatherError, match="Could not reach Open-Meteo"):
            weather._default_fetch("https://example.test/x")


def test_default_fetch_refuses_plain_http():
    with pytest.raises(WeatherError, match="non-HTTPS"):
        weather._default_fetch("http://example.test/x")
