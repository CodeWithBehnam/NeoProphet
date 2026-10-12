"""The component panels of the "Components" section: pure wording, then the panels over the fixture models.

The weekly and daily curves are drawn over a whole cycle, so these tests check
that they are trimmed to the weekdays and hours the data has and that the words
agree with the Seasonality section.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet, forecast, visuals
from prophet.dashboard import panels
from prophet.dashboard import seasonality as sea
from prophet.dashboard._context import build_context
from prophet.plot import _above_below, plot_components_plotly
from prophet.tests.test_visuals import CASES, fit_case, no_warnings

EM_DASH = chr(0x2014)
DAYS5 = frozenset(range(5))
ALL_DAYS = frozenset(range(7))
HOURS = frozenset(range(24))


def raw_panels(case: str) -> tuple[Any, list[tuple[str, str, Any]]]:
    """The context and the panels as ``plot_components_plotly`` draws them, before any fix."""
    go = pytest.importorskip("plotly.graph_objects")
    m, df, fcst = fit_case(case)
    ctx = build_context(m, fcst, df=df)
    combined = plot_components_plotly(
        m, fcst, title=False, figsize=visuals._PANEL_FIGSIZE
    )
    return ctx, visuals._panel_figures(go, combined)


def refined(case: str, covered: tuple[str, ...] = ()) -> dict[str, panels.Panel]:
    ctx, raw = raw_panels(case)
    drawn, _ = panels.refine(ctx, raw, covered)
    return {p.name: p for p in drawn}


# ---- pure: which points of a cycle the data has ----------------------------------


def week_grid() -> pd.DatetimeIndex:
    return pd.date_range("2017-01-01", periods=7, freq="D")  # Sunday to Saturday


def test_weekly_points_on_weekdays_without_data_are_dropped() -> None:
    keep = panels.observed_mask(
        week_grid(), weekly=True, intraday=False, days=DAYS5, hours=HOURS
    )
    assert keep.tolist() == [False, True, True, True, True, True, False]
    everything = panels.observed_mask(
        week_grid(), weekly=True, intraday=False, days=ALL_DAYS, hours=HOURS
    )
    assert everything.all()
    # A daily curve ignores the weekday; an intraday one looks at the hour.
    hours = pd.date_range("2017-01-01", periods=24, freq="h")
    office = frozenset(range(9, 17))
    keep = panels.observed_mask(
        hours, weekly=False, intraday=True, days=ALL_DAYS, hours=office
    )
    assert np.flatnonzero(keep).tolist() == list(range(9, 17))
    assert panels.observed_mask(
        hours, weekly=False, intraday=False, days=DAYS5, hours=office
    ).all()


def test_the_left_out_note_names_only_what_is_missing() -> None:
    both = panels.left_out_phrase(
        weekly=True, intraday=True, days=DAYS5, hours=frozenset(range(9, 17))
    )
    assert both == (
        "Saturdays and Sundays are left out because the data never has them. "
        "Hours of the day the data never has are left out."
    )
    assert (
        panels.left_out_phrase(weekly=True, intraday=False, days=ALL_DAYS, hours=HOURS)
        == ""
    )
    assert (
        panels.left_out_phrase(weekly=False, intraday=True, days=DAYS5, hours=HOURS)
        == ""
    )


def test_range_breaks_close_the_gaps_for_the_days_and_hours_left_out() -> None:
    assert panels.missing_breaks(
        weekly=True, intraday=False, days=DAYS5, hours=HOURS
    ) == [{"bounds": ["sat", "mon"]}]
    assert (
        panels.missing_breaks(weekly=True, intraday=False, days=ALL_DAYS, hours=HOURS)
        == []
    )
    office = frozenset(range(9, 17))
    assert panels.missing_breaks(
        weekly=False, intraday=True, days=ALL_DAYS, hours=office
    ) == [{"pattern": "hour", "bounds": [17, 9]}]


# ---- pure: wording --------------------------------------------------------------


def a_fact(**changes: Any) -> sea.Fact:
    base: dict[str, Any] = {
        "name": "yearly",
        "label": "Yearly",
        "kind": "yearly",
        "multiplicative": False,
        "high": "28 February",
        "low": "9 June",
        "high_value": 8.75,
        "low_value": -9.0,
        "swing": 17.75,
        "swing_y": 17.75,
        "flat": False,
    }
    return sea.Fact(**{**base, **changes})


def test_yearly_and_cycle_text_uses_the_peak_days_of_the_seasonality_section() -> None:
    head, sentence = panels.fact_text(a_fact(), "yearly")
    assert head == "Yearly pattern: peaks around 28 February"
    assert (
        sentence
        == "28 February is 8.75 above the usual level and 9 June is 9 below it."
    )
    cycle = a_fact(
        name="hourly",
        kind="cycle",
        high="50 minutes in",
        low="30 minutes in",
        high_value=0.0426,
        low_value=-0.0338,
    )
    head, sentence = panels.fact_text(cycle, "hourly")
    assert head == "Hourly pattern: peaks 50 minutes in"
    assert sentence == (
        "The high point (50 minutes in) is 0.0426 above the usual level; "
        "the low point (30 minutes in) is 0.0338 below it."
    )
    percent = a_fact(multiplicative=True, high_value=0.227, low_value=-0.156)
    assert panels.fact_text(percent, "yearly")[1] == (
        "28 February is 23% above the usual level and 9 June is 16% below it."
    )
    flat = a_fact(flat=True)
    assert panels.fact_text(flat, "yearly") == ("Yearly pattern: little variation", "")


def test_a_peak_exactly_at_the_usual_level_is_not_called_above_it() -> None:
    sentence = panels.fact_text(a_fact(high_value=0.0, low_value=-4.0), "yearly")[1]
    assert sentence == "28 February is at the usual level and 9 June is 4 below it."
    sentence = panels.fact_text(a_fact(high_value=4.0, low_value=0.0), "yearly")[1]
    assert sentence == "28 February is 4 above the usual level and 9 June is at it."


def test_the_largest_holiday_keeps_its_whole_column_name() -> None:
    columns = ["Christmas Day", "Christmas Day (observed)", "Boxing Day"]
    values = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.427, 0.0, 0.3],
            [0.0, -1.08, 0.0],
        ]
    )
    assert panels.largest_holiday(columns, values) == (
        "Christmas Day (observed)",
        2,
        -1.08,
    )
    assert panels.largest_holiday(columns, np.zeros((3, 3))) is None
    assert panels.largest_holiday(columns, np.full((3, 3), np.nan)) is None
    assert panels.largest_holiday([], np.zeros((3, 0))) is None
    head, sentence = panels.holiday_text("Christmas Day (observed)", -1.08, False)
    assert head == "Holidays: Christmas Day (observed) has the largest effect"
    assert sentence == (
        "On Christmas Day (observed) the forecast is 1.08 below its usual level."
    )
    assert "20% above" in panels.holiday_text("Sale", 0.2, True)[1]


def test_a_flat_trend_says_it_stays_where_it_is() -> None:
    head, sentence = panels.flat_trend_text(37.6)
    assert head == "Trend: broadly flat"
    assert sentence == (
        "With seasonal and holiday effects removed, the underlying level stays at about 37.6."
    )
    assert "moved from" not in sentence


@pytest.mark.parametrize(
    ("dropped", "expected"),
    [
        (["trend"], "The trend has a section of its own, so it is not repeated here."),
        (
            ["holidays"],
            "The holidays have a section of their own, so they are not repeated here.",
        ),
        (
            ["holidays", "trend"],
            "The trend and the holidays have sections of their own, so they are not repeated here.",
        ),
        ([], ""),
    ],
)
def test_the_note_says_which_panels_other_sections_draw(
    dropped: list[str], expected: str
) -> None:
    assert panels.covered_note(dropped) == expected


def test_hover_labels_read_value_position_and_interval() -> None:
    plain = panels.hover_template(
        "%{x|%d %b %Y}", "Trend", multiplicative=False, signed=False, interval=None
    )
    assert plain == "%{x|%d %b %Y}<br>Trend <b>%{y:,.4g}</b><extra></extra>"
    banded = panels.hover_template(
        "%{x|%A}",
        "Weekly",
        multiplicative=True,
        signed=True,
        interval="80% interval",
    )
    assert banded == (
        "%{x|%A}<br>Weekly <b>%{y:+.2%}</b>"
        "<br>80% interval %{customdata[0]:+.2%} to %{customdata[1]:+.2%}<extra></extra>"
    )
    names = panels.hover_template(
        "%{x|%d %b}",
        "Holidays",
        multiplicative=False,
        signed=True,
        interval=None,
        prefix="%{customdata[2]}",
    )
    assert names.startswith("%{x|%d %b}<br>%{customdata[2]}Holidays")


def test_holiday_names_in_a_hover_label_are_escaped_and_get_a_line_break() -> None:
    assert panels._escaped_names(["R&D <day>", "", "Easter"], 3) == [
        "R&amp;D &lt;day&gt;<br>",
        "",
        "Easter<br>",
    ]
    assert panels._escaped_names(None, 2) == ["", ""]


def test_positions_in_a_cycle_are_worded_for_the_cycle() -> None:
    year = pd.date_range("2017-01-01", periods=365, freq="D")
    labels = panels.cycle_labels(year, "yearly")
    assert labels[0] == "1 January" and labels[58] == "28 February"
    start = pd.Timestamp("2017-01-01")
    grid = pd.DatetimeIndex(
        [start, start + pd.Timedelta(minutes=50), start + pd.Timedelta(days=21)]
    )
    assert panels.cycle_labels(grid, "cycle") == [
        "at the start",
        "50 minutes in",
        "21 days in",
    ]


# ---- the panels over the fixture models -------------------------------------------


@pytest.mark.parametrize("case", [*CASES, "temperature"])
def test_every_panel_gets_a_hover_label_and_silent_bands(case: str) -> None:
    drawn = refined(case)
    assert drawn
    for panel in drawn.values():
        main, *bands = panel.figure.data
        assert main.hovertemplate and main.hovertemplate.endswith("<extra></extra>")
        assert "%{y" in main.hovertemplate
        for trace in bands:
            if trace.name in ("Cap", "Floor"):
                assert "capacity" in trace.hovertemplate
            else:
                assert trace.hoverinfo == "skip", (case, panel.name, trace.name)
        assert EM_DASH not in panel.head + panel.sentence + main.hovertemplate


@pytest.mark.parametrize(
    "case", ["linear", "temperature", "multiplicative", "regressors"]
)
def test_the_weekly_panel_has_only_the_weekdays_the_data_has_and_agrees_with_the_seasonality_section(
    case: str,
) -> None:
    ctx, raw = raw_panels(case)
    weekly_before = next(f for _, _, f in raw if f.data[0].name == "weekly")
    assert len(weekly_before.data[0].x) == 7
    drawn, _ = panels.refine(ctx, raw)
    weekly = next(p for p in drawn if p.name == "weekly")
    x = pd.DatetimeIndex(np.asarray(weekly.figure.data[0].x))
    assert sorted(set(x.dayofweek)) == [0, 1, 2, 3, 4]
    assert (
        "Saturday" not in weekly.head
        and "Saturdays and Sundays are left out" in weekly.sentence
    )
    assert weekly.figure.layout.xaxis.rangebreaks[0].bounds == ("sat", "mon")
    # The curve is centred on the days it keeps, as the Seasonality section's numbers are.
    assert abs(float(np.mean(weekly.figure.data[0].y))) < 1e-9
    days, hours = sea.observed_calendar(ctx.history["ds"], False)
    (fact,) = [
        f
        for f in sea.seasonal_facts(
            ctx.m, ctx.fcst, ctx.history_end, ctx.step, days, hours
        )
        if f.name == "weekly"
    ]
    if "little variation" not in weekly.head:
        assert weekly.head == f"Weekly pattern: {fact.high}s are highest"
        assert (
            f"{fact.high}s are {_above_below(fact.high_value, fact.multiplicative)}"
            in weekly.sentence
        )
        assert (
            f"{fact.low}s {_above_below(fact.low_value, fact.multiplicative)}"
            in weekly.sentence
        )


def test_the_weekly_sentence_no_longer_names_the_days_the_data_never_has() -> None:
    # data.csv holds Monday to Friday only, so the weekend values of the weekly curve are
    # extrapolation. Which weekend day the unfixed sentence names, and its numbers, depend on the
    # optimiser: "Saturdays 72.6 below" on macOS, "Sundays 72.9 below" on Linux x86.
    weekend = ("Saturday", "Sunday")
    ctx, raw = raw_panels("regressors")
    assert set(ctx.history["ds"].dt.day_name()).isdisjoint(weekend)
    before = next(s for h, s, f in raw if f.data[0].name == "weekly")
    assert any(day in before for day in weekend)
    after = refined("regressors")["weekly"]
    assert not any(day in after.head for day in weekend)
    # A weekend day may only appear to say it was left out, never as the high or the low.
    extreme = re.compile(r"(Saturday|Sunday)s (are )?[+-]?\d[\d.,]* (above|below)")
    assert not extreme.search(after.sentence)
    assert "left out" in after.sentence
    numbers = re.compile(r"\d+(?:\.\d+)?")
    assert numbers.findall(after.sentence) != numbers.findall(before)


def test_yearly_text_matches_the_peak_days_of_the_seasonality_section() -> None:
    ctx, raw = raw_panels("linear")
    before = next(s for h, s, f in raw if f.data[0].name == "yearly")
    assert before.startswith("February is")  # a month average
    drawn, _ = panels.refine(ctx, raw)
    yearly = next(p for p in drawn if p.name == "yearly")
    days, hours = sea.observed_calendar(ctx.history["ds"], False)
    (fact,) = [
        f
        for f in sea.seasonal_facts(
            ctx.m, ctx.fcst, ctx.history_end, ctx.step, days, hours
        )
        if f.name == "yearly"
    ]
    assert yearly.head == f"Yearly pattern: peaks around {fact.high}"
    assert (
        f"{fact.high} is " in yearly.sentence and f"{fact.low} is " in yearly.sentence
    )


def test_a_custom_cycle_is_worded_in_the_units_of_the_cycle() -> None:
    drawn = refined("subdaily")
    hourly = drawn["hourly"]
    assert re.fullmatch(r"Hourly pattern: peaks \d+ minutes in", hourly.head)
    assert "at 00:50" not in hourly.head
    monthly = refined("no_regressors_custom_seasonality")["monthly"]
    assert re.fullmatch(r"Monthly pattern: peaks \d+(\.\d+)? days in", monthly.head)
    assert "day 21 of 30.5" not in monthly.head
    # Position labels for the hover come from the cycle.
    assert monthly.figure.data[0].customdata[0][2] == "at the start"
    assert monthly.figure.data[0].customdata[1][2] == "1 day in"


def test_hours_the_data_never_has_are_trimmed_from_the_daily_curve() -> None:
    pytest.importorskip("plotly")
    ds = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    ds = ds[(ds.hour >= 9) & (ds.hour < 17)]
    rng = np.random.default_rng(3)
    y = 10 + 3 * np.sin((ds.hour - 9) / 8 * np.pi) + rng.normal(0, 0.2, len(ds))
    df = pd.DataFrame({"ds": ds, "y": y})
    m = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=True,
        uncertainty_samples=0,
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=30, freq="h", include_history=False)
    future = future[(future["ds"].dt.hour >= 9) & (future["ds"].dt.hour < 17)]
    fcst = m.predict(pd.concat([df[["ds"]], future]))
    ctx = build_context(m, fcst, df=df)
    go = pytest.importorskip("plotly.graph_objects")
    combined = plot_components_plotly(
        m, fcst, title=False, figsize=visuals._PANEL_FIGSIZE
    )
    drawn, _ = panels.refine(ctx, visuals._panel_figures(go, combined))
    daily = next(p for p in drawn if p.name == "daily")
    hours = sorted(set(pd.DatetimeIndex(np.asarray(daily.figure.data[0].x)).hour))
    assert hours == list(range(9, 17))
    assert daily.figure.layout.xaxis.rangebreaks[0].bounds == (17, 9)
    assert "Hours of the day the data never has are left out." in daily.sentence


def test_the_holiday_panel_names_the_observed_day_when_that_is_the_big_one() -> None:
    pytest.importorskip("plotly")
    df = pd.read_csv(Path(__file__).parent / "data.csv", parse_dates=["ds"])
    plain = pd.to_datetime(["2012-12-25", "2013-12-25"])
    observed = pd.to_datetime(["2012-12-26", "2013-12-26"])
    holidays = pd.concat(
        [
            pd.DataFrame({"holiday": "Christmas Day", "ds": plain}),
            pd.DataFrame({"holiday": "Christmas Day (observed)", "ds": observed}),
        ]
    )
    bump = np.where(
        df["ds"].isin(observed), -9.0, np.where(df["ds"].isin(plain), -3.0, 0.0)
    )
    df = df.assign(y=df["y"] + bump)
    m = Prophet(holidays=holidays, uncertainty_samples=0).fit(df)
    fcst = m.predict(m.make_future_dataframe(periods=10))
    go = pytest.importorskip("plotly.graph_objects")
    ctx = build_context(m, fcst, df=df)
    combined = plot_components_plotly(
        m, fcst, title=False, figsize=visuals._PANEL_FIGSIZE
    )
    raw = visuals._panel_figures(go, combined)
    before = next(h for h, _, f in raw if f.data[0].name == "holidays")
    drawn, _ = panels.refine(ctx, raw)
    holiday = next(p for p in drawn if p.name == "holidays")
    # prophet.plot strips "(observed)" and keeps the observed day's number.
    assert before == "Holidays: Christmas Day has the largest effect"
    assert holiday.head == "Holidays: Christmas Day (observed) has the largest effect"
    assert holiday.sentence.startswith("On Christmas Day (observed) the forecast is ")
    cols = {
        c: float(fcst[c].abs().max())
        for c in ("Christmas Day", "Christmas Day (observed)")
    }
    assert cols["Christmas Day (observed)"] > cols["Christmas Day"]
    # Hover labels carry the names of the holidays on the day, escaped.
    trace = holiday.figure.data[0]
    assert trace.hovertemplate.startswith("%{x|%d %b %Y}<br>%{customdata[2]}Holidays")
    named = [row[2] for row in trace.customdata if row[2]]
    assert named and all(n.endswith("<br>") for n in named)


def test_a_flat_trend_panel_does_not_say_it_moved() -> None:
    trend = refined("flat")["trend"]
    assert trend.head == "Trend: broadly flat"
    assert "moved from" not in trend.sentence and "stays at about" in trend.sentence


# ---- leaving out what other sections draw ----------------------------------------------


def test_panels_with_a_section_of_their_own_are_left_out_and_the_rest_stay() -> None:
    ctx, raw = raw_panels("temperature")
    names_before = [f.data[0].name for _, _, f in raw]
    assert names_before[0] == "trend"
    drawn, dropped = panels.refine(ctx, raw, ("trend", "holidays"))
    assert dropped == ["trend"]  # this model has no holidays
    assert [p.name for p in drawn] == names_before[1:]


def test_components_section_leaves_out_the_trend_when_it_has_its_own() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("temperature")
    ctx = build_context(m, fcst, df=df)
    with_trend = visuals._components_section(ctx)
    without = visuals._components_section(ctx, covered=["trend"])
    assert with_trend is not None and without is not None
    assert with_trend.title == "Components"
    assert "Trend: " in with_trend.body and "Trend: " not in without.body
    assert (
        without.body.count("Plotly.newPlot")
        == with_trend.body.count("Plotly.newPlot") - 1
    )
    assert without.body.startswith(
        '<p class="lede">The trend has a section of its own, so it is not repeated here.</p>'
    )
    assert "section of its own" not in with_trend.body


def test_a_components_section_with_nothing_left_is_not_drawn() -> None:
    pytest.importorskip("plotly")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=60), "y": rng.normal(10, 1, 60)}
    )
    m = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).fit(df)
    fcst = m.predict(m.make_future_dataframe(periods=5))
    ctx = build_context(m, fcst, df=df)
    assert visuals._components_section(ctx) is not None  # the trend alone is a panel
    assert visuals._components_section(ctx, covered=["trend"]) is None


def test_the_page_draws_the_trend_once_when_the_trend_section_exists(
    tmp_path: Any,
) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    page = visuals.dashboard_html(m, fcst, tmp_path / "d.html", df=df).read_text(
        encoding="utf-8"
    )
    sections = dict(
        re.findall(r'<section id="(\w+)">(.*?)</section>', page, flags=re.S)
    )
    assert "components" in sections and "trend" in sections
    assert "Trend: " not in sections["components"]
    assert "has a section of its own" in sections["components"]
    only = visuals.dashboard_html(
        m, fcst, tmp_path / "e.html", df=df, sections=["components"]
    ).read_text(encoding="utf-8")
    assert "Trend: " in only and "has a section of its own" not in only


def test_sections_that_need_to_know_the_others_are_built_last() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    ctx = build_context(m, fcst, df=df)
    seen: list[frozenset[str]] = []

    def components(c: Any, present: frozenset[str]) -> Any:
        seen.append(present)
        return visuals._components_section(c, covered=present & {"trend"})

    built = visuals._build_sections(
        ["components", "trend", "forecast"],
        ctx,
        {"forecast": lambda c: visuals._forecast_section(c, ctx.step)},
        {"components": components},
    )
    assert [s.key for s in built] == ["components", "trend", "forecast"]
    assert seen == [frozenset({"trend", "forecast"})]


def test_weekday_data_keeps_its_range_breaks_after_the_fix(tmp_path: Any) -> None:
    pytest.importorskip("plotly")
    index = pd.bdate_range("2024-03-01", periods=90)
    rng = np.random.default_rng(5)
    series = pd.Series(50 + np.arange(90) * 0.1 + rng.normal(0, 0.5, 90), index=index)
    result = forecast(series, horizon=10, skip_days="weekends", uncertainty_samples=20)
    page = result.dashboard(
        tmp_path / "w.html", backtest=False, sections=["components"]
    ).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    weekly = None
    for call in re.finditer(r'Plotly\.newPlot\(\s*"[^"]+",\s*', page):
        data, end = decoder.raw_decode(page, call.end())
        if data and data[0].get("name") == "weekly":
            weekly = data
    assert weekly is not None
    assert len(weekly[0]["x"]) == 5
    shown = page.split("Weekly pattern", 1)[1][:300]
    assert "Saturdays are" not in shown and "Saturdays " + "below" not in shown


def test_the_smoke_matrix_builds_a_section_for_every_fixture() -> None:
    pytest.importorskip("plotly")
    for case in [*CASES, "temperature"]:
        m, df, fcst = fit_case(case)
        ctx = build_context(m, fcst, df=df)
        with no_warnings():
            section = visuals._components_section(ctx)
        assert section is not None and section.title == "Components"
        assert EM_DASH not in section.body
        assert len(section.body) < 1_500_000
