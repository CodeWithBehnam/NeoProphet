"""The "Seasonal patterns" dashboard section: pure numbers and wording, then a smoke matrix over the fixture models."""

from __future__ import annotations

import calendar
import html
import re
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet
from prophet.dashboard import seasonality as sea
from prophet.dashboard._context import Section, build_context
from prophet.dashboard._palette import DARK_SWAP
from prophet.plot import skip_axis
from prophet.tests.test_visuals import DATA, fit_case, no_warnings

CASES = [
    "linear",
    "logistic",
    "flat",
    "multiplicative",
    "subdaily",
    "no_uncertainty",
    "regressors",
    "temperature",
    "no_regressors_custom_seasonality",
]
DAYS = frozenset(range(7))
HOURS = frozenset(range(24))
WEEK = tuple(calendar.day_name)
#: A weekly curve that averages to zero: Saturday is the peak and Tuesday the low, 3.2 apart.
CENTRED = [0.0, -1.6, 0.3, 0.2, -0.5, 1.6, 0.0]


# ---- helpers ----------------------------------------------------------------


def toy_forecast(n: int = 100) -> pd.DataFrame:
    i = np.arange(n)
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n),
            "trend": np.linspace(10, 20, n),
            "weekly": 2 * np.sin(2 * np.pi * i / 7),
            "yearly": 0.1 * np.cos(i / 10),
            "promo": 3.0 * (i % 5 == 0),
        }
    )


def toy_model(**overrides: Any) -> Any:
    """Just the attributes component_importance reads, so no fit is needed."""
    base: dict[str, Any] = {
        "component_modes": {
            "additive": ["weekly", "promo", "additive_terms"],
            "multiplicative": ["yearly", "multiplicative_terms"],
        },
        "seasonalities": {"weekly": {}, "yearly": {}},
        "train_holiday_names": None,
        "extra_regressors": {"promo": {}},
        "history": pd.DataFrame({"y": [10.0, 20.0]}),
    }
    return SimpleNamespace(**{**base, **overrides})


def weekly_profile(values: list[float]) -> sea.Profile:
    return sea.Profile("weekly", WEEK, tuple(values))


def importance(
    name: str, share: float, part: sea.Part = "seasonality"
) -> sea.Importance:
    noun = "The trend" if part == "trend" else f"The {name} pattern"
    return sea.Importance(name, name.capitalize(), noun, part, share * 10, share)


def hourly_frame(days: int = 21) -> pd.DataFrame:
    """Hourly rows from a Monday; y encodes the weekday and the hour so each cell mean is known."""
    ds = pd.date_range("2024-01-01", periods=days * 24, freq=pd.Timedelta(hours=1))
    return pd.DataFrame({"ds": ds, "y": ds.dayofweek * 100.0 + ds.hour})


def daily_frame(start: str, days: int) -> pd.DataFrame:
    ds = pd.date_range(start, periods=days)
    return pd.DataFrame({"ds": ds, "y": ds.dayofweek * 10.0 + ds.month})


def without_scripts(body: str) -> str:
    """The page text: the section body with the plotly scripts taken out."""
    return re.sub(r"<script.*?</script>", "", body, flags=re.DOTALL)


# ---- pure: wording ----------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (12.2, "12 days"),
        (1.0, "1 day"),
        (3.5, "3.5 days"),
        (1 / 24, "1 hour"),
        (5 / 24, "5 hours"),
        (30 / 1440, "30 minutes"),
    ],
)
def test_span_words(days: float, expected: str) -> None:
    assert sea._span_words(days) == expected


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (30.5, "30.5-day"),
        (14.0, "14-day"),
        (1 / 24, "1-hour"),
        (10 / 1440, "10-minute"),
    ],
)
def test_cycle_words(period: float, expected: str) -> None:
    assert sea._cycle_words(period) == expected


@pytest.mark.parametrize(
    ("period", "kind"),
    [
        (7, "weekly"),
        (1, "daily"),
        (365.25, "yearly"),
        (30.5, "cycle"),
        (1 / 24, "cycle"),
    ],
)
def test_seasonality_kind(period: float, kind: str) -> None:
    assert sea.seasonality_kind(period) == kind


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The weekly pattern", "the weekly pattern"),
        ("Holidays", "holidays"),
        ("NFL season", "NFL season"),
        ("Q4 effect", "Q4 effect"),
        ("The NFL season pattern", "the NFL season pattern"),
        ("b<i>g", "b<i>g"),
        ("", ""),
    ],
)
def test_lower_first_keeps_names_that_start_with_an_acronym(
    text: str, expected: str
) -> None:
    assert sea._lower_first(text) == expected


def test_missing_days_phrase_names_the_weekdays_the_data_never_has() -> None:
    weekdays = frozenset(range(5))
    assert (
        sea.missing_days_phrase(weekdays)
        == "Saturdays and Sundays are left out because the data never has them."
    )
    assert sea.missing_days_phrase(frozenset(range(6))).startswith(
        "Sundays are left out"
    )
    assert sea.missing_days_phrase(DAYS) == ""
    assert sea.missing_days_phrase(frozenset()) == ""


# ---- pure: peaks and lows ---------------------------------------------------


def test_weekly_fact_reads_like_the_example_in_the_brief() -> None:
    profile = weekly_profile(CENTRED)
    fact = sea.fact_from_profile(
        profile,
        name="weekly",
        label="Weekly",
        multiplicative=False,
        scale=100.0,
        level=1.0,
    )
    assert (fact.high, fact.low) == ("Saturday", "Tuesday")
    assert (
        fact.sentence
        == "Weekly: highest on Saturday, lowest on Tuesday, a swing of 3.2"
    )
    assert not fact.flat and fact.swing_y == pytest.approx(3.2)


def test_multiplicative_fact_is_in_percent_and_compares_in_units_of_the_data() -> None:
    profile = weekly_profile([0.0, 0.0, 0.0, 0.0, 0.0, 0.06, -0.06])
    fact = sea.fact_from_profile(
        profile,
        name="weekly",
        label="Weekly",
        multiplicative=True,
        scale=100.0,
        level=50.0,
    )
    assert (
        fact.sentence
        == "Weekly: highest on Saturday, lowest on Sunday, a swing of 12% of the trend"
    )
    assert fact.swing_y == pytest.approx(0.12 * 50.0)


def test_flat_profiles_say_so_instead_of_naming_a_peak() -> None:
    profile = weekly_profile([1.0, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0])
    fact = sea.fact_from_profile(
        profile,
        name="weekly",
        label="Weekly",
        multiplicative=False,
        scale=100.0,
        level=1.0,
    )
    assert fact.flat and fact.sentence == "Weekly: almost no pattern"
    rows = sea.fact_rows([fact], {})
    assert rows == [["Weekly", "No clear peak", "No clear low", "0.001"]]


def test_peak_and_low_are_distances_from_the_average_of_the_curve() -> None:
    kwargs: dict[str, Any] = {
        "name": "weekly",
        "label": "Weekly",
        "multiplicative": False,
        "scale": 100.0,
        "level": 1.0,
    }
    plain = sea.fact_from_profile(weekly_profile(CENTRED), **kwargs)
    # The same curve sitting at an arbitrary level, as an unconstrained seasonal curve may do.
    lifted = sea.fact_from_profile(weekly_profile([4.5 + v for v in CENTRED]), **kwargs)
    assert lifted.high_value == pytest.approx(plain.high_value) == pytest.approx(1.6)
    assert lifted.low_value == pytest.approx(plain.low_value) == pytest.approx(-1.6)
    assert lifted.swing == pytest.approx(3.2)
    assert plain.high_value > 0 > plain.low_value
    # Over a few weekdays only, the kept days are centred on their own average.
    three = sea.Profile("weekly", WEEK[:3], (10.0, 12.0, 14.0))
    fact = sea.fact_from_profile(three, **kwargs)
    assert (fact.high_value, fact.low_value) == (2.0, -2.0)


def test_flatness_is_judged_against_the_movement_of_the_data_not_its_level() -> None:
    kwargs: dict[str, Any] = {"name": "weekly", "label": "Weekly", "level": 1.0}
    # y sits near 10 000 and moves by about 10: a swing of 25 dominates.
    big = sea.fact_from_profile(
        weekly_profile([0, 5, 10, 15, 20, 25, 8]),
        multiplicative=False,
        scale=10.0,
        **kwargs,
    )
    assert not big.flat
    tiny = sea.fact_from_profile(
        weekly_profile([0, 0.01, 0.02, 0, 0, 0.05, 0]),
        multiplicative=False,
        scale=10.0,
        **kwargs,
    )
    assert tiny.flat
    # A multiplicative swing of 0.4% of a trend near 1000 is 4 units of data that moves by 10.
    kwargs["level"] = 1000.0
    fraction = sea.fact_from_profile(
        weekly_profile([0, 0.004, 0, 0, 0, 0, 0]),
        multiplicative=True,
        scale=10.0,
        **kwargs,
    )
    assert not fraction.flat and fraction.swing_y == pytest.approx(4.0)


def test_data_spread_is_the_standard_deviation_of_y() -> None:
    assert sea._data_spread(toy_model()) == pytest.approx(5.0)
    constant = toy_model(history=pd.DataFrame({"y": [7.0, 7.0, 7.0]}))
    assert sea._data_spread(constant) == 7.0  # no spread: falls back to the size of y
    empty = toy_model(history=pd.DataFrame({"y": np.empty(0)}))
    assert sea._data_spread(empty) == 1.0


def test_each_kind_of_profile_has_its_own_phrasing() -> None:
    kwargs: dict[str, Any] = {"multiplicative": False, "scale": 10.0, "level": 1.0}
    daily = sea.Profile("daily", ("00:00", "14:00", "20:00"), (0.0, 2.0, -1.0))
    assert (
        sea.fact_from_profile(daily, name="daily", label="Daily", **kwargs).sentence
        == "Daily: highest around 14:00, lowest around 20:00, a swing of 3"
    )
    yearly = sea.Profile("yearly", ("3 January", "12 July"), (-4.0, 5.0))
    assert (
        sea.fact_from_profile(yearly, name="yearly", label="Yearly", **kwargs).sentence
        == "Yearly: highest around 12 July, lowest around 3 January, a swing of 9"
    )
    cycle = sea.Profile("cycle", ("at the start", "12 days in"), (-1.0, 1.0))
    assert (
        sea.fact_from_profile(
            cycle, name="monthly", label="Monthly (30.5-day cycle)", **kwargs
        ).detail
        == "highest 12 days in, lowest at the start, a swing of 2"
    )


def test_strongest_fact_compares_additive_and_multiplicative_in_one_unit() -> None:
    weekly = sea.fact_from_profile(
        weekly_profile([0, 1, 2, 3, 4, 5, 6]),
        name="weekly",
        label="Weekly",
        multiplicative=False,
        scale=10.0,
        level=1.0,
    )
    yearly = sea.Profile("yearly", ("3 January", "12 July"), (-0.5, 0.5))
    big = sea.fact_from_profile(
        yearly,
        name="yearly",
        label="Yearly",
        multiplicative=True,
        scale=10.0,
        level=100.0,
    )
    assert sea.strongest_fact([weekly, big]) is big
    assert sea.strongest_fact([]) is None


def test_fact_table_rows_carry_signed_values_and_the_share() -> None:
    fact = sea.fact_from_profile(
        weekly_profile(CENTRED),
        name="weekly",
        label="Weekly",
        multiplicative=False,
        scale=100.0,
        level=1.0,
    )
    assert sea.fact_rows([fact], {"weekly": 0.12}) == [
        ["Weekly", "Saturday (+1.6)", "Tuesday (-1.6)", "3.2", "12%"]
    ]
    assert sea.fact_rows([fact], {"other": 1.0})[0][-1] == "0%"
    assert sea.fact_rows([fact], {}) == [
        ["Weekly", "Saturday (+1.6)", "Tuesday (-1.6)", "3.2"]
    ]


# ---- pure: how much each component matters ----------------------------------


def test_importance_measures_every_component_in_units_of_the_data() -> None:
    fc = toy_forecast()
    items = sea.component_importance(toy_model(), fc, fc["ds"].max())
    by_name = {i.name: i for i in items}
    assert set(by_name) == {"trend", "weekly", "yearly", "promo"}
    assert by_name["weekly"].spread == pytest.approx(np.std(fc["weekly"]))
    # The multiplicative yearly term is a fraction of the trend, so it is centred and scaled by the trend first,
    # and its average (a constant multiple of the trend) is booked to the trend.
    centred = (fc["yearly"] - fc["yearly"].mean()) * fc["trend"]
    assert by_name["yearly"].spread == pytest.approx(np.std(centred))
    assert by_name["trend"].spread == pytest.approx(
        np.std(fc["trend"] * (1 + fc["yearly"].mean()))
    )
    assert sum(i.share for i in items) == pytest.approx(1.0)
    shares = [i.share for i in items]
    assert shares == sorted(shares, reverse=True)
    assert (
        by_name["promo"].label == "Promo (regressor)"
        and by_name["promo"].part == "regressor"
    )
    assert by_name["yearly"].noun == "The yearly pattern"


def test_a_constant_level_in_a_multiplicative_component_belongs_to_the_trend() -> None:
    # The weekly fraction sits at 4.5 with a real swing of 0.01: weekday-only data leave the curve's level free.
    fc = toy_forecast().assign(
        weekly=4.5 + 0.01 * np.sin(np.arange(100) * 2 * np.pi / 7)
    )
    model = toy_model(
        component_modes={
            "additive": ["promo"],
            "multiplicative": ["weekly", "yearly"],
        },
        extra_regressors={},
    )
    items = {i.name: i for i in sea.component_importance(model, fc, fc["ds"].max())}
    # Without centring, std(4.5 * trend) would make "weekly" carry about 90% of the movement.
    assert items["weekly"].share < 0.01
    assert items["weekly"].spread == pytest.approx(
        np.std((fc["weekly"] - fc["weekly"].mean()) * fc["trend"])
    )
    assert next(iter(items)) == "trend" and items["trend"].share > 0.9
    # The trend is the trend as the model scales it: trend * (1 + the constant parts).
    constant = fc["weekly"].mean() + fc["yearly"].mean()
    assert items["trend"].spread == pytest.approx(np.std(fc["trend"] * (1 + constant)))


def test_additive_components_ignore_their_level() -> None:
    fc = toy_forecast()
    plain = {
        i.name: i.spread
        for i in sea.component_importance(toy_model(), fc, fc["ds"].max())
    }
    lifted = {
        i.name: i.spread
        for i in sea.component_importance(
            toy_model(), fc.assign(weekly=fc["weekly"] + 50.0), fc["ds"].max()
        )
    }
    assert lifted["weekly"] == pytest.approx(plain["weekly"])


def test_plural_subjects_take_a_plural_verb() -> None:
    fc = toy_forecast()
    fc["holidays"] = 90.0 * (np.arange(len(fc)) % 10 == 0)
    names = [f"r{k}" for k in range(7)]
    for k, name in enumerate(names):
        fc[name] = 40.0 * np.sin(np.arange(len(fc)) / 3 + k)
    holiday_model = toy_model(
        train_holiday_names=pd.Series(["day"]),
        extra_regressors={},
        component_modes={
            "additive": ["weekly", "holidays"],
            "multiplicative": ["yearly"],
        },
    )
    items = sea.component_importance(holiday_model, fc, fc["ds"].max())
    assert items[0].name == "holidays" and items[0].verb == "carry"
    assert sea.importance_sentence(items).startswith("Holidays carry ")
    assert sea.summary_sentence(items, None).startswith("Holidays carry ")
    many = toy_model(
        extra_regressors={name: {} for name in names},
        component_modes={"additive": ["weekly", *names], "multiplicative": ["yearly"]},
    )
    items = sea.component_importance(many, fc, fc["ds"].max())
    assert items[0].name == "extra_regressors"
    assert sea.importance_sentence(items).startswith("The 7 regressors together carry ")
    single = sea.component_importance(toy_model(), toy_forecast(), fc["ds"].max())
    assert sea.importance_sentence(single).startswith(f"{single[0].noun} carries ")


def test_custom_seasonality_names_keep_their_acronyms_in_sentences() -> None:
    fc = toy_forecast().rename(columns={"weekly": "NFL_season"})
    model = toy_model(
        seasonalities={"NFL_season": {}, "yearly": {}},
        component_modes={
            "additive": ["NFL_season", "promo"],
            "multiplicative": ["yearly"],
        },
    )
    items = sea.component_importance(model, fc, fc["ds"].max())
    noun = next(i.noun for i in items if i.name == "NFL_season")
    assert noun == "The NFL season pattern"
    assert "the NFL season pattern at " in sea.importance_sentence(items[::-1])


def test_importance_leaves_out_components_that_do_not_move() -> None:
    fc = toy_forecast().assign(trend=10.0, promo=0.0)
    names = {i.name for i in sea.component_importance(toy_model(), fc, fc["ds"].max())}
    assert names == {"weekly", "yearly"}


def test_importance_adds_holidays_and_combines_many_regressors() -> None:
    fc = toy_forecast()
    fc["holidays"] = 5.0 * (np.arange(len(fc)) % 30 == 0)
    names = [f"r{k}" for k in range(7)]
    for k, name in enumerate(names):
        fc[name] = k * np.sin(np.arange(len(fc)) / 3 + k)
    model = toy_model(
        train_holiday_names=pd.Series(["day"]),
        extra_regressors={name: {} for name in names},
        component_modes={
            "additive": ["weekly", "holidays", *names],
            "multiplicative": ["yearly"],
        },
    )
    items = {i.name: i for i in sea.component_importance(model, fc, fc["ds"].max())}
    assert "holidays" in items and items["holidays"].part == "holidays"
    combined = items["extra_regressors"]
    assert (
        combined.label == "Regressors (7)"
        and combined.noun == "The 7 regressors together"
    )
    assert combined.spread == pytest.approx(np.std(sum(fc[n] for n in names)))
    assert not any(n in items for n in names)


def test_importance_of_a_future_only_forecast_uses_all_its_rows() -> None:
    fc = toy_forecast()
    rows, in_sample = sea.basis_rows(fc, fc["ds"].min() - pd.Timedelta(days=1))
    assert len(rows) == len(fc) and not in_sample
    rows, in_sample = sea.basis_rows(fc, fc["ds"].iloc[49])
    assert len(rows) == 50 and in_sample
    assert sea.component_importance(toy_model(), fc.head(1), fc["ds"].max()) == ()


def test_importance_sentence_names_the_leaders() -> None:
    items = [
        importance("trend", 0.7, "trend"),
        importance("yearly", 0.21),
        importance("weekly", 0.09),
    ]
    assert sea.importance_sentence(items) == (
        "The trend carries most of the movement in the fitted values (70%), "
        "followed by the yearly pattern at 21% and the weekly pattern at 9%."
    )
    even = [
        importance("yearly", 0.4),
        importance("weekly", 0.35),
        importance("trend", 0.25, "trend"),
    ]
    sentence = sea.importance_sentence(even, in_sample=False)
    assert "the largest share of the movement in the forecast (40%)" in sentence
    assert sea.importance_sentence(items[:1]) == ""


def test_summary_and_lede_say_what_matters_and_where_it_peaks() -> None:
    weekly = sea.fact_from_profile(
        weekly_profile(CENTRED),
        name="weekly",
        label="Weekly",
        multiplicative=False,
        scale=100.0,
        level=1.0,
    )
    items = [importance("trend", 0.8, "trend"), importance("weekly", 0.2)]
    assert sea.summary_sentence(items, weekly) == (
        "The trend carries most of the movement (80%); the strongest seasonal "
        "pattern is weekly, peaking on Saturday (a swing of 3.2)."
    )
    lead = [importance("weekly", 0.9), importance("trend", 0.1, "trend")]
    assert sea.summary_sentence(lead, weekly).endswith(
        "(90%), peaking on Saturday (a swing of 3.2)."
    )
    assert sea.summary_sentence([], weekly) == (
        "The strongest seasonal pattern is weekly, peaking on Saturday (a swing of 3.2)."
    )
    assert (
        sea.summary_sentence(items, None)
        == "The trend carries most of the movement (80%)."
    )
    # A swing wider than the data says so, in the same sentence.
    assert sea.summary_sentence(items, weekly, "range").endswith(
        "(a swing of 3.2, wider than the whole range of the data)."
    )
    assert sea.summary_sentence(items, weekly, "trend").endswith(
        "(a swing of 3.2, larger than the trend itself)."
    )
    assert all(
        len(sea.summary_sentence(i, weekly).split()) <= 25 for i in (items, lead)
    )
    assert sea.summary_sentence([], None) == ""
    lede = sea.lede_text(items, [weekly], True, True)
    assert (
        lede.count("%. The strongest") == 1
        and "followed by the weekly pattern at 20%" in lede
    )
    flat = sea.fact_from_profile(
        weekly_profile([1.0] * 7),
        name="weekly",
        label="Weekly",
        multiplicative=False,
        scale=10.0,
        level=1.0,
    )
    assert "close to flat" in sea.lede_text(items, [flat], True, True)
    assert "fitted no seasonal pattern" in sea.lede_text([], [], True, True)


# ---- pure: swings wider than the data ------------------------------------------


def fact_of(
    swing_y: float, *, multiplicative: bool = False, swing: float | None = None
) -> sea.Fact:
    return sea.Fact(
        name="yearly",
        label="Yearly",
        kind="yearly",
        multiplicative=multiplicative,
        high="1 January",
        low="21 June",
        high_value=swing_y / 2,
        low_value=-swing_y / 2,
        swing=swing_y if swing is None else swing,
        swing_y=swing_y,
        flat=False,
    )


def test_a_swing_wider_than_the_data_or_the_trend_is_called_out() -> None:
    observed = (18.0, 72.0)  # a range of 54
    assert sea.swing_excess(fact_of(20.0), observed) is None
    assert sea.swing_excess(fact_of(159.0), observed) == "range"
    assert sea.swing_excess(fact_of(159.0), None) is None
    # A multiplicative swing of more than 100% of the trend flips the sign of the trend.
    assert (
        sea.swing_excess(fact_of(30.0, multiplicative=True, swing=4.68), observed)
        == "trend"
    )
    assert (
        sea.swing_excess(fact_of(30.0, multiplicative=True, swing=0.4), observed)
        is None
    )
    flat = sea.Fact(**{**fact_of(159.0).__dict__, "flat": True})
    assert sea.swing_excess(flat, observed) is None


def test_the_caution_says_what_to_read_the_swing_with() -> None:
    wide = fact_of(159.0)
    assert sea.swing_caution(wide, "range", (17.7, 72.0), ["temperature"]) == (
        "The yearly pattern swings by more than the whole range of the data (17.7 to 72); "
        "the temperature regressor largely offsets it, so read the two together."
    )
    assert sea.swing_caution(
        wide, "range", (17.7, 72.0), ["ad_spend", "promo"]
    ).endswith(
        "the ad spend and promo regressors largely offsets it, so read the two together."
    )
    unexplained = sea.swing_caution(wide, "range", (17.7, 72.0), [])
    assert unexplained.endswith(
        "Another part of the model is probably cancelling it out, so read the components together."
    )
    trendy = fact_of(30.0, multiplicative=True, swing=4.68)
    assert sea.swing_caution(trendy, "trend", (18.0, 72.0), []).startswith(
        "The yearly pattern swings by 468% of the trend, more than the trend itself."
    )
    for text in (sea.EXCESS_PHRASE["range"], sea.EXCESS_PHRASE["trend"]):
        assert "—" not in text


def test_an_offsetting_regressor_is_noted_even_when_the_swing_is_ordinary() -> None:
    ordinary = fact_of(20.0)
    assert sea.swing_caution(ordinary, None, (18.0, 72.0), ["temperature"]) == (
        "The yearly pattern is largely offset by the temperature regressor, "
        "so read the two together."
    )
    assert sea.swing_caution(ordinary, None, (18.0, 72.0), []) == ""
    assert sea.swing_caution(ordinary, None, None, ["a", "b"]).startswith(
        "The yearly pattern is largely offset by the a and b regressors"
    )


def test_a_regressor_that_mirrors_a_seasonal_pattern_is_named() -> None:
    n = 200
    i = np.arange(n)
    season = 40 * np.cos(2 * np.pi * i / 100)
    rows = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n),
            "trend": np.full(n, 50.0),
            "yearly": season,
            "temp": -season / 2 + 1,  # half the size, opposite sign
            "noise": np.random.default_rng(0).normal(0, 20, n),
            "tiny": -season / 100,  # opposite but negligible
            "twin": season,  # same sign: it adds, it does not offset
        }
    )
    model = toy_model(
        seasonalities={"yearly": {}},
        extra_regressors={"temp": {}, "noise": {}, "tiny": {}, "twin": {}},
        component_modes={
            "additive": ["yearly", "temp", "noise", "tiny", "twin"],
            "multiplicative": [],
        },
    )
    assert sea.offsetting_regressors(model, rows, "yearly") == ["temp"]
    assert sea.offsetting_regressors(model, rows.head(2), "yearly") == []
    assert sea.offsetting_regressors(model, rows, "missing") == []
    flat = rows.assign(yearly=0.0)
    assert sea.offsetting_regressors(model, flat, "yearly") == []


def test_the_observed_range_is_read_from_the_training_values() -> None:
    model = toy_model(history=pd.DataFrame({"y": [3.0, np.nan, 9.0, 5.0]}))
    assert sea.observed_range(model) == (3.0, 9.0)
    assert sea.observed_range(toy_model(history=None)) is None
    assert sea.observed_range(toy_model(history=pd.DataFrame({"y": [np.nan]}))) is None


def test_the_section_warns_when_a_regressor_cancels_a_swing_wider_than_the_data() -> (
    None
):
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("temperature")
    with no_warnings():
        section = check_section(sea.build(build_context(m, fcst, df=df)))
    assert "swings by more than the whole range of the data" in section.body
    assert "the temperature regressor largely offsets it" in section.body
    assert section.summary.endswith("wider than the whole range of the data).")
    assert len(section.summary.split()) <= 32
    # An ordinary fit is not warned about.
    quiet_m, quiet_df, quiet_fcst = fit_case("linear")
    with no_warnings():
        quiet = check_section(
            sea.build(build_context(quiet_m, quiet_fcst, df=quiet_df))
        )
    assert "swings by" not in quiet.body and "wider than" not in quiet.summary


def test_a_multiplicative_swing_past_the_trend_is_warned_about() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("multiplicative")
    with no_warnings():
        section = check_section(sea.build(build_context(m, fcst, df=df)))
    assert "more than the trend itself" in section.body
    assert section.summary.endswith("larger than the trend itself).")


# ---- pure: observed averages ------------------------------------------------


def test_intraday_grid_averages_by_weekday_and_hour() -> None:
    grid = sea.observed_grid(hourly_frame(), pd.Timedelta(hours=1), None)
    assert grid is not None and grid.kind == "weekday_hour"
    assert grid.rows == WEEK and len(grid.cols) == 24 and grid.cols[14] == "14:00"
    assert grid.mean[2, 14] == 2 * 100 + 14  # Wednesday at 14:00
    assert (grid.count == 3).all()
    assert grid.cell(2, 14) == "Wednesday at 14:00"


def test_five_minute_data_is_averaged_into_hour_cells() -> None:
    ds = pd.date_range("2024-01-01", periods=15 * 288, freq=pd.Timedelta(minutes=5))
    obs = pd.DataFrame({"ds": ds, "y": ds.hour * 1.0})
    grid = sea.observed_grid(obs, pd.Timedelta(minutes=5), None)
    assert grid is not None and grid.rows == WEEK and (grid.count >= 24).all()
    assert grid.mean[0, 5] == 5.0


def test_a_couple_of_days_of_intraday_data_get_no_weekday_by_hour_grid() -> None:
    # Each cell would hold a dozen readings from one day: the series restated, not a pattern.
    ds = pd.date_range("2024-01-01", periods=3 * 288, freq=pd.Timedelta(minutes=5))
    obs = pd.DataFrame({"ds": ds, "y": ds.hour * 1.0})
    assert sea.observed_grid(obs, pd.Timedelta(minutes=5), None) is None
    just_short = ds[: 14 * 288 - 1]
    assert (
        sea.observed_grid(
            pd.DataFrame({"ds": just_short, "y": just_short.hour * 1.0}),
            pd.Timedelta(minutes=5),
            None,
        )
        is None
    )


def test_the_grid_note_names_the_periods_that_pull_a_cell() -> None:
    hourly = sea.observed_grid(hourly_frame(), pd.Timedelta(hours=1), None)
    assert hourly is not None
    assert "the busier weeks pull their cells" in sea._grid_note(hourly, False)
    assert "years" not in sea._grid_note(hourly, False)
    monthly = sea.observed_grid(
        daily_frame("2023-01-01", 400), pd.Timedelta(days=1), None
    )
    assert monthly is not None
    assert "the busier years or weeks pull their cells" in sea._grid_note(
        monthly, False
    )


def test_grids_leave_out_skipped_weekdays_and_hours() -> None:
    obs = hourly_frame()
    skip = skip_axis(obs["ds"], skip_days="weekends", skip_hours=range(0, 8))
    kept = obs[skip.keeps(obs["ds"])]
    grid = sea.observed_grid(kept, pd.Timedelta(hours=1), skip)
    assert grid is not None
    assert grid.rows == WEEK[:5] and grid.cols[0] == "08:00" and len(grid.cols) == 16


def test_grid_drops_weekdays_and_hours_the_data_never_has() -> None:
    obs = hourly_frame()
    obs = obs[(obs["ds"].dt.dayofweek < 5) & (obs["ds"].dt.hour >= 9)]
    grid = sea.observed_grid(obs, pd.Timedelta(hours=1), None)
    assert grid is not None and grid.rows == WEEK[:5] and grid.cols[0] == "09:00"


def test_daily_grid_averages_by_month_and_weekday() -> None:
    obs = daily_frame("2023-01-01", 400)
    grid = sea.observed_grid(obs, pd.Timedelta(days=1), None)
    assert grid is not None and grid.kind == "month_weekday"
    assert grid.rows == WEEK and grid.cols[:3] == ("Jan", "Feb", "Mar")
    # y is weekday * 10 + month, so every cell mean is known.
    assert grid.mean[4, 2] == 4 * 10 + 3  # Friday in March
    assert grid.cell(4, 2) == "Fridays in March"
    jan = obs[obs["ds"].dt.month == 1]
    assert grid.count[0, 0] == (jan["ds"].dt.dayofweek == 0).sum()


@pytest.mark.parametrize(
    ("obs", "step"),
    [
        (daily_frame("2023-01-01", 200), pd.Timedelta(days=1)),  # under ten months
        (daily_frame("2020-01-01", 800).iloc[::7], pd.Timedelta(days=7)),  # weekly
        (daily_frame("2020-01-01", 900).iloc[::31], pd.Timedelta(days=31)),  # monthly
        (hourly_frame(1), pd.Timedelta(hours=1)),  # one weekday only
        (daily_frame("2023-01-01", 0), pd.Timedelta(days=1)),  # nothing
    ],
)
def test_no_grid_when_the_data_cannot_support_one(
    obs: pd.DataFrame, step: pd.Timedelta
) -> None:
    assert sea.observed_grid(obs, step, None) is None


def test_grid_extremes_sentence_names_both_cells_and_their_support() -> None:
    grid = sea.observed_grid(hourly_frame(), pd.Timedelta(hours=1), None)
    assert grid is not None
    assert sea.grid_extremes_sentence(grid) == (
        "The highest average is Sunday at 23:00 (623, from 3 observations) "
        "and the lowest is Monday at 00:00 (0, from 3 observations)."
    )


def test_grid_extremes_sentence_does_not_name_one_cell_twice() -> None:
    obs = daily_frame("2023-01-01", 400).assign(y=5.0)
    grid = sea.observed_grid(obs, pd.Timedelta(days=1), None)
    assert grid is not None
    assert sea.grid_extremes_sentence(grid) == "Every cell has the same average."


def test_daily_grid_has_no_columns_for_months_without_observations() -> None:
    grid = sea.observed_grid(daily_frame("2023-01-01", 310), pd.Timedelta(days=1), None)
    assert grid is not None
    # 310 days from 1 January stop in early November.
    assert grid.cols[0] == "Jan" and grid.cols[-1] == "Nov" and "Dec" not in grid.cols
    assert grid.mean.shape == (7, 11) and (grid.count > 0).any(axis=0).all()


def test_observed_calendar_lists_the_weekdays_and_hours_present() -> None:
    obs = hourly_frame()
    obs = obs[(obs["ds"].dt.dayofweek < 5) & (obs["ds"].dt.hour >= 9)]
    assert sea.observed_calendar(obs["ds"], True) == (
        frozenset(range(5)),
        frozenset(range(9, 24)),
    )
    days, hours = sea.observed_calendar(obs["ds"], False)
    assert days == frozenset(range(5)) and hours == HOURS


# ---- pure: daily values and the calendar ------------------------------------


def test_daily_values_mark_forecast_days_after_the_last_observed_day() -> None:
    obs = daily_frame("2024-01-01", 30)
    fcst = pd.DataFrame({"ds": pd.date_range("2024-01-15", periods=21), "yhat": 7.0})
    days = sea.daily_values(obs, fcst, pd.Timedelta(days=1))
    assert days is not None
    assert list(days.columns) == ["day", "value", "forecast"]
    assert len(days) == 35 and int(days["forecast"].sum()) == 5
    assert days.loc[days["forecast"], "value"].eq(7.0).all()
    assert days.loc[~days["forecast"], "day"].max() == pd.Timestamp("2024-01-30")


def test_daily_values_average_intraday_data_and_drop_partial_days() -> None:
    ds = pd.date_range(
        "2024-01-01", periods=8 * 288 + 100, freq=pd.Timedelta(minutes=5)
    )
    obs = pd.DataFrame({"ds": ds, "y": 1.0})
    ahead = pd.date_range(
        ds[-1] + pd.Timedelta(minutes=5),
        periods=2 * 288 + 20,
        freq=pd.Timedelta(minutes=5),
    )
    fcst = pd.DataFrame({"ds": ahead, "yhat": 3.0})
    days = sea.daily_values(obs, fcst, pd.Timedelta(minutes=5))
    assert days is not None
    # The ninth day is only a third observed and the forecast's last day is partial: one full forecast day remains.
    assert list(days["day"]) == [
        *pd.date_range("2024-01-01", periods=8),
        pd.Timestamp("2024-01-10"),
    ]
    assert list(days["forecast"]) == [False] * 8 + [True]


def test_forecast_days_leave_out_weekdays_the_data_never_has() -> None:
    obs = daily_frame("2024-01-01", 28)
    obs = obs[obs["ds"].dt.dayofweek < 5]
    fcst = pd.DataFrame({"ds": pd.date_range("2024-01-29", periods=14), "yhat": 7.0})
    days = sea.daily_values(obs, fcst, pd.Timedelta(days=1))
    assert days is not None
    ahead = days.loc[days["forecast"], "day"]
    assert len(ahead) == 10 and (ahead.dt.dayofweek < 5).all()
    cal = sea.calendar_layout(days)
    assert cal is not None and cal.weekdays == tuple(range(5))


def test_forecast_days_leave_out_hours_the_data_never_has() -> None:
    ds = pd.date_range("2024-01-01", periods=10 * 24, freq=pd.Timedelta(hours=1))
    obs = pd.DataFrame({"ds": ds, "y": 1.0})
    obs = obs[(obs["ds"].dt.hour >= 9) & (obs["ds"].dt.hour < 17)]
    ahead = pd.date_range("2024-01-11", periods=3 * 24, freq=pd.Timedelta(hours=1))
    fcst = pd.DataFrame({"ds": ahead, "yhat": ahead.hour * 1.0})
    days = sea.daily_values(obs, fcst, pd.Timedelta(hours=1))
    assert days is not None
    # Hours 0 to 8 and 17 to 23 are extrapolations: only the business hours enter the daily mean.
    values = days.loc[days["forecast"], "value"]
    assert len(values) == 3 and values.eq(np.mean(range(9, 17))).all()


def test_daily_values_need_daily_or_finer_data() -> None:
    obs = daily_frame("2024-01-01", 60).iloc[::7]
    fcst = pd.DataFrame({"ds": pd.to_datetime(["2024-04-01"]), "yhat": [1.0]})
    assert sea.daily_values(obs, fcst, pd.Timedelta(days=7)) is None
    assert sea.daily_values(obs.iloc[0:0], fcst, pd.Timedelta(days=1)) is None


def three_years(forecast_from: str = "2024-03-01") -> pd.DataFrame:
    day = pd.date_range("2021-06-01", "2024-03-31")
    return pd.DataFrame(
        {
            "day": day,
            "value": day.dayofyear * 1.0,
            "forecast": day >= pd.Timestamp(forecast_from),
        }
    )


def test_calendar_keeps_three_years_and_puts_days_in_week_columns() -> None:
    cal = sea.calendar_layout(three_years())
    assert cal is not None
    assert [b.year for b in cal.blocks] == [2022, 2023, 2024]
    assert cal.weekdays == tuple(range(7)) and cal.n_rows == 3 * 8 - 1
    # 1 January 2023 is a Sunday in the week that starts on Monday 26 December 2022 (column 0).
    top = cal.blocks[1].top
    assert top == 8 and cal.observed[top + 6, 0] == 1.0
    assert cal.text_observed[top + 6][0] == "Sun 1 Jan 2023"
    assert cal.observed[top, 1] == 2.0  # Monday 2 January, the next column
    assert np.isnan(cal.observed[top, 0])  # Monday 26 December belongs to 2022
    assert ("Jan", 0) in cal.blocks[1].months and ("Feb", 5) in cal.blocks[1].months


def test_calendar_draws_forecast_days_in_their_own_matrix_with_a_cut_between() -> None:
    cal = sea.calendar_layout(three_years())
    assert cal is not None and cal.has_forecast
    top = cal.blocks[2].top
    # 29 February 2024 (Thursday, week 8) is the last observed day; 1 March is forecast.
    assert cal.observed[top + 3, 8] == pd.Timestamp("2024-02-29").dayofyear
    assert (
        np.isnan(cal.forecast[top + 3, 8])
        and cal.forecast[top + 4, 8] == pd.Timestamp("2024-03-01").dayofyear
    )
    assert cal.text_forecast[top + 4][8] == "Fri 1 Mar 2024 (forecast)"
    assert not np.isfinite(cal.forecast[: top - 1]).any()
    assert cal.cut == sea.stair_cut(8, 3, 7, top)


def test_calendar_without_forecast_or_with_it_in_a_new_year_has_no_cut() -> None:
    assert sea.calendar_layout(three_years("2099-01-01")).cut is None  # type: ignore[union-attr]
    day = pd.date_range("2022-01-01", "2023-01-10")
    days = pd.DataFrame(
        {"day": day, "value": 1.0, "forecast": day >= pd.Timestamp("2023-01-01")}
    )
    cal = sea.calendar_layout(days)
    assert cal is not None and cal.has_forecast and cal.cut is None


def test_calendar_rows_follow_the_weekdays_that_exist() -> None:
    days = three_years("2099-01-01")
    days = days[~days["day"].dt.dayofweek.isin([5, 6])]
    cal = sea.calendar_layout(days, frozenset({5, 6}))
    assert (
        cal is not None and cal.weekdays == tuple(range(5)) and cal.n_rows == 3 * 6 - 1
    )
    # Friday 7 January 2022 is the first Friday of the first block's column 1 (week of Monday 3 January).
    assert cal.observed[4, 1] == pd.Timestamp("2022-01-07").dayofyear
    assert sea.calendar_layout(days, frozenset(range(5)) | {5, 6}) is None


def test_calendar_needs_a_few_weeks_of_days() -> None:
    days = three_years().head(sea.MIN_CALENDAR_DAYS - 1)
    assert sea.calendar_layout(days) is None
    assert sea.calendar_layout(days.iloc[0:0]) is None


def test_stair_cut_follows_the_edge_between_observed_and_forecast_cells() -> None:
    assert sea.stair_cut(8, 3, 7) == ((8.5, -0.5), (8.5, 3.5), (7.5, 3.5), (7.5, 6.5))
    assert sea.stair_cut(8, 3, 7, top=16)[0] == (8.5, 15.5)
    # The last row has no cells below it to separate, so the cut stops at the bottom edge.
    assert sea.stair_cut(2, 6, 7) == ((2.5, -0.5), (2.5, 6.5))


def test_colour_limits_ignore_spikes_and_survive_constants() -> None:
    values = np.concatenate([np.linspace(10, 20, 200), [5000.0]])
    low, high = sea.colour_limits(values)
    assert 10 <= low < 11 and 19 < high < 21
    assert sea.colour_limits(np.array([3.0, 3.0])) == (2.5, 3.5)
    assert sea.colour_limits(np.array([np.nan])) == (0.0, 1.0)
    assert sea.colour_limits(np.array([1.0, 5.0])) == (1.0, 5.0)


# ---- section: smoke matrix --------------------------------------------------


def check_section(section: Section | None) -> Section:
    assert section is not None
    assert (section.key, section.title) == ("seasonality", "Seasonal patterns")
    assert "—" not in section.body and "–" not in section.body
    assert "—" not in section.summary and "–" not in section.summary
    assert "\n" not in section.summary and section.summary.endswith(".")
    assert len(section.body.encode()) < 1_000_000
    assert section.body.count("Plotly.newPlot") >= 1
    return section


@pytest.mark.parametrize("case", CASES)
def test_section_across_the_fixture_models(case: str) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case(case)
    ctx = build_context(m, fcst, df=df)
    with no_warnings():
        section = check_section(sea.build(ctx))
    body = section.body
    assert "What moves the fitted values" in body and "Peaks and lows" in body
    # Five-minute data over three days is the series itself, not a weekday pattern.
    assert "Average by weekday and hour" not in body
    daily = case != "subdaily"
    assert ("Average by month and weekday" in body) == daily
    assert ("Day by day" in body) == daily  # five-minute data spans only three days
    assert "Peaks and lows" in body and "Swing" in body


def test_multiplicative_patterns_are_given_in_percent_of_the_trend() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("multiplicative")
    with no_warnings():
        section = check_section(sea.build(build_context(m, fcst, df=df)))
    assert "% of the trend" in section.body
    assert "Weekly" in section.body and "Yearly" in section.body


def test_regressors_and_custom_seasonalities_get_their_own_rows() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("temperature")
    with no_warnings():
        body = check_section(sea.build(build_context(m, fcst, df=df))).body
    assert "Temperature (regressor)" in body and "Promo (regressor)" in body
    m, df, fcst = fit_case("no_regressors_custom_seasonality")
    with no_warnings():
        body = check_section(sea.build(build_context(m, fcst, df=df))).body
    assert "Monthly (30.5-day cycle)" in body
    assert "highest " in body and " days in" in body


@pytest.mark.parametrize(
    "case", ["linear", "subdaily", "temperature", "multiplicative"]
)
def test_every_colour_in_the_figures_has_a_dark_step(case: str) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case(case)
    with no_warnings():
        body = check_section(sea.build(build_context(m, fcst, df=df))).body
    colours = set(re.findall(r"#[0-9a-fA-F]{6}\b|rgba?\([^)]*\)", body))
    assert colours
    assert {c for c in colours if c.upper().replace(" ", "") not in DARK_SWAP} == set()


@lru_cache(maxsize=None)
def holiday_case() -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    """Country holidays and a conditional seasonality, which the fixture matrix lacks."""
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"])
    df["on"] = df["ds"].dt.month.isin([1, 2, 3])
    m = Prophet(holidays_mode="multiplicative")
    m.add_country_holidays("US")
    m.add_seasonality("seasonal_on", period=7, fourier_order=2, condition_name="on")
    m.fit(df)
    future = m.make_future_dataframe(periods=30)
    future["on"] = future["ds"].dt.month.isin([1, 2, 3])
    return m, df, m.predict(future)


def test_holidays_and_conditional_seasonalities_are_included() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = holiday_case()
    ctx = build_context(m, fcst, df=df)
    with no_warnings():
        body = check_section(sea.build(ctx)).body
    names = {i.name for i in sea.component_importance(m, fcst, ctx.history_end)}
    assert {"trend", "yearly", "weekly", "seasonal_on", "holidays"} <= names
    assert "Holidays" in body and "Seasonal on" in body


def test_section_does_not_touch_the_context_frames() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    ctx = build_context(m, fcst, df=df)
    before = (ctx.fcst.copy(), ctx.history.copy(), df.copy())
    with no_warnings():
        sea.build(ctx)
    pd.testing.assert_frame_equal(ctx.fcst, before[0])
    pd.testing.assert_frame_equal(ctx.history, before[1])
    pd.testing.assert_frame_equal(df, before[2])


def test_intraday_section_reads_hours_in_its_facts() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("subdaily")
    with no_warnings():
        body = check_section(sea.build(build_context(m, fcst, df=df))).body
    assert "Daily" in body and "Hourly (1-hour cycle)" in body and ":00" in body
    # Only the weekly curve cares about missing weekdays, and this model has none.
    assert "never has them" not in without_scripts(body)


def peaks_table_headers(body: str) -> list[tuple[str, bool]]:
    """Header texts of the peaks table with whether each is right-aligned."""
    head = re.search(r"<thead><tr>(.*?)</tr></thead>", body)
    assert head is not None
    return [
        (html.unescape(text), cls == ' class="num"')
        for cls, text in re.findall(r"<th( class=\"num\")?>(.*?)</th>", head.group(1))
    ]


@pytest.mark.parametrize("case", ["linear", "temperature"])
def test_numeric_columns_of_the_peaks_table_sit_together_on_the_right(
    case: str,
) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case(case)
    with no_warnings():
        body = check_section(sea.build(build_context(m, fcst, df=df))).body
    assert peaks_table_headers(body) == [
        ("Pattern", False),
        ("Highest", False),
        ("Lowest", False),
        ("Swing", True),
        ("Share of movement", True),
    ]
    assert "table above" in body  # the heatmap note points at a table that is there


def test_multiplicative_fixture_does_not_credit_the_movement_to_a_flat_weekly_curve() -> (
    None
):
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("multiplicative")
    ctx = build_context(m, fcst, df=df)
    items = {i.name: i for i in sea.component_importance(m, fcst, ctx.history_end)}
    # data.csv has no weekends, so the weekly curve sits at about 4.5: that level is not movement.
    assert items["weekly"].share < 0.05
    assert max(items.values(), key=lambda i: i.share).name == "trend"
    with no_warnings():
        section = check_section(sea.build(ctx))
    assert not section.summary.startswith("The weekly pattern")
    assert "weekly pattern carries most" not in section.body


@lru_cache(maxsize=None)
def high_level_case() -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    """Daily data near 10 000 that move by about 10, nearly all of it a weekly pattern."""
    rng = np.random.default_rng(0)
    ds = pd.date_range("2022-01-01", periods=400)
    y = (
        10_000
        + np.array([0, 5, 10, 15, 20, 25, 8])[ds.dayofweek]
        + rng.normal(0, 2, len(ds))
    )
    df = pd.DataFrame({"ds": ds, "y": y})
    m = Prophet(
        yearly_seasonality=False, daily_seasonality=False, uncertainty_samples=0
    )
    m.fit(df)
    return m, df, m.predict(m.make_future_dataframe(periods=30))


def test_a_dominant_pattern_on_a_high_level_is_not_called_flat() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = high_level_case()
    ctx = build_context(m, fcst, df=df)
    with no_warnings():
        section = check_section(sea.build(ctx))
    assert "peaking on Saturday" in section.summary
    assert "close to flat" not in section.body and "No clear peak" not in section.body
    (weekly,) = sea.seasonal_facts(
        m, fcst, ctx.history_end, ctx.step, frozenset(range(7)), HOURS
    )
    assert not weekly.flat and weekly.swing > 20
    # The distances are from the average of the curve, not from the 10 000 level.
    assert weekly.high_value > 0 > weekly.low_value
    assert max(abs(weekly.high_value), abs(weekly.low_value)) < weekly.swing


def test_a_model_without_seasonalities_does_not_point_at_a_missing_table() -> None:
    pytest.importorskip("plotly")
    ds = pd.date_range("2021-01-01", periods=800)
    df = pd.DataFrame({"ds": ds, "y": np.arange(800) % 50 + 3.0})
    m = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
    )
    m.fit(df)
    fcst = m.predict(m.make_future_dataframe(periods=10))
    with no_warnings():
        section = sea.build(build_context(m, fcst, df=df))
    assert section is not None and section.summary == ""  # nothing to lift to the top
    assert "Average by month and weekday" in section.body
    assert "Peaks and lows" not in section.body and "table above" not in section.body


def test_the_calendar_has_no_rows_for_weekdays_the_data_never_has() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("logistic")  # data.csv has no weekends and no skip is given
    assert not df["ds"].dt.dayofweek.ge(5).any()
    assert fcst["ds"].dt.dayofweek.ge(5).any()
    with no_warnings():
        body = check_section(sea.build(build_context(m, fcst, df=df))).body
    assert "Day by day" in body
    assert re.search(r"S(at|un) \d+ \w{3} \d{4}", body) is None
    assert re.search(r"(Mon|Fri) \d+ \w{3} \d{4} \(forecast\)", body)


def test_the_calendar_scrolls_sideways_instead_of_shrinking_on_a_narrow_screen() -> (
    None
):
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    with no_warnings():
        body = check_section(sea.build(build_context(m, fcst, df=df))).body
    wrapper = (
        f'<div class="tbl-wrap"><div style="min-width:{sea.CALENDAR_MIN_WIDTH}px">'
    )
    assert body.count(wrapper) == 1
    # Only the calendar is wrapped, and it sits after its heading and before the closing note.
    start = body.index(wrapper)
    assert body.index("Day by day") < start < body.index("One block per year")
    assert "plotly-graph-div" in body[start : body.index("One block per year")]
    assert body.count("min-width") == 1


# ---- section: unusual inputs ------------------------------------------------


def test_forecast_with_only_future_rows_still_builds() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    future = fcst[fcst["ds"] > df["ds"].max()]
    assert not future.empty
    ctx = build_context(m, future, df=df)
    with no_warnings():
        section = check_section(sea.build(ctx))
    # No in-sample rows: the comparison falls back to the forecast rows and says so.
    assert "in the forecast" in section.body
    assert "Peaks and lows" in section.body


def test_forecast_with_only_in_sample_rows_has_no_forecast_marks() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    ctx = build_context(m, fcst[fcst["ds"] <= df["ds"].max()], df=df)
    with no_warnings():
        section = check_section(sea.build(ctx))
    assert (
        "Forecast starts" not in section.body and "smaller squares" not in section.body
    )
    assert "Day by day" in section.body


def test_weekday_only_data_with_skipped_weekends() -> None:
    pytest.importorskip("plotly")
    m, df, _ = fit_case("linear")
    skip = skip_axis(df["ds"], skip_days="weekends")
    future = m.make_future_dataframe(periods=30)
    future = future[skip.keeps(future["ds"])]
    fcst = m.predict(future)
    ctx = build_context(m, fcst, df=df, skip=skip)
    with no_warnings():
        section = check_section(sea.build(ctx))
    body = section.body
    assert "Saturdays and Sundays are left out" in body
    for day in ("Monday", "Friday"):
        assert day in body
    assert "Saturday" not in without_scripts(body).replace(
        "Saturdays and Sundays are left out", ""
    )
    assert "Sat 1" not in body and "Sun 2" not in body
    weekly = [
        f
        for f in sea.seasonal_facts(
            m, fcst, ctx.history_end, ctx.step, frozenset(range(5)), HOURS
        )
        if f.kind == "weekly"
    ]
    assert weekly and weekly[0].high in WEEK[:5] and weekly[0].low in WEEK[:5]


@lru_cache(maxsize=None)
def short_case() -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    """A model fitted on about three months of data, with a custom seasonality named to test escaping."""
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"]).head(60)
    m = Prophet()
    m.add_seasonality(name="b<i>g", period=14, fourier_order=2)
    m.fit(df)
    return m, df, m.predict(m.make_future_dataframe(periods=14))


def test_short_history_has_no_month_grid_but_keeps_the_calendar_and_escapes_names() -> (
    None
):
    pytest.importorskip("plotly")
    m, df, fcst = short_case()
    with no_warnings():
        section = check_section(sea.build(build_context(m, fcst, df=df)))
    body = section.body
    assert "Average by month and weekday" not in body and "Day by day" in body
    assert "B&lt;i&gt;g" in body and "B<i>g" not in without_scripts(body)
    assert html.escape("B<i>g") in without_scripts(body)
    assert "B<i>g" not in body  # plotly escapes the angle brackets inside its JSON too


def test_section_is_none_when_nothing_applies() -> None:
    pytest.importorskip("plotly")
    # Weekly data with every seasonality switched off: no patterns, no grid and no calendar.
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"]).iloc[::7].head(60)
    m = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
    )
    m.fit(df)
    fcst = m.predict(m.make_future_dataframe(periods=8, freq="7D"))
    with no_warnings():
        assert sea.build(build_context(m, fcst, df=df)) is None
