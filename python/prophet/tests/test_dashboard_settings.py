"""The "settings" section of the dashboard: plain-words facts and the code that recreates the model."""

import ast
import html
import inspect
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet, skip_axis
from prophet.dashboard import settings
from prophet.dashboard._context import build_context
from prophet.tests.test_visuals import CASES, fit_case, no_warnings

DATA = Path(__file__).parent
EM_DASH = "\u2014"
DAY = pd.Timedelta(days=1)
ALL_CASES = CASES + ["temperature"]
SIGNATURES = {
    "Prophet": inspect.signature(Prophet.__init__),
    "add_seasonality": inspect.signature(Prophet.add_seasonality),
    "add_regressor": inspect.signature(Prophet.add_regressor),
    "add_country_holidays": inspect.signature(Prophet.add_country_holidays),
}


# ---- pure: wording ----------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "words"),
    [
        (365.25, "1 year"),
        (730.5, "2 years"),
        (30.5, "1 month"),
        (91.3125, "3 months"),
        (7, "1 week"),
        (14, "2 weeks"),
        (1, "1 day"),
        (30, "30 days"),
        (45, "45 days"),
        (0.5, "12 hours"),
        (1 / 24, "1 hour"),
        (1 / 1440, "1 minute"),
        (2.7, "2.7 days"),
        (29.53, "29.5 days"),
    ],
)
def test_period_words(days: float, words: str) -> None:
    assert settings.period_words(days) == words


def test_against_default_says_which_way_and_by_how_much() -> None:
    assert (
        settings.against_default(0.05, 0.05, "more flexible", "stiffer")
        == "0.05, the default"
    )
    assert (
        settings.against_default(0.5, 0.05, "more flexible", "stiffer")
        == "0.5, more flexible than the default (0.05)"
    )
    assert (
        settings.against_default(0.01, 0.05, "more flexible", "stiffer")
        == "0.01, stiffer than the default (0.05)"
    )


def test_join_words() -> None:
    assert settings.join_words([]) == ""
    assert settings.join_words(["a"]) == "a"
    assert settings.join_words(["a", "b"]) == "a and b"
    assert settings.join_words(["a", "b", "c"]) == "a, b and c"


def test_standardised_words_read_what_prophet_left_behind() -> None:
    assert (
        settings.standardised_words({"standardize": "auto", "mu": 3.0, "std": 2.0})
        == "yes"
    )
    assert (
        settings.standardised_words({"standardize": "auto", "mu": 0.0, "std": 1.0})
        == "no (binary or constant)"
    )
    assert (
        settings.standardised_words({"standardize": False, "mu": 0.0, "std": 1.0})
        == "no"
    )


def test_seasonality_and_regressor_rows_put_text_before_numbers() -> None:
    rows = settings.seasonality_rows(
        {
            "monthly": {
                "period": 30.5,
                "fourier_order": 3,
                "prior_scale": 2.5,
                "mode": "multiplicative",
                "condition_name": "on",
            },
            "weekly": {
                "period": 7,
                "fourier_order": 3,
                "prior_scale": 10.0,
                "mode": "additive",
                "condition_name": None,
            },
        }
    )
    assert settings.SEASONALITY_HEADERS == [
        "Name", "Period", "Mode", "Condition", "Fourier order", "Prior scale"
    ]  # fmt: skip
    assert rows == [
        ["monthly", "1 month", "multiplicative", "on", "3", "2.5"],
        ["weekly", "1 week", "additive", "none", "3", "10"],
    ]
    regressors = settings.regressor_rows(
        {
            "temperature": {
                "prior_scale": 10.0,
                "mode": "additive",
                "standardize": "auto",
                "mu": 9.0,
                "std": 5.0,
            },
            "promo": {
                "prior_scale": 0.5,
                "mode": "multiplicative",
                "standardize": "auto",
                "mu": 0.0,
                "std": 1.0,
            },
        },
        {"temperature": "°C"},
    )
    assert settings.REGRESSOR_HEADERS == ["Name", "Mode", "Standardised", "Prior scale"]
    assert regressors == [
        ["temperature (°C)", "additive", "yes", "10"],
        ["promo", "multiplicative", "no (binary or constant)", "0.5"],
    ]


def test_growth_and_changepoint_facts() -> None:
    linear = dict(settings.trend_pairs(Prophet(), None, DAY))
    assert linear["Growth"].startswith("Linear")
    assert linear["Changepoint prior scale"] == "0.05, the default"
    flat = dict(settings.trend_pairs(Prophet(growth="flat"), None, DAY))
    assert flat["Changepoints"].startswith("Not used")
    assert "Changepoint prior scale" not in flat
    by_hand = Prophet(
        changepoints=["2024-02-01", "2024-03-01"], changepoint_prior_scale=0.5
    )
    facts = dict(settings.trend_pairs(by_hand, None, DAY))
    assert facts["Changepoints"] == "2 chosen by hand, from 1 Feb 2024 to 1 Mar 2024"
    assert (
        facts["Changepoint prior scale"] == "0.5, more flexible than the default (0.05)"
    )
    none = Prophet(changepoints=[])
    assert dict(settings.trend_pairs(none, None, DAY))["Changepoints"].startswith(
        "None"
    )


def test_logistic_facts_read_cap_and_floor_from_the_history() -> None:
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=3),
            "cap": [10.0, 10.0, 12.0],
            "floor": [1.0, 1.0, 2.0],
        }
    )
    plain = Prophet(growth="logistic")
    facts = dict(settings.trend_pairs(plain, history, DAY))
    assert facts["Capacity (cap)"] == "10 to 12, varying by date"
    assert facts["Floor"] == "0, the default"
    floored = Prophet(growth="logistic")
    floored.logistic_floor = True
    facts = dict(settings.trend_pairs(floored, history, DAY))
    assert facts["Floor"] == "1 to 2, varying by date"
    constant = history.assign(cap=10.0)
    assert dict(settings.trend_pairs(plain, constant, DAY))["Capacity (cap)"] == "10"


def scaled_logistic(
    scaling: str, floor: float | None = None
) -> tuple[Prophet, pd.DataFrame]:
    """A logistic model with its scales set from data, the way ``fit`` does, without fitting."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2022-01-01", periods=200),
            "y": 17.7 + rng.random(200) * 20.5,
            "cap": 57.3,
        }
    )
    if floor is not None:
        df["floor"] = floor
    m = Prophet(growth="logistic", scaling=scaling)  # type: ignore[arg-type]
    m.history = m.setup_dataframe(df, initialize_scales=True)
    return m, m.history


def test_logistic_floor_is_the_lowest_y_under_min_max_scaling() -> None:
    m, history = scaled_logistic("minmax")
    assert not m.logistic_floor
    assert history["floor"].nunique() == 1  # what Prophet really uses
    facts = dict(settings.trend_pairs(m, history, DAY))
    assert facts["Floor"] == (
        f"{settings._fmt_number(float(m.y_min or 0))}, the lowest value of y (min-max scaling)"
    )
    assert float(history["floor"].iloc[0]) == pytest.approx(float(m.y_min or 0))
    absmax, history = scaled_logistic("absmax")
    assert dict(settings.trend_pairs(absmax, history, DAY))["Floor"] == "0, the default"
    floored, history = scaled_logistic("minmax", floor=10.0)
    assert dict(settings.trend_pairs(floored, history, DAY))["Floor"] == "10"


def test_fitting_facts_name_the_method_and_intervals() -> None:
    sampled = dict(
        settings.fitting_pairs(Prophet(mcmc_samples=300, interval_width=0.95))
    )
    assert sampled["Fitting method"] == "MCMC sampling with 300 samples"
    assert sampled["Uncertainty"] == "95% intervals from 1,000 simulated futures"
    assert sampled["Backend"] == "CmdStan through CmdStanPy"
    plain = dict(settings.fitting_pairs(Prophet(uncertainty_samples=0)))
    assert plain["Fitting method"].startswith("MAP optimisation")
    assert plain["Uncertainty"] == "Not estimated: uncertainty samples is 0"
    assert plain["Scaling"].startswith("Absolute maximum")
    assert dict(settings.fitting_pairs(Prophet(scaling="minmax")))[
        "Scaling"
    ].startswith("Min-max")


def test_scaling_words_quote_the_fitted_scale() -> None:
    m, _, _ = fit_case("linear")
    assert settings.scaling_words(m) == "Absolute maximum: y is divided by 72"
    scaled = Prophet(scaling="minmax")
    scaled.y_min, scaled.y_scale = 2.0, 30.0
    assert (
        settings.scaling_words(scaled) == "Min-max: y is shifted by 2 and divided by 30"
    )


def test_scaling_words_take_the_floor_off_first() -> None:
    absmax, _ = scaled_logistic("absmax", floor=10.0)
    assert absmax.logistic_floor
    assert settings.scaling_words(absmax) == (
        f"Absolute maximum: y minus the floor is divided by "
        f"{settings._fmt_number(float(absmax.y_scale or 0))}"
    )
    minmax, _ = scaled_logistic("minmax", floor=10.0)
    assert settings.scaling_words(minmax).startswith(
        "Min-max: y minus the floor is divided by"
    )
    unfitted = Prophet(growth="logistic")
    unfitted.logistic_floor = True
    assert settings.scaling_words(unfitted) == (
        "Absolute maximum: y minus the floor is divided by its largest absolute value"
    )
    plain, _ = scaled_logistic("absmax")
    assert "minus the floor" not in settings.scaling_words(plain)
    assert settings.scaling_words(plain).startswith("Absolute maximum: y is divided by")


@pytest.mark.parametrize(
    ("lower", "upper", "words"),
    [
        ([-1, -1], [2, 2], "1 day before to 2 days after"),
        ([-3], [0], "3 days before to the day itself"),
        ([0, 0], [1, 1], "the day itself to 1 day after"),
        ([0], [0], "None: only the day itself"),
        (
            [-1, 0],
            [2, 3],
            "Varies by holiday: up to 1 day before and up to 3 days after",
        ),
        ([0, 0], [1, 4], "Varies by holiday: up to 4 days after"),
        ([-2, -1], [0, 0], "Varies by holiday: up to 2 days before"),
    ],
)
def test_holiday_windows_are_not_taken_for_one_holidays_window(
    lower: list[int], upper: list[int], words: str
) -> None:
    assert settings.window_words(pd.Series(lower), pd.Series(upper)) == words


def test_holiday_facts() -> None:
    assert settings.holiday_pairs(Prophet()) == [("Holidays", "None")]
    m, _ = rich_case()
    facts = dict(settings.holiday_pairs(m))
    assert facts["Your holidays"] == "2 names, 3 dates"
    # superbowl reaches 1 day either way, election 0 before and 2 after
    assert facts["Days around each date"] == (
        "Varies by holiday: up to 1 day before and up to 2 days after"
    )
    assert facts["Per-holiday prior scales"] == "5 to 15"
    assert facts["Country holidays"] == "US"
    assert facts["Holiday prior scale"] == "8, tighter than the default (10)"
    assert facts["Holiday mode"].startswith("Additive")
    assert facts["Holidays"].endswith("holiday names")


def test_lede_sums_the_model_up() -> None:
    m, _, _ = fit_case("regressors")
    assert settings.lede_text(m) == (
        "Linear growth with 25 changepoints, plus 2 seasonal patterns (yearly and weekly) "
        "and 2 extra regressors. It was fitted by MAP optimisation."
    )
    assert settings.lede_text(fit_case("flat")[0]).startswith(
        "Flat growth, plus 2 seasonal patterns"
    )
    rich, _ = rich_case()
    text = settings.lede_text(rich)
    assert text == (
        "Linear growth with 4 changepoints, plus 3 seasonal patterns "
        "(quarterly, winter_weekly and weekly), 14 holidays and 1 extra regressor. "
        "It was fitted by MAP optimisation."
    )
    bare = Prophet(changepoints=[], yearly_seasonality=False, weekly_seasonality=False)
    assert settings.lede_text(bare) == (
        "Linear growth with no changepoints, with no seasonality, holidays or regressors. "
        "It was fitted by MAP optimisation."
    )


# ---- pure: Python literals --------------------------------------------------


def test_py_literal() -> None:
    assert settings.py_literal(None) == "None"
    assert settings.py_literal(True) == "True"
    assert settings.py_literal(np.bool_(False)) == "False"
    assert settings.py_literal(np.int64(7)) == "7"
    assert settings.py_literal(np.float64(0.05)) == "0.05"
    assert settings.py_literal(1 / 3) == "0.3333333333333333"
    assert settings.py_literal("auto") == '"auto"'
    assert settings.py_literal('say "hi"\n') == '"say \\"hi\\"\\n"'
    assert settings.py_literal("café") == '"café"'
    assert settings.py_literal([1, "a", None]) == '[1, "a", None]'
    with pytest.raises(ValueError, match="no Python literal"):
        settings.py_literal(float("nan"))
    with pytest.raises(TypeError, match="dict"):
        settings.py_literal({})


def test_literals_evaluate_back_to_the_same_value() -> None:
    for value in ["x", 'a"b\\c', "éè", 3, 0.1 + 0.2, True, None, [1.5, "z"]]:
        assert ast.literal_eval(settings.py_literal(value)) == value


def test_call_source_wraps_only_when_it_has_to() -> None:
    assert (
        settings.call_source("m.add_regressor", [(None, '"x"')])
        == 'm.add_regressor("x")'
    )
    assert settings.call_source("m = Prophet", []) == "m = Prophet()"
    wide = settings.call_source(
        "m.add_seasonality",
        [
            ("name", '"a_rather_long_seasonality_name"'),
            ("period", "30.4375"),
            ("fourier_order", "12"),
            ("prior_scale", "0.123456"),
        ],
    )
    assert wide.splitlines()[0] == "m.add_seasonality("
    assert all(len(line) <= 88 for line in wide.splitlines())
    ast.parse(wide)


def test_long_lists_are_packed_into_lines() -> None:
    items = [settings.py_literal(f"2024-01-{d:02d}") for d in range(1, 29)]
    lines = settings._list_lines("changepoints = ", items, "", suffix="")
    assert len(lines) > 2 and all(len(line) <= 88 for line in lines)
    assert ast.literal_eval("\n".join(lines).split("=", 1)[1].strip()) == [
        f"2024-01-{d:02d}" for d in range(1, 29)
    ]


# ---- the code that recreates the model ---------------------------------------


def run_recreated(code: str, **namespace: Any) -> Prophet:
    scope: dict[str, Any] = dict(namespace)
    exec(compile(code, "<recreated>", "exec"), scope)  # noqa: S102
    clone = scope["m"]
    assert isinstance(clone, Prophet)
    assert clone.history is None
    return clone


def assert_valid_calls(code: str) -> None:
    """Every call to Prophet or its add_* methods binds to the real signature."""
    seen = 0
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if name in SIGNATURES:
            seen += 1
            SIGNATURES[name].bind(
                None,
                *[None] * len(node.args),
                **{kw.arg: None for kw in node.keywords if kw.arg},
            )
            assert all(kw.arg in SIGNATURES[name].parameters for kw in node.keywords)
    assert seen >= 1


def assert_same_settings(
    original: Prophet, clone: Prophet, ignore: frozenset[str] | set[str] = frozenset()
) -> None:
    """The unfitted ``clone`` would fit the same model as ``original``.

    ``ignore`` names constructor settings that only agree after both models see
    the same history.
    """
    for name in settings.init_defaults():
        if name in {"changepoints", "holidays", "stan_backend", *ignore}:
            continue
        assert getattr(clone, name) == getattr(original, name), name
    assert clone.specified_changepoints == original.specified_changepoints
    if original.specified_changepoints:
        assert list(clone.changepoints) == list(original.changepoints)
    assert clone.country_holidays == original.country_holidays
    assert (clone.holidays is None) == (original.holidays is None)
    if original.holidays is not None and clone.holidays is not None:
        columns = [c for c in original.holidays.columns if c in clone.holidays.columns]
        assert set(columns) >= {"holiday", "ds"}
        pd.testing.assert_frame_equal(
            clone.holidays[columns].reset_index(drop=True),
            original.holidays[columns].reset_index(drop=True),
            check_dtype=False,
        )
    assert list(clone.extra_regressors) == list(original.extra_regressors)
    for name, props in original.extra_regressors.items():
        for key in ("prior_scale", "standardize", "mode"):
            assert clone.extra_regressors[name][key] == props[key], (name, key)
    # Built-in seasonalities appear at fit time; let the clone resolve them on the same history.
    if original.history is not None:
        clone.history = original.history
        clone.set_auto_seasonalities()
    assert dict(clone.seasonalities) == dict(original.seasonalities)


@pytest.mark.parametrize("name", ALL_CASES)
def test_recreate_code_rebuilds_every_fixture(name: str) -> None:
    m, _, _ = fit_case(name)
    code = settings.recreate_code(m)
    assert_valid_calls(code)
    assert code.startswith("from prophet import Prophet\n")
    assert all(len(line) <= 88 for line in code.splitlines())
    assert EM_DASH not in code
    assert_same_settings(m, run_recreated(code))


def test_default_model_needs_no_arguments() -> None:
    m, _, _ = fit_case("linear")
    assert settings.recreate_code(m) == (
        "from prophet import Prophet\n\n"
        "m = Prophet()\n\n"
        "# Fit with m.fit(df), where df has the columns: ds, y.\n"
    )
    assert settings.constructor_args(m) == []


def test_only_changed_arguments_are_written() -> None:
    m, _, _ = fit_case("multiplicative")
    assert settings.constructor_args(m) == [("seasonality_mode", '"multiplicative"')]
    m, _, _ = fit_case("logistic")
    assert 'growth="logistic"' in settings.recreate_code(m)
    assert "ds, y, cap, floor." in settings.recreate_code(m)
    m, _, _ = fit_case("no_uncertainty")
    assert settings.constructor_args(m) == [("uncertainty_samples", "0")]


def test_custom_seasonalities_and_regressors_become_add_calls() -> None:
    m, _, _ = fit_case("no_regressors_custom_seasonality")
    assert settings.seasonality_calls(m) == [
        'm.add_seasonality(name="monthly", period=30.5, fourier_order=3)'
    ]
    m, _, _ = fit_case("regressors")
    assert settings.regressor_calls(m) == [
        'm.add_regressor("x")',
        'm.add_regressor("z", mode="multiplicative")',
    ]


def test_built_in_seasonalities_are_left_to_the_constructor() -> None:
    m, _, _ = fit_case("linear")
    assert set(m.seasonalities) == {"yearly", "weekly"}
    assert settings.seasonality_calls(m) == []


def test_a_rich_model_is_rebuilt_exactly() -> None:
    m, _ = rich_case()
    code = settings.recreate_code(m)
    assert_valid_calls(code)
    assert_same_settings(m, run_recreated(code))
    assert "import pandas as pd" in code
    assert "holidays = pd.DataFrame(" in code
    assert "changepoints = [" in code
    assert "changepoints=changepoints" in code and "holidays=holidays" in code
    assert 'm.add_country_holidays("US")' in code
    assert 'condition_name="on"' in code
    assert "n_changepoints" not in code
    assert all(len(line) <= 88 for line in code.splitlines())
    assert "ds, y, reg, on." in code


def test_overridden_built_in_seasonality_is_written_out() -> None:
    m = Prophet(yearly_seasonality=False)
    m.add_seasonality(name="yearly", period=365.25, fourier_order=4, prior_scale=3.0)
    assert settings.seasonality_calls(m) == [
        'm.add_seasonality(name="yearly", period=365.25, fourier_order=4, prior_scale=3.0)'
    ]
    clone = run_recreated(settings.recreate_code(m))
    assert clone.seasonalities["yearly"] == m.seasonalities["yearly"]
    assert clone.yearly_seasonality is False


def test_a_changepoint_count_cut_by_a_short_history_is_not_written_as_a_choice() -> (
    None
):
    m, df, _ = short_case()
    default = settings.init_defaults()["n_changepoints"]
    assert m.n_changepoints < default and settings.changepoints_cut_by_history(m)
    code = settings.recreate_code(m)
    assert "n_changepoints=" not in code
    assert f"# Prophet cut n_changepoints to {m.n_changepoints} to fit" in code
    assert settings.constructor_args(m) == [("uncertainty_samples", "50")]
    clone = run_recreated(code)
    assert clone.n_changepoints == default
    # On the same short history the clone makes the same cut and the same changepoints.
    clone.history = clone.setup_dataframe(df.copy(), initialize_scales=True)
    clone.set_changepoints()
    assert clone.n_changepoints == m.n_changepoints
    assert list(clone.changepoints) == list(m.changepoints)
    assert_same_settings(m, clone, ignore={"n_changepoints"})


def test_a_changepoint_count_someone_chose_is_still_written() -> None:
    m, _, _ = fit_case("linear")
    assert not settings.changepoints_cut_by_history(m)
    few = Prophet(n_changepoints=10)
    few.history = m.history
    assert not settings.changepoints_cut_by_history(few)
    assert ("n_changepoints", "10") in settings.constructor_args(few)
    by_hand = Prophet(changepoints=["2013-01-01"])
    by_hand.history = m.history
    assert not settings.changepoints_cut_by_history(by_hand)
    assert not settings.changepoints_cut_by_history(Prophet())


def test_long_holiday_frames_are_not_written_out() -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="30D")
    holidays = pd.DataFrame({"holiday": [f"h{i % 6}" for i in range(60)], "ds": dates})
    m = Prophet(holidays=holidays)
    code = settings.recreate_code(m)
    assert "pd.DataFrame" not in code and "import pandas" not in code
    assert (
        "# holidays: define the DataFrame you passed to Prophet (60 rows, 6 holidays)"
        in code
    )
    assert "holidays=holidays" in code
    assert_same_settings(m, run_recreated(code, holidays=holidays))
    with pytest.raises(NameError):
        run_recreated(code)


def test_holiday_frame_with_nan_prior_scale_round_trips() -> None:
    holidays = pd.DataFrame(
        {
            "holiday": ["a", "b"],
            "ds": [pd.Timestamp("2024-01-01 06:30:00"), pd.Timestamp("2024-02-01")],
            "lower_window": [-1, 0],
            "upper_window": [1, 0],
            "prior_scale": [3.0, np.nan],
        }
    )
    m = Prophet(holidays=holidays)
    clone = run_recreated(settings.recreate_code(m))
    assert clone.holidays is not None
    assert clone.holidays["ds"].tolist() == holidays["ds"].tolist()
    assert clone.holidays["prior_scale"].isna().tolist() == [False, True]


def test_mlx_backend_is_written_by_name() -> None:
    pytest.importorskip("mlx")
    m = Prophet(stan_backend="MLX")
    assert ("stan_backend", '"MLX"') in settings.constructor_args(m)
    assert settings.backend_label(m) == "MLX (experimental)"


def test_every_constructor_parameter_is_known() -> None:
    """The recreate code reads one attribute per ``__init__`` parameter, so none may be missing."""
    m = Prophet()
    for name in settings.init_defaults():
        assert hasattr(m, name) or name == "stan_backend", name


# ---- the section ------------------------------------------------------------


def code_in(body: str) -> str:
    match = re.search(r'<pre class="code"><code>(.*?)</code></pre>', body, re.S)
    assert match is not None
    return html.unescape(match.group(1))


@pytest.mark.parametrize("name", ALL_CASES)
def test_section_survives_every_fixture(name: str) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case(name)
    with no_warnings():
        section = settings.build(
            build_context(m, fcst, df=df, units={"temperature": "°C"})
        )
    assert section is not None
    assert (section.key, section.title) == ("settings", "Model settings")
    assert EM_DASH not in section.body
    assert code_in(section.body) == settings.recreate_code(m)
    assert section.body.count("<h3>") >= 6
    assert len(section.body) < 20_000


def test_the_training_rows_and_range_are_left_to_the_data_section() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    section = settings.build(build_context(m, fcst, df=df))
    assert section is not None
    assert "Training rows" not in section.body and "Training range" not in section.body
    assert not hasattr(settings, "data_pairs")


def test_section_shows_the_rich_model_in_words() -> None:
    pytest.importorskip("plotly")
    m, df = rich_case()
    section = settings.build(build_context(m, m.predict(m.history.copy()), df=df))
    assert section is not None
    text = re.sub(
        r"<[^>]+>", " ", re.sub(r"<pre.*?</pre>", "", section.body, flags=re.S)
    )
    text = html.unescape(re.sub(r"\s+", " ", text))
    assert "Multiplicative: effects scale with the trend" in text
    assert "0.5, more flexible than the default (0.05)" in text
    assert "95% intervals from 50 simulated futures" in text
    assert "4 chosen by hand" in text
    assert "quarterly 3 months additive none 4 3" in text
    assert "winter_weekly 1 week multiplicative on 2 5" in text
    assert "reg additive no 2" in text
    assert "Country holidays US" in text


def test_section_lists_regressor_units() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("temperature")
    section = settings.build(build_context(m, fcst, df=df, units={"temperature": "°C"}))
    assert section is not None
    assert "temperature (°C)" in section.body


def test_section_names_the_cases_without_seasonality_or_regressors() -> None:
    pytest.importorskip("plotly")
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"]).head(40)
    m.fit(df)
    section = settings.build(
        build_context(m, m.predict(m.make_future_dataframe(5)), df=df)
    )
    assert section is not None
    assert "The model has no seasonal patterns." in section.body
    assert "The model has no extra regressors." in section.body
    assert "Not estimated: uncertainty samples is 0" in section.body


def test_section_ignores_what_the_forecast_frame_holds() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    full = settings.build(build_context(m, fcst, df=df))
    future = fcst[fcst["ds"] > m.history["ds"].max()]
    only_future = settings.build(build_context(m, future))
    in_sample = settings.build(
        build_context(m, fcst[fcst["ds"] <= m.history["ds"].max()])
    )
    assert full is not None and only_future is not None and in_sample is not None
    assert full.body == only_future.body == in_sample.body


def test_section_survives_skipped_weekdays_and_short_history() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = short_case()
    skip = skip_axis(fcst["ds"], skip_days="weekends")
    kept = fcst[skip.keeps(fcst["ds"])].reset_index(drop=True)
    with no_warnings():
        weekdays = settings.build(build_context(m, kept, df=df, skip=skip))
        short = settings.build(build_context(m, fcst, df=df))
    assert weekdays is not None and short is not None
    assert code_in(short.body) == settings.recreate_code(m)
    assert "Linear growth" in short.body


# ---- fixtures ----------------------------------------------------------------


@lru_cache(maxsize=None)
def rich_case() -> tuple[Prophet, pd.DataFrame]:
    """A model that uses every setting the recreate code has to carry."""
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"])
    df = df.assign(
        on=df["ds"].dt.month.isin([1, 2, 3]), reg=np.sin(np.arange(len(df)) / 9.0)
    )
    holidays = pd.DataFrame(
        {
            "holiday": ["superbowl", "superbowl", "election"],
            "ds": pd.to_datetime(["2013-02-03", "2014-02-02", "2012-11-06"]),
            "lower_window": [-1, -1, 0],
            "upper_window": [1, 1, 2],
            "prior_scale": [5.0, 5.0, 15.0],
        }
    )
    m = Prophet(
        holidays=holidays,
        changepoints=["2012-09-03", "2013-01-15", "2013-06-03", "2013-11-04"],
        seasonality_mode="multiplicative",
        holidays_mode="additive",
        seasonality_prior_scale=5.0,
        holidays_prior_scale=8.0,
        changepoint_prior_scale=0.5,
        interval_width=0.95,
        uncertainty_samples=50,
        weekly_seasonality=5,
        yearly_seasonality=False,
    )
    m.add_country_holidays("US")
    m.add_seasonality("quarterly", 91.3125, 4, prior_scale=3.0, mode="additive")
    m.add_seasonality("winter_weekly", 7, 2, condition_name="on")
    m.add_regressor("reg", prior_scale=2.0, standardize=False, mode="additive")
    m.fit(df)
    return m, df


@lru_cache(maxsize=None)
def short_case() -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA / "data.csv", parse_dates=["ds"]).head(30)
    m = Prophet(uncertainty_samples=50).fit(df)
    return m, df, m.predict(m.make_future_dataframe(periods=14))
