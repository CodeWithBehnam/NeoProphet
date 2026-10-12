"""Tests for ``tune(search="optuna")`` and the search-space helpers.

The pure helpers are tested without Optuna; everything that runs a study is
skipped when Optuna is not installed.
"""

import dataclasses
import importlib
import math
import sys
from typing import Any

import pandas as pd
import pytest

from prophet import tuning
from prophet.quick import forecast
from prophet.tuning import (
    FloatRange,
    IntRange,
    Trial,
    TuningError,
    _check_search_options,
    _score_params,
    _score_params_by_fold,
    _Task,
    default_search_space,
    prophet_defaults,
    rank_trials,
    scores_frame,
    space_contains,
    space_size,
    tune,
    validate_param_grid,
    validate_search_space,
)

WINDOWS: dict[str, Any] = {
    "horizon": "30 days",
    "initial": "365 days",
    "period": "180 days",
}


# --- ranges and spaces (no Optuna needed) ---------------------------------


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: FloatRange(1.0, 0.5), "low <= high"),
        (lambda: FloatRange(0.0, 1.0, log=True), "low > 0"),
        (lambda: FloatRange(0.0, math.inf), "finite"),
        (lambda: FloatRange(True, 2.0), "finite numbers"),
        (lambda: IntRange(5, 1), "low <= high"),
        (lambda: IntRange(1, 10, step=0), "step >= 1"),
        (lambda: IntRange(0, 10, log=True), "log=True"),
        (lambda: IntRange(1, 10, log=True, step=2), "log=True"),
        (lambda: IntRange(1, 2.5), "integers"),  # type: ignore[arg-type]
    ],
)
def test_ranges_reject_invalid_bounds(make: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make()


def test_default_search_space_is_valid_and_contains_prophet_defaults() -> None:
    space = validate_search_space(default_search_space())
    assert set(space) == {
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "seasonality_mode",
    }
    assert space_contains(space, prophet_defaults(list(space)))


@pytest.mark.parametrize(
    ("space", "kwargs", "message"),
    [
        ({}, None, "at least one"),
        ({"changepoint_prior_scal": FloatRange(0.1, 1.0)}, None, "Unknown"),
        ({"seasonality_mode": []}, None, "is empty"),
        ({"seasonality_mode": "additive"}, None, "list or tuple"),
        ({"holidays": [pd.DataFrame()]}, None, "cannot store"),
        (
            {"changepoint_prior_scale": FloatRange(0.1, 1.0)},
            {"changepoint_prior_scale": 0.2},
            "both",
        ),
    ],
)
def test_validate_search_space_errors(
    space: dict[str, Any], kwargs: dict[str, Any] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_search_space(space, kwargs)


@pytest.mark.parametrize(
    "choices", [("auto", True, 1, 10), (False, 0), (1, 1.0), ("a", "a")]
)
def test_validate_search_space_rejects_choices_that_compare_equal(
    choices: tuple[Any, ...],
) -> None:
    with pytest.raises(ValueError, match="compare equal"):
        validate_search_space({"yearly_seasonality": choices})


def test_grid_search_rejects_ranges_with_a_hint() -> None:
    with pytest.raises(ValueError, match="search='optuna'"):
        validate_param_grid({"changepoint_prior_scale": FloatRange(0.01, 0.5)})


def test_space_contains_is_strict_about_types_bounds_and_steps() -> None:
    space = validate_search_space(
        {
            "changepoint_prior_scale": FloatRange(0.01, 0.5),
            "n_changepoints": IntRange(5, 25, step=5),
            "yearly_seasonality": ("auto", True, 10),
        }
    )
    ok = {
        "changepoint_prior_scale": 0.5,
        "n_changepoints": 15,
        "yearly_seasonality": 10,
    }
    assert space_contains(space, ok)
    assert space_contains(space, {**ok, "yearly_seasonality": True})
    # True == 1 and 10 == 10.0 in Python, but they are different to Prophet.
    assert not space_contains(space, {**ok, "yearly_seasonality": 1})
    assert not space_contains(space, {**ok, "yearly_seasonality": 10.0})
    assert not space_contains(space, {**ok, "changepoint_prior_scale": 0.6})
    assert not space_contains(space, {**ok, "n_changepoints": 12})
    assert not space_contains(space, {**ok, "n_changepoints": True})
    assert not space_contains(space, {"changepoint_prior_scale": 0.1})


def test_space_size_counts_finite_spaces_only() -> None:
    assert space_size({"a": (1, 2, 3), "b": IntRange(0, 10, step=5)}) == 9
    assert space_size({"a": (1, 2), "b": FloatRange(0.0, 1.0)}) is None


def test_pruned_trials_rank_after_every_completed_trial() -> None:
    trials = [
        Trial({"a": 1}, {"rmse": 0.1}, 1.0, pruned=True),
        Trial({"a": 2}, {"rmse": math.nan}, 1.0),
        Trial({"a": 3}, {"rmse": 2.0}, 1.0),
    ]
    assert [t.params["a"] for t in rank_trials(trials, "rmse")] == [3, 2, 1]


def test_scores_frame_optuna_columns() -> None:
    ranked = [
        Trial({"a": 1}, {"rmse": 1.0}, 1.0, number=0),
        Trial({"a": 2}, {"rmse": 2.0}, 1.0, is_default=True),
        Trial({"a": 3}, {"rmse": 0.5}, 1.0, number=1, pruned=True),
    ]
    plain = scores_frame(ranked, "rmse")
    assert "trial" not in plain.columns and "pruned" not in plain.columns
    frame = scores_frame(ranked, "rmse", optuna_columns=True)
    assert str(frame["trial"].dtype) == "Int64"
    assert frame["trial"].isna().tolist() == [False, True, False]
    assert frame["pruned"].tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("random", None, None, False, True), "Unknown search"),
        (("grid", 10, None, False, True), "only apply to search='optuna'"),
        (("grid", None, 5.0, False, True), "only apply to search='optuna'"),
        (("grid", None, None, True, False), "only apply to search='optuna'"),
        (("optuna", 0, None, False, True), "at least 1"),
        (("optuna", 2.5, None, False, True), "integer"),
        (("optuna", 4.0, None, False, True), "integer"),
        (("optuna", "3", None, False, True), "integer"),
        (("optuna", True, None, False, True), "integer"),
        (("optuna", None, 0.0, False, True), "positive"),
        (("optuna", None, "5", False, True), "positive"),
        (("optuna", None, True, False, True), "positive"),
        (("optuna", None, None, True, True), "parallel=False"),
    ],
)
def test_check_search_options_errors(args: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _check_search_options(*args)


def test_check_search_options_defaults() -> None:
    assert _check_search_options("grid", None, None, False, True) == 0
    assert _check_search_options("optuna", None, None, False, True) == 40
    assert _check_search_options("optuna", 7, 2.0, True, False) == 7


def test_tune_rejects_optuna_arguments_for_grid_before_fitting(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="only apply"):
        tune(daily_univariate_ts, n_trials=5)


# --- fold-wise scoring ----------------------------------------------------


def _task(df: pd.DataFrame, params: dict[str, Any]) -> _Task:
    windows = tuning.derive_windows(df["ds"], **WINDOWS)
    return _Task(df, params, {"uncertainty_samples": 0}, (), windows, 0)


def test_score_by_fold_matches_score_params_and_skips_last_fold(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    task = _task(daily_univariate_ts, {"changepoint_prior_scale": 0.1})
    assert task.windows.n_folds >= 2
    steps: list[int] = []

    def record(step: int, metrics: Any) -> bool:
        steps.append(step)
        return False

    by_fold = _score_params_by_fold(task, record)
    whole = _score_params(task)
    assert not by_fold.pruned
    assert by_fold.metrics == pytest.approx(whole.metrics)
    assert steps == list(range(task.windows.n_folds - 1))


def test_score_by_fold_stops_when_asked(daily_univariate_ts: pd.DataFrame) -> None:
    task = _task(daily_univariate_ts, {"changepoint_prior_scale": 0.1})
    first_fold = dataclasses.replace(
        task,
        windows=dataclasses.replace(task.windows, cutoffs=task.windows.cutoffs[:1]),
    )
    pruned = _score_params_by_fold(task, lambda step, metrics: True)
    assert pruned.pruned
    assert pruned.metrics == pytest.approx(_score_params(first_fold).metrics)


# --- Optuna studies --------------------------------------------------------


def test_missing_optuna_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "optuna", None)
    monkeypatch.delitem(sys.modules, "prophet.tuning_optuna", raising=False)
    with pytest.raises(ImportError, match=r"neoprophet\[optuna\]"):
        importlib.import_module("prophet.tuning_optuna")


def test_to_distributions_maps_each_dimension() -> None:
    optuna = pytest.importorskip("optuna")
    from prophet.tuning_optuna import to_distributions

    dists = to_distributions(
        {
            "a": FloatRange(0.01, 1.0, log=True),
            "b": IntRange(5, 25, step=5),
            "c": ("additive", "multiplicative"),
        }
    )
    assert dists["a"] == optuna.distributions.FloatDistribution(0.01, 1.0, log=True)
    assert dists["b"] == optuna.distributions.IntDistribution(5, 25, step=5)
    assert dists["c"] == optuna.distributions.CategoricalDistribution(
        ("additive", "multiplicative")
    )


def test_tune_optuna_end_to_end_sequential(daily_univariate_ts: pd.DataFrame) -> None:
    pytest.importorskip("optuna")
    result = tune(
        daily_univariate_ts, search="optuna", n_trials=4, parallel=False, **WINDOWS
    )
    scores = result.scores
    assert len(scores) == 5  # 4 trials plus Prophet's defaults
    assert scores["is_default"].sum() == 1
    assert scores["rank"].tolist() == [1, 2, 3, 4, 5]
    assert scores["score"].is_monotonic_increasing
    assert not scores["pruned"].any()
    # The defaults lie inside the default space, so the study holds them too.
    assert scores["trial"].notna().all()
    assert result.study is not None and len(result.study.trials) == 5
    space = default_search_space()
    for _, row in scores.iterrows():
        assert space_contains(space, {k: row[k] for k in space})
    top = scores.iloc[0]
    assert all(top[k] == v for k, v in result.best_params.items())
    assert result.study.best_value == pytest.approx(top["score"])

    model = result.best_model()
    assert model.seasonality_mode == result.best_params["seasonality_mode"]
    assert model.uncertainty_samples > 0


def test_tune_optuna_is_reproducible(daily_univariate_ts: pd.DataFrame) -> None:
    pytest.importorskip("optuna")
    kwargs: dict[str, Any] = {
        "search": "optuna",
        "n_trials": 3,
        "parallel": False,
        "include_defaults": False,
        "seed": 7,
        **WINDOWS,
    }
    first = tune(daily_univariate_ts, **kwargs)
    second = tune(daily_univariate_ts, **kwargs)
    pd.testing.assert_frame_equal(
        first.scores.drop(columns="seconds"), second.scores.drop(columns="seconds")
    )


def test_tune_optuna_parallel_batches(daily_univariate_ts: pd.DataFrame) -> None:
    pytest.importorskip("optuna")
    result = tune(
        daily_univariate_ts,
        search="optuna",
        n_trials=3,
        parallel=True,
        max_workers=2,
        param_grid={"changepoint_prior_scale": FloatRange(0.001, 0.5, log=True)},
        **WINDOWS,
    )
    assert len(result.scores) == 4
    assert result.study is not None and len(result.study.trials) == 4
    assert result.scores["trial"].notna().all()


def test_tune_optuna_baseline_outside_space_is_scored_but_not_studied(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    pytest.importorskip("optuna")
    result = tune(
        daily_univariate_ts,
        search="optuna",
        n_trials=2,
        parallel=False,
        param_grid={"changepoint_prior_scale": FloatRange(0.1, 0.5)},
        **WINDOWS,
    )
    baseline = result.scores[result.scores["is_default"]]
    assert len(baseline) == 1
    assert baseline["changepoint_prior_scale"].item() == 0.05
    assert baseline["trial"].isna().all()
    assert result.study is not None and len(result.study.trials) == 2


def test_tune_optuna_timeout_still_runs_one_trial(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    pytest.importorskip("optuna")
    result = tune(
        daily_univariate_ts,
        search="optuna",
        n_trials=10,
        timeout=1e-6,
        parallel=False,
        **WINDOWS,
    )
    assert len(result.scores) == 2  # the baseline and one Optuna trial


def test_tune_optuna_warns_when_space_is_small(
    daily_univariate_ts: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("optuna")
    with caplog.at_level("WARNING", logger="prophet"):
        tune(
            daily_univariate_ts,
            search="optuna",
            n_trials=2,
            parallel=False,
            include_defaults=False,
            param_grid={"seasonality_mode": ["additive", "multiplicative"]},
            **WINDOWS,
        )
    assert "only 2 combinations" in caplog.text


def _always_prune(optuna: Any) -> Any:
    class AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study: Any, trial: Any) -> bool:
            return True

    return AlwaysPrune()


def test_tune_optuna_prune_marks_pruned_trials(
    daily_univariate_ts: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    optuna = pytest.importorskip("optuna")
    monkeypatch.setattr(
        "prophet.tuning_optuna.FoldMedianPruner",
        lambda **kwargs: _always_prune(optuna),
    )
    result = tune(
        daily_univariate_ts,
        search="optuna",
        n_trials=3,
        parallel=False,
        prune=True,
        **WINDOWS,
    )
    scores = result.scores
    assert scores["pruned"].tolist() == [False, True, True, True]
    assert scores.iloc[0]["is_default"]  # the only completed trial wins
    assert result.study is not None
    states = [t.state for t in result.study.trials]
    assert states.count(optuna.trial.TrialState.PRUNED) == 3
    baseline = next(t for t in result.study.trials if t.user_attrs.get("is_default"))
    assert baseline.intermediate_values  # the pruner can compare against it


def test_tune_optuna_all_pruned_without_baseline_fails(
    daily_univariate_ts: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    optuna = pytest.importorskip("optuna")
    monkeypatch.setattr(
        "prophet.tuning_optuna.FoldMedianPruner",
        lambda **kwargs: _always_prune(optuna),
    )
    with pytest.raises(TuningError, match="finite"):
        tune(
            daily_univariate_ts,
            search="optuna",
            n_trials=2,
            parallel=False,
            prune=True,
            include_defaults=False,
            **WINDOWS,
        )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((40, True, True, None, 12), 10),  # capped at n_trials // 4
        ((40, True, True, None, 4), 4),  # capped at the CPU count
        ((3, True, True, None, 12), 1),  # tiny searches stay serial
        ((40, True, True, 16, 12), 16),  # an explicit pool is honoured
        ((3, True, True, 16, 12), 4),  # but never exceeds the evaluations
        ((40, True, True, 0, 12), 10),  # 0 means the default, as in grid
        ((40, True, True, -1, 12), 1),  # negative means serial, as in grid
        ((40, True, False, 8, 12), 1),  # parallel=False
    ],
)
def test_pool_size(args: tuple[Any, ...], expected: int) -> None:
    pytest.importorskip("optuna")
    from prophet.tuning_optuna import pool_size

    assert pool_size(*args) == expected


def _study_with_peers(optuna: Any, peers: int) -> Any:
    from prophet.tuning_optuna import FoldMedianPruner

    study = optuna.create_study(
        direction="minimize",
        pruner=FoldMedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    for _ in range(peers):
        study.add_trial(
            optuna.trial.create_trial(value=10.0, intermediate_values={0: 1.0, 1: 10.0})
        )
    return study


@pytest.mark.parametrize(
    ("reports", "peers", "expected"),
    [
        ({0: 0.5, 1: 1000.0}, 5, True),  # good first fold does not protect it
        ({0: 0.5, 1: 5.0}, 5, False),  # better than the median at fold 1
        ({0: 1000.0}, 5, False),  # fold 0 is the warm-up
        ({0: 0.5, 1: 1000.0}, 4, False),  # too few completed trials
        ({0: 0.5, 1: math.nan}, 5, True),  # NaN can never win
    ],
)
def test_fold_median_pruner_compares_the_current_pooled_score(
    reports: dict[int, float], peers: int, expected: bool
) -> None:
    optuna = pytest.importorskip("optuna")
    study = _study_with_peers(optuna, peers)
    trial = study.ask()
    for step, value in reports.items():
        trial.report(value, step)
    assert trial.should_prune() is expected


def test_tune_optuna_real_pruner_reports_every_fold_but_the_last(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    pytest.importorskip("optuna")
    windows: dict[str, Any] = {
        "horizon": "30 days",
        "initial": "365 days",
        "period": "60 days",
    }
    n_folds = tuning.derive_windows(daily_univariate_ts["ds"], **windows).n_folds
    assert n_folds >= 3
    result = tune(
        daily_univariate_ts,
        search="optuna",
        n_trials=6,
        parallel=False,
        prune=True,
        **windows,
    )
    assert result.study is not None
    optuna = pytest.importorskip("optuna")
    for trial in result.study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            assert set(trial.intermediate_values) == set(range(n_folds - 1))
    pruned = result.scores["pruned"].tolist()
    assert pruned == sorted(pruned)  # pruned rows rank after completed ones


def test_tune_optuna_warns_when_too_few_folds_to_prune(
    daily_univariate_ts: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("optuna")
    assert tuning.derive_windows(daily_univariate_ts["ds"], **WINDOWS).n_folds == 2
    with caplog.at_level("WARNING", logger="prophet"):
        tune(
            daily_univariate_ts,
            search="optuna",
            n_trials=1,
            parallel=False,
            prune=True,
            include_defaults=False,
            **WINDOWS,
        )
    assert "prune=True has no effect" in caplog.text


def test_tune_optuna_negative_max_workers_runs_serially(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    pytest.importorskip("optuna")
    result = tune(
        daily_univariate_ts, search="optuna", n_trials=2, max_workers=-1, **WINDOWS
    )
    assert len(result.scores) == 3


def test_tune_optuna_mixed_space_samples_ints_and_choices(
    daily_univariate_ts: pd.DataFrame,
) -> None:
    pytest.importorskip("optuna")
    space = {
        "n_changepoints": IntRange(5, 25, step=5),
        "yearly_seasonality": ["auto", True, 4],
    }
    result = tune(
        daily_univariate_ts,
        search="optuna",
        n_trials=3,
        parallel=False,
        param_grid=space,
        **WINDOWS,
    )
    assert len(result.scores) == 4
    assert result.scores["trial"].notna().all()  # defaults (25, "auto") are inside
    for _, row in result.scores.iterrows():
        params = {k: row[k] for k in space}
        params["n_changepoints"] = int(params["n_changepoints"])
        assert space_contains(validate_search_space(space), params)


# --- forecast(tune="optuna") ------------------------------------------------


class _FakeTuning:
    def __init__(self, best_params: dict[str, Any]) -> None:
        self.best_params = best_params


def test_forecast_tune_optuna_passes_search_and_space(
    daily_univariate_ts: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    def fake_tune(df: pd.DataFrame, **kwargs: Any) -> _FakeTuning:
        calls.update(kwargs)
        return _FakeTuning({"changepoint_prior_scale": 0.2})

    monkeypatch.setattr(tuning, "tune", fake_tune)
    result = forecast(
        daily_univariate_ts,
        horizon=7,
        tune="optuna",
        seasonality_mode="additive",
        uncertainty_samples=0,
    )
    assert calls["search"] == "optuna"
    assert isinstance(calls["param_grid"]["changepoint_prior_scale"], FloatRange)
    assert "seasonality_mode" not in calls["param_grid"]
    assert result.model.changepoint_prior_scale == 0.2


def test_forecast_tune_true_still_uses_grid(
    daily_univariate_ts: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    def fake_tune(df: pd.DataFrame, **kwargs: Any) -> _FakeTuning:
        calls.update(kwargs)
        return _FakeTuning({})

    monkeypatch.setattr(tuning, "tune", fake_tune)
    forecast(daily_univariate_ts, horizon=7, tune=True, uncertainty_samples=0)
    assert calls["search"] == "grid"
    assert isinstance(calls["param_grid"]["changepoint_prior_scale"], tuple)


def test_forecast_rejects_unknown_tune_value(daily_univariate_ts: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="tune must be"):
        forecast(daily_univariate_ts, horizon=7, tune="bayes")  # type: ignore[arg-type]
