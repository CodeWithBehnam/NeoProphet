"""Dashboard section "Accuracy": a held-back test and cross-validation by horizon.

The held-back test (``ctx.backtest``) draws the end of the training data next to
the actuals and forecast of the holdout, then the miss at each step into it.
Cross-validation (``ctx.cv``) draws one small panel per metric against the
distance ahead of the cutoff, and a table and chart per cutoff. Numbers and
wording come from the pure functions at the top; ``build`` only assembles.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from prophet.dashboard import _style
from prophet.dashboard._context import DashboardContext, Section
from prophet.dashboard._palette import BAND
from prophet.diagnostics import performance_metrics
from prophet.plot import (
    ACCENT,
    FORECAST_TINT,
    HISTORY_TINT,
    INK,
    INK_SOFT,
    OBSERVED,
    OBSERVED_TEXT,
    SkipAxis,
)
from prophet.visuals import (
    Backtest,
    BacktestMetrics,
    Tile,
    _duration_parts,
    _fmt_day,
    _fmt_number,
    _fmt_pct,
    _lead_phrase,
    _metric_tiles,
    _plural,
    _tiles_html,
    backtest_metrics,
    describe_periods,
)

KEY = "accuracy"
TITLE = "Accuracy"

#: Most points any one trace carries, so a long holdout stays well under a megabyte.
MAX_POINTS = 1500
#: Most horizons drawn in a cross-validation panel; nearer ones are averaged together.
MAX_HORIZONS = 200
#: Most cutoffs listed in the table and chart.
MAX_CUTOFFS = 300
#: Fewer cutoffs than this are left to the table under the chart, which says the same.
MIN_CUTOFFS_DRAWN = 5
#: The training context drawn beside the holdout is this many holdouts long.
CONTEXT_FACTOR = 3
#: ... but never fewer rows than this, so a one-row holdout still has company.
MIN_CONTEXT = 20
#: Share of the horizon range read as "near" and "far" in the cross-validation sentence.
EDGE_SHARE = 0.15
#: Change in error between near and far that counts as growing or falling.
GROWTH = 1.1
#: A band within this many points of its nominal width counts as well calibrated.
CALIBRATION_TOLERANCE = 0.05
#: Past that, a gap must also exceed this many standard errors of the coverage
#: before it is called a fault, so a short holdout cannot condemn a sound band.
CALIBRATION_SIGMAS = 2.0
#: Actuals smaller than this make a percentage error meaningless; the same guard as
#: ``prophet.diagnostics.performance_metrics``.
TINY_ACTUAL = 1e-8
#: A holdout described by its row count is re-worded when its dates cover this
#: many times more calendar than the rows suggest.
SPAN_SLACK = 1.25
#: Below this many points the coverage figure is only a rough guide.
FEW_POINTS = 20
#: A smoothed line needs at least this many steps to say anything.
MIN_SMOOTHED = 8

#: Anything an array of numbers can be made from.
Floats = Sequence[float] | np.ndarray | pd.Series

_MINUTE = pd.Timedelta(minutes=1)
_HOUR = pd.Timedelta(hours=1)
_DAY = pd.Timedelta(days=1)
_FINER = {"day": ("hour", _HOUR), "hour": ("minute", _MINUTE)}

# ----------------------------------------------------------------------------
# Pure helpers: units, wording and small calculations
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class HorizonScale:
    """The unit distances ahead of a cutoff are shown in: minutes, hours or days."""

    unit: str
    size: pd.Timedelta

    def of(self, delta: pd.Timedelta) -> float:
        return float(delta / self.size)

    def column(self, deltas: pd.Series) -> np.ndarray:
        seconds = deltas.dt.total_seconds().to_numpy(dtype=float)
        return seconds / self.size.total_seconds()

    def phrase(self, delta: pd.Timedelta) -> str:
        """A duration in words, in the next finer unit when it is under one of ours."""
        value = self.of(delta)
        if value < 1 and self.unit in _FINER:
            unit, size = _FINER[self.unit]
            return amount_phrase(float(delta / size), unit)
        return amount_phrase(value, self.unit)


def horizon_scale(shortest: pd.Timedelta, longest: pd.Timedelta) -> HorizonScale:
    """Days when the nearest forecast is a day or more ahead or the farthest is days away.

    Otherwise hours or minutes, whichever keeps the numbers small. It reads the
    distances themselves, so the unit never disagrees with the axis it labels.
    """
    if shortest >= pd.Timedelta(hours=20) or longest >= pd.Timedelta(days=3):
        return HorizonScale("day", _DAY)
    if longest >= pd.Timedelta(hours=2):
        return HorizonScale("hour", _HOUR)
    return HorizonScale("minute", _MINUTE)


def amount_phrase(value: float, unit: str) -> str:
    """'3 days', '1 hour', '2.5 days'."""
    shown = f"{value:.0f}" if value >= 10 else f"{value:.2g}"
    return f"{shown} {unit}" if shown == "1" else f"{shown} {unit}s"


def _signed(value: float) -> str:
    return f"{'-' if value < 0 else '+'}{_fmt_number(abs(value))}"


def change_phrase(new: float, old: float) -> str:
    """'about 40% larger than', 'about the same as'; empty when ``old`` is not positive."""
    if not old > 0 or not math.isfinite(new):
        return ""
    ratio = new / old
    if abs(ratio - 1.0) <= 0.15:
        return "about the same as"
    if ratio >= 2.0:
        return f"about {ratio:.1f} times as large as"
    if ratio > 1.0:
        return f"about {_fmt_pct(ratio - 1.0)} larger than"
    return f"about {_fmt_pct(1.0 - ratio)} smaller than"


def calibration_tolerance(width: float, n: int) -> float:
    """How far coverage may sit from the nominal width before the gap means something.

    The larger of five points and a few standard errors of a share measured on
    ``n`` points, so that a short run cannot call a sound band too narrow by chance.
    """
    if n <= 0:
        return 1.0
    spread = math.sqrt(width * max(1.0 - width, 0.0) / n)
    return max(CALIBRATION_TOLERANCE, CALIBRATION_SIGMAS * spread)


def coverage_sentence(coverage: float | None, width: float | None, n: int) -> str:
    """How many actuals the interval caught, and whether that is as many as it promises."""
    if coverage is None or not width:
        return ""
    target = f"{width:.0%}"
    head = f"{_fmt_pct(coverage)} of actuals fell inside the {target} band"
    gap = coverage - width
    fewer = gap < 0
    if abs(gap) <= CALIBRATION_TOLERANCE + 1e-9:
        tail = f", close to the {target} it is meant to catch."
    elif abs(gap) <= calibration_tolerance(width, n):
        more_or_fewer = "fewer" if fewer else "more"
        tail = (
            f", {more_or_fewer} than the {target} it is meant to catch, "
            "though a gap this size could be chance."
        )
    elif fewer:
        tail = f", fewer than the {target} it is meant to catch, so the band is too narrow."
    else:
        tail = f", more than the {target} it is meant to catch, so the band is wider than it needs to be."
    rough = (
        f" With only {_plural(n, 'point')} this is a rough guide."
        if 0 < n < FEW_POINTS
        else ""
    )
    return head + tail + rough


def padded_range(
    values: Sequence[Any], bottom: float = 0.06, top: float = 0.14
) -> tuple[float, float] | None:
    """The span of all finite values with some room below and more above, or None."""
    arrays = [np.asarray(v, dtype=float).ravel() for v in values]
    flat = np.concatenate(arrays) if arrays else np.empty(0)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    low, high = float(flat.min()), float(flat.max())
    span = high - low if high > low else max(abs(high), 1.0) * 0.1
    return low - bottom * span, high + top * span


def thin(frame: pd.DataFrame, limit: int = MAX_POINTS) -> pd.DataFrame:
    """At most ``limit`` evenly spaced rows, always keeping the first and last."""
    if len(frame) <= limit:
        return frame
    keep = np.unique(np.linspace(0, len(frame) - 1, limit).round().astype(int))
    return frame.iloc[keep]


def date_labels(ds: pd.Series, step: pd.Timedelta) -> list[str]:
    """Dates as text at the resolution of the data, for hover labels on a numeric axis."""
    fmt = "%d %b %Y" if step >= pd.Timedelta(hours=20) else "%d %b %Y %H:%M"
    return ds.dt.strftime(fmt).tolist()


# ----------------------------------------------------------------------------
# Held-back test: pure parts
# ----------------------------------------------------------------------------


def has_interval(frame: pd.DataFrame) -> bool:
    """Both interval columns are present and complete."""
    return bool(
        {"yhat_lower", "yhat_upper"} <= set(frame.columns)
        and frame[["yhat_lower", "yhat_upper"]].notna().all().all()
    )


def holdout_errors(bt: Backtest) -> pd.DataFrame:
    """The holdout next to its forecast, one row per held-back point.

    Columns: ``ds``, ``y``, ``yhat``, ``ahead`` (time after the cutoff), ``error``
    (actual minus forecast, as the diagnostics section reads a miss), ``abs_error`` and, when the forecast has a complete
    interval, ``yhat_lower``, ``yhat_upper`` and ``inside``. Rows without an
    actual or a forecast are dropped.
    """
    columns = ["ds", "yhat"]
    if has_interval(bt.forecast):
        columns += ["yhat_lower", "yhat_upper"]
    pairs = (
        bt.holdout[["ds", "y"]]
        .merge(bt.forecast[columns], on="ds", how="inner")
        .dropna(subset=["y", "yhat"])
        .sort_values("ds")
        .reset_index(drop=True)
    )
    error = pairs["y"] - pairs["yhat"]
    out = pairs.assign(
        ahead=pairs["ds"] - bt.cutoff, error=error, abs_error=error.abs()
    )
    if "yhat_lower" in out:
        out["inside"] = (out["y"] >= out["yhat_lower"]) & (
            out["y"] <= out["yhat_upper"]
        )
    return out


def context_rows(n_holdout: int, n_train: int) -> int:
    """How much of the end of the training data to draw: a few holdouts' worth."""
    return max(
        0, min(n_train, MAX_POINTS, max(CONTEXT_FACTOR * n_holdout, MIN_CONTEXT))
    )


def thirds_means(values: Floats) -> tuple[float, float] | None:
    """Mean of the first and of the last third, or None below six values."""
    arr = np.asarray(values, dtype=float)
    third = len(arr) // 3
    if third < 2:
        return None
    return float(np.mean(arr[:third])), float(np.mean(arr[-third:]))


def growth_sentence(abs_error: Floats) -> str:
    """Whether misses were bigger late in the holdout than early in it."""
    thirds = thirds_means(abs_error)
    if thirds is None:
        return ""
    first, last = thirds
    phrase = change_phrase(last, first)
    if not phrase:
        return ""
    return f"Misses in the last third of the holdout were {phrase} those in the first third."


def bias_sentence(error: Floats) -> str:
    """Whether the forecast leaned high or low, ignoring leans that noise alone would give.

    ``error`` is actual minus forecast, so a positive average means the forecast ran low.
    """
    arr = np.asarray(error, dtype=float)
    mae = float(np.mean(np.abs(arr))) if arr.size else 0.0
    if arr.size == 0 or mae <= 0:
        return ""
    bias = float(np.mean(arr))
    if abs(bias) < max(0.25, 1.0 / math.sqrt(arr.size)) * mae:
        return "The forecast did not lean consistently high or low."
    side = "below" if bias > 0 else "above"
    return f"On average the forecast ran {_fmt_number(abs(bias))} {side} the actuals."


def visible_metrics(metrics: BacktestMetrics, intervals: bool) -> BacktestMetrics:
    """The metrics without the interval figures when the page is not showing intervals."""
    if intervals:
        return metrics
    return replace(metrics, coverage=None, interval_width=None)


def span_phrase(bt: Backtest) -> str:
    """The held-back stretch in words that agree with the dates the chart shows.

    A backtest given a row count words its span from the rows and the spacing of
    the data, so 30 daily rows from data with gaps are called "the last 30 days"
    although they can run over six weeks. This re-words only that case, from the
    dates the rows cover; wording the caller chose, and spans already measured on
    the calendar, are kept as they are.
    """
    rows = len(bt.holdout)
    if rows < 2 or len(bt.train) < 2:
        return bt.span
    step = pd.Timedelta(bt.train["ds"].diff().dropna().median())
    if bt.span != describe_periods(rows, step):
        return bt.span
    reach = pd.Timedelta(bt.holdout["ds"].max() - bt.holdout["ds"].min()) + step
    if reach <= SPAN_SLACK * rows * step:
        return bt.span
    return _lead_phrase(*_duration_parts(reach), "the last")


def backtest_lede(bt: Backtest, metrics: BacktestMetrics, step: pd.Timedelta) -> str:
    """What the held-back test found, in a sentence or two."""
    lead = (
        f"Refitted on data up to {_fmt_day(bt.cutoff, step)} and asked to forecast "
        f"{bt.span}, the model "
    )
    if metrics.mape is not None:
        first = (
            f"{lead}had a typical error of {_fmt_pct(metrics.mape)}, "
            f"an average miss of {_fmt_number(metrics.mae)}."
        )
    else:
        first = f"{lead}missed by {_fmt_number(metrics.mae)} on average."
    second = coverage_sentence(metrics.coverage, metrics.interval_width, metrics.n)
    return " ".join(s for s in (first, second) if s)


def metrics_clause(metrics: BacktestMetrics) -> str:
    """'typical error 15%, with 78% of actuals inside the 80% band': the error and, when there is one, the coverage."""
    if metrics.mape is not None:
        head = f"typical error {_fmt_pct(metrics.mape)}"
    else:
        head = f"forecasts missed by {_fmt_number(metrics.mae)} on average"
    if metrics.coverage is None:
        return head
    band = (
        f"{metrics.interval_width:.0%} band" if metrics.interval_width else "interval"
    )
    return f"{head}, with {_fmt_pct(metrics.coverage)} of actuals inside the {band}"


def backtest_summary(bt: Backtest, metrics: BacktestMetrics) -> str:
    """One plain sentence for the key findings at the top of the page."""
    return f"Held back {bt.span}: {metrics_clause(metrics)}."


def cv_follow(metrics: BacktestMetrics, n_cutoffs: int) -> str:
    """The cross-validation result in the same words as :func:`backtest_summary`, to follow it in the key findings."""
    return f"Cross-validation over {_plural(n_cutoffs, 'cutoff')}: {metrics_clause(metrics)}."


def largest_misses(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The ``n`` biggest misses, biggest first."""
    return frame.nlargest(n, "abs_error")


# ----------------------------------------------------------------------------
# Cross-validation: pure parts
# ----------------------------------------------------------------------------

_CV_REQUIRED = ("ds", "cutoff", "y", "yhat")


def prepare_cv(cv: pd.DataFrame, skip: SkipAxis | None = None) -> pd.DataFrame:
    """Cross-validation output ready for scoring: dates parsed, incomplete rows and skipped days gone.

    Raises ``ValueError`` when a column ``cross_validation`` always writes is missing.
    """
    missing = [c for c in _CV_REQUIRED if c not in cv.columns]
    if missing:
        raise ValueError(
            f"cv lacks the column(s) {missing}. Pass the output of "
            "prophet.diagnostics.cross_validation."
        )
    out = cv.assign(ds=pd.to_datetime(cv["ds"]), cutoff=pd.to_datetime(cv["cutoff"]))
    out = out.dropna(subset=list(_CV_REQUIRED))
    if skip is not None:
        out = out[skip.keeps(out["ds"])]
    return out[out["ds"] > out["cutoff"]].reset_index(drop=True)


def mape_is_meaningful(y: Floats) -> bool:
    """Percentage errors say something only when no actual is zero or tiny."""
    arr = np.asarray(y, dtype=float)
    mean = float(np.mean(np.abs(arr))) if arr.size else 0.0
    return bool(
        arr.size
        and np.all(arr > 0)
        and mean > 0
        and arr.min() >= 0.1 * mean
        and arr.min() >= TINY_ACTUAL
    )


@dataclass(frozen=True)
class CvSummary:
    """The shape of a cross-validation run."""

    cutoffs: int
    first: pd.Timestamp
    last: pd.Timestamp
    shortest: pd.Timedelta
    longest: pd.Timedelta


def cv_summary(cv: pd.DataFrame) -> CvSummary:
    ahead = cv["ds"] - cv["cutoff"]
    return CvSummary(
        cutoffs=int(cv["cutoff"].nunique()),
        first=pd.Timestamp(cv["cutoff"].min()),
        last=pd.Timestamp(cv["cutoff"].max()),
        shortest=pd.Timedelta(ahead.min()),
        longest=pd.Timedelta(ahead.max()),
    )


def horizon_table(
    cv: pd.DataFrame, *, mape: bool, coverage: bool
) -> pd.DataFrame | None:
    """Metrics at each distance ahead, averaged over the cutoffs (no rolling window).

    Columns ``horizon`` (a Timedelta), ``mse``, ``mae`` and, as asked, ``mape``
    and ``coverage``. The squared error is kept so that RMSE can be recovered
    after distances are merged. None when nothing could be computed.
    """
    names = (
        ["mse", "mae"] + (["mape"] if mape else []) + (["coverage"] if coverage else [])
    )
    perf = performance_metrics(cv, metrics=names, rolling_window=0)
    return None if perf is None else perf.reset_index(drop=True)


def bin_horizons(perf: pd.DataFrame, limit: int = MAX_HORIZONS) -> pd.DataFrame:
    """Average neighbouring distances so that at most ``limit`` remain."""
    n = len(perf)
    if n <= limit:
        return perf.reset_index(drop=True)
    groups = np.arange(n) * limit // n
    return perf.groupby(groups, sort=True).mean().reset_index(drop=True)


def with_rmse(perf: pd.DataFrame) -> pd.DataFrame:
    return perf.assign(rmse=np.sqrt(perf["mse"]))


@dataclass(frozen=True)
class EdgeReading:
    """A metric near the cutoff and far from it.

    ``near_edge`` is the farthest distance counted as near and ``far_edge`` the
    nearest counted as far. They are equal when only one distance was scored.
    """

    near: float
    near_edge: pd.Timedelta
    far: float
    far_edge: pd.Timedelta


def edge_reading(
    perf: pd.DataFrame, column: str, share: float = EDGE_SHARE
) -> EdgeReading | None:
    """The mean of ``column`` over the nearest and farthest ``share`` of the distances."""
    if perf.empty or column not in perf:
        return None
    horizon = perf["horizon"]
    low, high = horizon.min(), horizon.max()
    reach = (high - low) * share
    near = perf[horizon <= low + reach]
    far = perf[horizon >= high - reach]
    near_value, far_value = float(near[column].mean()), float(far[column].mean())
    if not (math.isfinite(near_value) and math.isfinite(far_value)):
        return None
    return EdgeReading(
        near_value,
        pd.Timedelta(near["horizon"].max()),
        far_value,
        pd.Timedelta(far["horizon"].min()),
    )


def cv_headline(
    n_cutoffs: int, reading: EdgeReading, *, percent: bool, scale: HorizonScale
) -> str:
    """'Across 5 cutoffs the typical error grows from 4.2% for forecasts up to 5 days ahead to ...'."""
    fmt = _fmt_pct if percent else _fmt_number
    noun = "typical error" if percent else "average miss"
    cuts = _plural(n_cutoffs, "cutoff")
    near, far = fmt(reading.near), fmt(reading.far)
    near_h, far_h = scale.phrase(reading.near_edge), scale.phrase(reading.far_edge)
    if reading.near_edge == reading.far_edge:
        return f"Across {cuts} the {noun} was {near} at {near_h} ahead."
    if reading.near > 0:
        ratio = reading.far / reading.near
    else:
        ratio = math.inf if reading.far > 0 else 1.0
    if ratio >= GROWTH:
        verb = "grows"
    elif ratio <= 1.0 / GROWTH:
        verb = "falls"
    else:
        return (
            f"Across {cuts} the {noun} barely changes with distance: {near} for "
            f"forecasts up to {near_h} ahead and {far} for those {far_h} or more ahead."
        )
    return (
        f"Across {cuts} the {noun} {verb} from {near} for forecasts up to {near_h} "
        f"ahead to {far} for forecasts {far_h} or more ahead."
    )


_GLOSS = {
    "mae": "MAE is the average miss",
    "rmse": "RMSE weights big misses more",
    "mape": "MAPE is the typical error as a share of the actual",
}


def metric_gloss(columns: Sequence[str]) -> str:
    """What the abbreviations in the hover labels and table headers stand for."""
    parts = [text for key, text in _GLOSS.items() if key in columns]
    if not parts:
        return ""
    joined = (
        parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    )
    return joined + "."


def cutoff_table(cv: pd.DataFrame, *, mape: bool, coverage: bool) -> pd.DataFrame:
    """One row per cutoff: ``cutoff``, ``points``, ``mae``, ``rmse`` and, as asked, ``mape`` and ``coverage``."""
    err = cv["yhat"] - cv["y"]
    work = pd.DataFrame(
        {"cutoff": cv["cutoff"], "ae": err.abs(), "se": err**2}, index=cv.index
    )
    spec: dict[str, tuple[str, str]] = {
        "points": ("ae", "size"),
        "mae": ("ae", "mean"),
        "mse": ("se", "mean"),
    }
    if mape:
        work["ape"] = err.abs() / cv["y"].abs()
        spec["mape"] = ("ape", "mean")
    if coverage:
        work["inside"] = (
            (cv["y"] >= cv["yhat_lower"]) & (cv["y"] <= cv["yhat_upper"])
        ).astype(float)
        spec["coverage"] = ("inside", "mean")
    table = work.groupby("cutoff", sort=True).agg(**spec).reset_index()
    return table.assign(rmse=np.sqrt(table["mse"])).drop(columns="mse")


def cutoff_rows(
    table: pd.DataFrame, step: pd.Timedelta
) -> tuple[list[str], list[list[str]]]:
    """Headers and text cells for the per-cutoff table."""
    headers = ["Cutoff", "Points", "MAE", "RMSE"]
    if "mape" in table:
        headers.append("MAPE")
    if "coverage" in table:
        headers.append("Coverage")
    rows = []
    for rec in table.to_dict("records"):
        row = [
            _fmt_day(pd.Timestamp(rec["cutoff"]), step),
            f"{int(rec['points']):,}",
            _fmt_number(float(rec["mae"])),
            _fmt_number(float(rec["rmse"])),
        ]
        if "mape" in table:
            row.append(_fmt_pct(float(rec["mape"])))
        if "coverage" in table:
            row.append(_fmt_pct(float(rec["coverage"])))
        rows.append(row)
    return headers, rows


def _count(n: int, noun: str) -> str:
    """'1 point', '2,400 points'."""
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def cv_tiles(
    overall: BacktestMetrics,
    summary: CvSummary,
    scale: HorizonScale,
    step: pd.Timedelta,
) -> list[Tile]:
    """Key numbers for a cross-validation run."""
    when = (
        f"{_fmt_day(summary.first, step)} to {_fmt_day(summary.last, step)}"
        if summary.cutoffs > 1
        else f"on {_fmt_day(summary.first, step)}"
    )
    lead = [
        Tile("CUTOFFS", f"{summary.cutoffs:,}", when),
        Tile(
            "HORIZON",
            scale.phrase(summary.longest),
            f"longest distance ahead of a cutoff, {_count(overall.n, 'point')} scored",
        ),
    ]
    return lead + [t for t in _metric_tiles(overall) if t.label != "HELD BACK"]


# ----------------------------------------------------------------------------
# Small multiples: pure layout
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """Where one panel sits, as fractions of the plot area."""

    x: tuple[float, float]
    y: tuple[float, float]
    bottom: bool


def panel_grid(
    n: int,
    cols: int,
    height: int,
    *,
    col_gap: float = 0.13,
    row_gap_px: int = 72,
    margin_px: int = 100,
) -> list[Cell]:
    """Domains for ``n`` panels in ``cols`` columns, with room above each for a title."""
    rows = math.ceil(n / cols)
    plot_px = max(height - margin_px, 1)
    row_gap = row_gap_px / plot_px
    cell_h = (1.0 - row_gap * (rows - 1)) / rows
    cell_w = (1.0 - col_gap * (cols - 1)) / cols
    cells = []
    for k in range(n):
        r, c = divmod(k, cols)
        left = c * (cell_w + col_gap)
        top = 1.0 - r * (cell_h + row_gap)
        cells.append(
            Cell((left, left + cell_w), (top - cell_h, top), bottom=k + cols >= n)
        )
    return cells


def panel_layout(n: int) -> tuple[int, int]:
    """Columns and figure height for ``n`` panels: never more than two across.

    Three across leaves each panel under 80 px on a phone, too narrow for its
    title and tick labels; two across stays readable there.
    """
    if n <= 1:
        return 1, 330
    if n == 2:
        return 2, 330
    return 2, 620


@dataclass(frozen=True)
class MetricPanel:
    key: str
    title: str
    abbreviation: str
    percent: bool


_PANELS = (
    MetricPanel("mae", "Average miss", "MAE", False),
    MetricPanel("rmse", "Big misses", "RMSE", False),
    MetricPanel("mape", "Typical error", "MAPE", True),
    MetricPanel("coverage", "Inside the band", "coverage", True),
)

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def _axis_title(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "font": {"family": _style.SANS, "color": INK_SOFT, "size": 12},
    }


def _backtest_figure(
    ctx: DashboardContext, bt: Backtest, frame: pd.DataFrame, intervals: bool
) -> Any:
    """Training context, then held-back actuals against the forecast and its band."""
    go = ctx.go
    date = _style.hover_date(ctx)
    train = bt.train.dropna(subset=["y"])
    train = thin(train.tail(context_rows(len(frame), len(train))))
    shown = thin(frame)
    start = min(train["ds"].min(), bt.cutoff) if len(train) else bt.cutoff
    end = max(shown["ds"].max(), bt.cutoff)
    dot = 5 if len(train) + len(shown) <= 400 else 4

    fig = go.Figure()
    fig.add_vrect(
        x0=start.to_pydatetime(),
        x1=bt.cutoff.to_pydatetime(),
        fillcolor=HISTORY_TINT,
        opacity=1,
        layer="below",
        line_width=0,
    )
    fig.add_vrect(
        x0=bt.cutoff.to_pydatetime(),
        x1=end.to_pydatetime(),
        fillcolor=FORECAST_TINT,
        opacity=1,
        layer="below",
        line_width=0,
    )
    fig.add_vline(
        x=bt.cutoff.to_pydatetime(),
        line={"color": INK_SOFT, "width": 1, "dash": "dash"},
        layer="below",
    )
    fig.add_trace(
        go.Scatter(
            name="Training data",
            x=train["ds"],
            y=train["y"],
            mode="markers",
            marker={"color": OBSERVED, "size": dot - 1, "opacity": 0.75},
            hovertemplate=f"%{{x|{date}}}<br>Training data <b>%{{y:,.4g}}</b><extra></extra>",
        )
    )
    if intervals:
        fig.add_trace(
            go.Scatter(
                x=shown["ds"],
                y=shown["yhat_lower"],
                mode="lines",
                line={"width": 0},
                legendgroup="band",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                name=f"{ctx.m.interval_width:.0%} interval",
                x=shown["ds"],
                y=shown["yhat_upper"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=BAND,
                legendgroup="band",
                hoverinfo="skip",
            )
        )
    bounds = (
        np.column_stack([shown["yhat_lower"], shown["yhat_upper"]])
        if intervals
        else np.empty((len(shown), 0))
    )
    fig.add_trace(
        go.Scatter(
            name="Forecast",
            x=shown["ds"],
            y=shown["yhat"],
            mode="lines",
            line={"color": ACCENT, "width": 2},
            customdata=bounds,
            hovertemplate=(
                f"%{{x|{date}}}<br>Forecast <b>%{{y:,.4g}}</b>"
                + (
                    "<br>Interval %{customdata[0]:,.4g} to %{customdata[1]:,.4g}"
                    if intervals
                    else ""
                )
                + "<extra></extra>"
            ),
        )
    )
    actual_hover = (
        f"%{{x|{date}}}<br>Actual <b>%{{y:,.4g}}</b><br>Forecast %{{customdata[0]:,.4g}}"
        "<br>Actual minus forecast %{customdata[1]:+,.4g}<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            name="Held-back actuals",
            x=shown["ds"],
            y=shown["y"],
            mode="markers",
            marker={"color": INK, "size": dot},
            customdata=np.column_stack([shown["yhat"], shown["error"]]),
            hovertemplate=actual_hover,
        )
    )
    if intervals:
        outside = shown[~shown["inside"]]
        if len(outside):
            fig.add_trace(
                go.Scatter(
                    name="Outside the interval",
                    x=outside["ds"],
                    y=outside["y"],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "color": ACCENT,
                        "size": dot + 6,
                        "line": {"color": ACCENT, "width": 1.5},
                    },
                    customdata=np.column_stack([outside["yhat"], outside["error"]]),
                    hovertemplate=actual_hover,
                )
            )
    for text, anchor, shift, colour in (
        ("TRAINING", "right", -8, OBSERVED_TEXT),
        ("HELD BACK", "left", 8, ACCENT),
    ):
        fig.add_annotation(
            x=bt.cutoff.to_pydatetime(),
            y=1,
            xref="x",
            yref="paper",
            text=f"<b>{text}</b>",
            showarrow=False,
            xanchor=anchor,
            yanchor="top",
            xshift=shift,
            yshift=-4,
            font={"family": _style.SANS, "size": 11, "color": colour},
        )
    drawn: list[Any] = [train["y"], shown["y"], shown["yhat"]]
    if intervals:
        drawn += [shown["yhat_lower"], shown["yhat_upper"]]
    span = padded_range(drawn)
    _, y_style = _style.axes()
    fig.update_layout(
        **_style.layout(
            420,
            hovermode="closest",
            hoverdistance=30,
            xaxis=_style.time_axis(ctx),
            yaxis={**y_style, **({"range": list(span)} if span else {})},
        )
    )
    return fig


def _error_figure(
    ctx: DashboardContext, frame: pd.DataFrame, scale: HorizonScale, mae: float
) -> Any:
    """The absolute miss at each step into the holdout, a smoothed line and the overall average."""
    go = ctx.go
    work = frame.assign(x=scale.column(frame["ahead"]))
    smoothed = len(work) >= MIN_SMOOTHED
    if smoothed:
        window = max(3, len(work) // 10)
        work["smooth"] = (
            work["abs_error"].rolling(window, center=True, min_periods=1).mean()
        )
    shown = thin(work)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Miss at each step",
            x=shown["x"],
            y=shown["abs_error"],
            mode="markers",
            marker={"color": OBSERVED, "size": 5, "opacity": 0.75},
            text=date_labels(shown["ds"], ctx.step),
            customdata=shown["error"],
            hovertemplate=(
                "%{text}<br>"
                + f"After the cutoff ({scale.unit}s) %{{x:.6~g}}"
                + "<br>Miss <b>%{y:,.4g}</b>"
                + "<br>Actual minus forecast %{customdata:+,.4g}<extra></extra>"
            ),
        )
    )
    if smoothed:
        fig.add_trace(
            go.Scatter(
                name="Rolling average",
                x=shown["x"],
                y=shown["smooth"],
                mode="lines",
                line={"color": ACCENT, "width": 2},
                hovertemplate=(
                    f"After the cutoff ({scale.unit}s) %{{x:.6~g}}"
                    "<br>Rolling average miss <b>%{y:,.4g}</b><extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            name=f"Overall average ({_fmt_number(mae)})",
            x=[float(work["x"].min()), float(work["x"].max())],
            y=[mae, mae],
            mode="lines",
            line={"color": INK_SOFT, "width": 1.5, "dash": "dash"},
            hovertemplate="Overall average miss <b>%{y:,.4g}</b><extra></extra>",
        )
    )
    x_style, y_style = _style.axes()
    fig.update_layout(
        **_style.layout(
            340,
            hovermode="closest",
            hoverdistance=30,
            margin={"l": 64, "r": 24, "t": 36, "b": 56},
            xaxis={
                **x_style,
                "type": "linear",
                "rangemode": "tozero",
                "title": _axis_title(f"{scale.unit}s after the cutoff"),
            },
            yaxis={
                **y_style,
                "rangemode": "tozero",
                "automargin": True,
                "title": _axis_title("Absolute miss"),
            },
        )
    )
    return fig


def _horizon_figure(
    ctx: DashboardContext,
    perf: pd.DataFrame,
    scale: HorizonScale,
    width: float | None,
) -> Any:
    """One small panel per metric against the distance ahead; panels share the horizon axis."""
    go = ctx.go
    panels = [p for p in _PANELS if p.key in perf.columns]
    cols, height = panel_layout(len(panels))
    cells = panel_grid(len(panels), cols, height)
    x = scale.column(perf["horizon"])
    x_style, y_style = _style.axes()
    fig = go.Figure()
    axes: dict[str, Any] = {}
    notes: list[dict[str, Any]] = []
    for k, (panel, cell) in enumerate(zip(panels, cells), start=1):
        suffix = "" if k == 1 else str(k)
        xref, yref = f"x{suffix}", f"y{suffix}"
        title = panel.title
        if panel.key == "coverage" and width:
            title = f"Inside the {width:.0%} band"
        number = "%{y:.1%}" if panel.percent else "%{y:,.4g}"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=perf[panel.key],
                xaxis=xref,
                yaxis=yref,
                mode="lines+markers" if len(x) <= 40 else "lines",
                line={"color": ACCENT, "width": 2},
                marker={"color": ACCENT, "size": 5},
                showlegend=False,
                hovertemplate=(
                    f"Horizon ({scale.unit}s) <b>%{{x:.6~g}}</b>"
                    f"<br>{panel.title} ({panel.abbreviation}) <b>{number}</b><extra></extra>"
                ),
            )
        )
        if panel.key == "coverage" and width:
            fig.add_trace(
                go.Scatter(
                    x=[float(x.min()), float(x.max())],
                    y=[width, width],
                    xaxis=xref,
                    yaxis=yref,
                    mode="lines",
                    line={"color": INK_SOFT, "width": 1.5, "dash": "dash"},
                    showlegend=False,
                    hovertemplate="Target <b>%{y:.0%}</b><extra></extra>",
                )
            )
            notes.append(
                {
                    "x": float(x.max()),
                    "y": width,
                    "xref": xref,
                    "yref": yref,
                    "text": f"{width:.0%} target",
                    "showarrow": False,
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "font": {"family": _style.SANS, "size": 11, "color": INK_SOFT},
                }
            )
        notes.append(
            {
                "x": cell.x[0],
                "y": cell.y[1],
                "xref": "paper",
                "yref": "paper",
                "text": f"<b>{title}</b>",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "bottom",
                "yshift": 8,
                "font": {"family": _style.SANS, "size": 13, "color": INK},
            }
        )
        axes[f"xaxis{suffix}"] = {
            **x_style,
            "type": "linear",
            "rangemode": "tozero",
            "domain": list(cell.x),
            "anchor": yref,
            **({"matches": "x"} if k > 1 else {}),
            **({"title": _axis_title(f"{scale.unit}s ahead")} if cell.bottom else {}),
        }
        axes[f"yaxis{suffix}"] = {
            **y_style,
            "domain": list(cell.y),
            "anchor": xref,
            "rangemode": "tozero",
            **({"tickformat": ".1~%"} if panel.percent else {}),
            **({"range": [0, 1.05]} if panel.key == "coverage" else {}),
        }
    fig.update_layout(
        **_style.layout(
            height,
            hovermode="closest",
            hoverdistance=30,
            showlegend=False,
            margin={"l": 64, "r": 24, "t": 36, "b": 60},
            annotations=notes,
            **axes,
        )
    )
    return fig


def _cutoff_figure(
    ctx: DashboardContext, table: pd.DataFrame, overall_mae: float
) -> Any:
    """The average miss of the forecasts made at each cutoff."""
    go = ctx.go
    date = _style.hover_date(ctx)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Average miss at each cutoff",
            x=table["cutoff"],
            y=table["mae"],
            mode="lines+markers",
            line={"color": ACCENT, "width": 2},
            marker={"color": ACCENT, "size": 6},
            customdata=table["points"],
            hovertemplate=(
                f"Cutoff %{{x|{date}}}<br>Average miss <b>%{{y:,.4g}}</b>"
                "<br>%{customdata:,.0f} forecast points<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            name=f"Overall average ({_fmt_number(overall_mae)})",
            x=[table["cutoff"].min(), table["cutoff"].max()],
            y=[overall_mae, overall_mae],
            mode="lines",
            line={"color": INK_SOFT, "width": 1.5, "dash": "dash"},
            hovertemplate="Overall average miss <b>%{y:,.4g}</b><extra></extra>",
        )
    )
    x_style, y_style = _style.axes()
    # A cutoff is where the training data ended, not an observation, so it can fall
    # on a skipped weekday or hour. A plain date axis (no rangebreaks) keeps every one.
    fig.update_layout(
        **_style.layout(
            300,
            hovermode="closest",
            hoverdistance=30,
            xaxis={**x_style, "type": "date"},
            yaxis={**y_style, "rangemode": "tozero"},
        )
    )
    return fig


# ----------------------------------------------------------------------------
# Imperative shell
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Part:
    body: str
    summary: str
    #: What this part adds to the key findings when another part supplies the lead sentence.
    follow: str = ""


def _backtest_part(ctx: DashboardContext, backtest: Backtest) -> _Part | None:
    bt = replace(backtest, span=span_phrase(backtest))
    frame = holdout_errors(bt)
    if frame.empty:
        return None
    intervals = ctx.uncertainty and "inside" in frame
    metrics = visible_metrics(bt.metrics, intervals)
    scale = horizon_scale(
        pd.Timedelta(frame["ahead"].min()), pd.Timedelta(frame["ahead"].max())
    )
    how = (
        "Grey dots are the training data. Dark dots are what actually happened in "
        "the held-back stretch and the claret line is the forecast made without "
        "seeing them"
    )
    if intervals:
        how += (
            f", with its {ctx.m.interval_width:.0%} interval shaded. Circled dots "
            "fall outside it."
        )
    else:
        how += "."
    misses = largest_misses(frame)
    table_headers = ["Date", "Actual", "Forecast", "Actual minus forecast"] + (
        ["Inside the band"] if intervals else []
    )
    table_rows = [
        [
            _fmt_day(pd.Timestamp(rec["ds"]), ctx.step),
            _fmt_number(float(rec["y"])),
            _fmt_number(float(rec["yhat"])),
            _signed(float(rec["error"])),
        ]
        + ([("Yes" if rec["inside"] else "No")] if intervals else [])
        for rec in misses.to_dict("records")
    ]
    findings = " ".join(
        s
        for s in (growth_sentence(frame["abs_error"]), bias_sentence(frame["error"]))
        if s
    )
    body = "".join(
        [
            _style.h3("Held-back test"),
            _style.lede(backtest_lede(bt, metrics, ctx.step)),
            ctx.embed(_backtest_figure(ctx, bt, frame, intervals)),
            _style.note(how),
            _style.h3("Misses by distance from the cutoff"),
            ctx.embed(_error_figure(ctx, frame, scale, metrics.mae)),
            _style.note(
                (findings + " " if findings else "")
                + "Each dot is the miss at one held-back point, positioned by how long after the cutoff it falls."
            ),
            _style.h3("Largest misses"),
            _style.table(
                table_headers,
                table_rows,
                numeric=tuple(range(1, len(table_headers))),
                table_id="accuracy-misses",
            ),
            _style.note(
                "A positive difference means the actual was higher than the forecast (the forecast ran low); "
                "a negative one, lower (the forecast ran high). The diagnostics section reads its misses the same way."
            ),
        ]
    )
    return _Part(body, backtest_summary(bt, metrics))


def _cv_part(ctx: DashboardContext, cv: pd.DataFrame) -> _Part | None:
    intervals = ctx.uncertainty and has_interval(cv)
    overall = backtest_metrics(
        cv["y"],
        cv["yhat"],
        cv["yhat_lower"] if intervals else None,
        cv["yhat_upper"] if intervals else None,
        interval_width=ctx.m.interval_width,
    )
    perf = horizon_table(
        cv, mape=mape_is_meaningful(cv["y"].to_numpy()), coverage=intervals
    )
    if perf is None or perf.empty:
        return None
    percent = "mape" in perf.columns
    if not percent:
        overall = replace(overall, mape=None)
    summary = cv_summary(cv)
    scale = horizon_scale(summary.shortest, summary.longest)
    reading = edge_reading(perf, "mape" if percent else "mae")
    headline = (
        cv_headline(summary.cutoffs, reading, percent=percent, scale=scale)
        if reading is not None
        else ""
    )
    lede = " ".join(
        s
        for s in (
            headline,
            coverage_sentence(overall.coverage, ctx.m.interval_width, overall.n),
        )
        if s
    )
    binned = len(perf) > MAX_HORIZONS
    drawn = with_rmse(bin_horizons(perf))
    how = (
        f"{metric_gloss(list(drawn.columns))} Each point averages the misses of every "
        "cutoff at that distance ahead"
        + (", merging neighbouring distances to keep the page light" if binned else "")
        + ". Panels share the horizontal axis, so zooming one zooms them all."
    )
    if intervals:
        how += (
            f" The dashed line marks the {ctx.m.interval_width:.0%} a well-calibrated "
            "interval should catch."
        )
    by_cutoff = cutoff_table(cv, mape=percent, coverage=intervals)
    shown = by_cutoff.tail(MAX_CUTOFFS)
    headers, rows = cutoff_rows(shown, ctx.step)
    numeric = tuple(range(1, len(headers)))
    parts = [
        _style.h3("Cross-validation"),
        *([_style.lede(lede)] if lede else []),
        _tiles_html(cv_tiles(overall, summary, scale, ctx.step)),
        ctx.embed(
            _horizon_figure(
                ctx, drawn, scale, ctx.m.interval_width if intervals else None
            )
        ),
        _style.note(how),
        _style.h3("By cutoff"),
    ]
    if len(shown) >= MIN_CUTOFFS_DRAWN:
        parts.append(ctx.embed(_cutoff_figure(ctx, shown, overall.mae)))
    parts.append(
        _style.table(
            headers,
            rows,
            numeric=numeric,
            scroll=len(rows) > 8,
            table_id="accuracy-cutoffs",
        )
    )
    if len(by_cutoff) > len(shown):
        parts.append(
            _style.note(f"Showing the latest {len(shown)} of {len(by_cutoff)} cutoffs.")
        )
    return _Part("".join(parts), headline, cv_follow(overall, summary.cutoffs))


def _key_finding(parts: Sequence[_Part]) -> str:
    """The first part's own summary, followed by what each later part adds: the two tests can disagree."""
    lead = next((p for p in parts if p.summary), None)
    if lead is None:
        return ""
    later = [p.follow for p in parts if p is not lead and p.follow]
    return " ".join([lead.summary, *later])


def build(ctx: DashboardContext) -> Section | None:
    """The section, or None when it does not apply to this model."""
    cv = (
        prepare_cv(ctx.cv, ctx.skip)
        if ctx.cv is not None and not ctx.cv.empty
        else None
    )
    parts = [
        part
        for part in (
            _backtest_part(ctx, ctx.backtest) if ctx.backtest is not None else None,
            _cv_part(ctx, cv) if cv is not None and not cv.empty else None,
        )
        if part is not None
    ]
    if not parts:
        return None
    return Section(KEY, TITLE, "".join(p.body for p in parts), _key_finding(parts))
