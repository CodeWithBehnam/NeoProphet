"""What a dashboard section receives and what it returns."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from prophet.dashboard._palette import FIGURE_TEMPLATE

if TYPE_CHECKING:
    from prophet.forecaster import Prophet
    from prophet.plot import SkipAxis
    from prophet.visuals import Backtest

#: plotly ``config`` for every embedded figure.
PLOTLY_CONFIG: Mapping[str, object] = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


@dataclass(frozen=True)
class Section:
    """One block of the page: an ``<h2>`` title over an HTML body.

    ``key`` is one of :data:`prophet.dashboard.SECTION_ORDER`; it is the anchor
    id of the section and the name callers pass in ``sections=``. ``body`` is
    finished HTML: every piece of text in it has been escaped. ``summary`` is
    one plain-text sentence (not escaped) for the key findings at the top of
    the page; leave it empty when the section has nothing worth lifting there.
    """

    key: str
    title: str
    body: str
    summary: str = ""


@dataclass(frozen=True)
class DashboardContext:
    """Everything a section may draw from, built once by :func:`build_context`.

    Rows on skipped weekdays or hours are already gone from ``fcst``, ``df``
    and ``history``; time axes should use ``breaks`` as plotly ``rangebreaks``
    so that the gaps close.
    """

    m: Prophet
    #: Output of ``m.predict``, sorted by ``ds``.
    fcst: pd.DataFrame
    #: The model's training rows (``m.history``), sorted by ``ds``.
    history: pd.DataFrame
    #: The observed data the caller passed, or None.
    df: pd.DataFrame | None
    #: Typical spacing between observations (median gap of ``history``).
    step: pd.Timedelta
    skip: SkipAxis | None
    breaks: tuple[Mapping[str, object], ...]
    #: Whether the caller wants intervals drawn (see :attr:`show_band`).
    uncertainty: bool
    backtest: Backtest | None
    #: Output of :func:`prophet.diagnostics.cross_validation`, or None.
    cv: pd.DataFrame | None
    #: Units by regressor name, for example ``{"temperature": "°C"}``.
    units: Mapping[str, str]
    #: ``plotly.graph_objects``.
    go: Any
    #: Figure to an HTML fragment without the plotly bundle. It gives the figure
    #: the dashboard's colour template (see ``_palette.FIGURE_TEMPLATE``), so call
    #: it once the figure is finished.
    embed: Callable[[Any], str]

    @property
    def history_end(self) -> pd.Timestamp:
        return cast("pd.Timestamp", self.history["ds"].max())

    @property
    def show_band(self) -> bool:
        """Intervals are wanted, were sampled and are present in ``fcst``."""
        return bool(
            self.uncertainty
            and self.m.uncertainty_samples
            and "yhat_lower" in self.fcst
        )

    @property
    def future(self) -> pd.DataFrame:
        """Rows of ``fcst`` after the last training date."""
        return self.fcst[self.fcst["ds"] > self.history_end]


def fitted_frame(ctx: DashboardContext) -> pd.DataFrame:
    """Training rows next to their in-sample prediction.

    Columns ``ds``, ``y``, ``yhat`` and, when :attr:`DashboardContext.show_band`,
    ``yhat_lower`` and ``yhat_upper``. Rows without an observed ``y`` or without
    a prediction in ``fcst`` are dropped, so the frame is empty when ``fcst``
    holds only future dates.
    """
    columns = ["ds", "yhat"] + (["yhat_lower", "yhat_upper"] if ctx.show_band else [])
    observed = ctx.history[["ds", "y"]].dropna(subset=["y"])
    merged = observed.merge(ctx.fcst[columns], on="ds", how="inner")
    return merged.dropna(subset=["yhat"]).sort_values("ds").reset_index(drop=True)


def _embedder(config: Mapping[str, object]) -> Callable[[Any], str]:
    def embed(fig: Any) -> str:
        fig.layout.template = FIGURE_TEMPLATE
        return cast(
            "str",
            fig.to_html(full_html=False, include_plotlyjs=False, config=dict(config)),
        )

    return embed


def build_context(
    m: Prophet,
    fcst: pd.DataFrame,
    *,
    df: pd.DataFrame | None = None,
    skip: SkipAxis | None = None,
    uncertainty: bool = True,
    backtest: Backtest | None = None,
    cv: pd.DataFrame | None = None,
    units: Mapping[str, str] | None = None,
    go: Any | None = None,
) -> DashboardContext:
    """The context for the sections of one dashboard.

    ``fcst`` and ``df`` must already be free of skipped rows (as
    :func:`prophet.visuals.dashboard_html` leaves them); the training history
    is filtered here. ``go`` defaults to ``plotly.graph_objects`` and raises
    the usual ImportError when plotly is missing.
    """
    from prophet.plot import _plotly_rangebreaks
    from prophet.visuals import _require_plotly

    if m.history is None:
        raise ValueError("The model has not been fitted.")
    history = m.history
    if skip is not None:
        history = history[skip.keeps(history["ds"])]
    history = history.sort_values("ds").reset_index(drop=True)
    gaps = history["ds"].diff().dropna()
    step = pd.Timedelta(gaps.median()) if len(gaps) else pd.Timedelta(days=1)
    breaks = (
        ()
        if skip is None
        else tuple(_plotly_rangebreaks(skip.skip_days, skip.skip_hours))
    )
    return DashboardContext(
        m=m,
        fcst=fcst.sort_values("ds").reset_index(drop=True),
        history=history,
        df=df,
        step=step,
        skip=skip,
        breaks=breaks,
        uncertainty=uncertainty,
        backtest=backtest,
        cv=cv,
        units=dict(units or {}),
        go=go if go is not None else _require_plotly(),
        embed=_embedder(PLOTLY_CONFIG),
    )
