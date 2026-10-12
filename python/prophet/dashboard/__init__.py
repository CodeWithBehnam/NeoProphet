"""Sections of the interactive dashboard written by :func:`prophet.visuals.dashboard_html`.

Each module builds one section from a :class:`DashboardContext` and returns a
:class:`Section`, or ``None`` when the section does not apply to the model (no
holidays, no in-sample rows, no backtest and so on). The numbers and the
wording come from small pure functions in each module; the ``build`` function
only draws. Plotly is never imported here: it arrives through the context.

Nothing in this package is imported by :mod:`prophet.visuals` at import time,
so the modules may import private helpers from it.
"""

from __future__ import annotations

from prophet.dashboard._context import DashboardContext, Section

#: Section keys in their default order, as accepted by ``dashboard_html(sections=...)``.
SECTION_ORDER: tuple[str, ...] = (
    "forecast",
    "accuracy",
    "components",
    "trend",
    "seasonality",
    "holidays",
    "regressors",
    "diagnostics",
    "data",
    "settings",
)

__all__ = ["SECTION_ORDER", "DashboardContext", "Section"]
