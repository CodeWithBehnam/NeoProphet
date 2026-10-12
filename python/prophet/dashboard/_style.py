"""The shared look of dashboard sections: plotly layout pieces and small HTML builders.

Sections use these instead of writing their own styles, so every figure and
every table matches the page and follows its theme. All HTML builders escape
the text they are given.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

from prophet.dashboard._context import DashboardContext
from prophet.visuals import (
    _CSS_SANS,
    _CSS_SERIF,
    _hover_date,
    _plotly_axes,
    _plotly_base_layout,
    _plotly_zones,
    _zone_args,
)

SANS = _CSS_SANS
SERIF = _CSS_SERIF


def hover_date(ctx: DashboardContext) -> str:
    """d3 date format for hover labels at the data's frequency."""
    return _hover_date(ctx.step)


def axes() -> tuple[dict[str, Any], dict[str, Any]]:
    """Styles for a plotly x axis (rule, outside ticks) and y axis (gridlines)."""
    return _plotly_axes()


def time_axis(ctx: DashboardContext) -> dict[str, Any]:
    """A date x axis that closes the gaps of skipped weekdays and hours."""
    x_style, _ = _plotly_axes()
    return {
        **x_style,
        "type": "date",
        **({"rangebreaks": list(ctx.breaks)} if ctx.breaks else {}),
    }


def layout(height: int, **overrides: Any) -> dict[str, Any]:
    """The base figure layout (paper, fonts, hover label, legend top right)."""
    return {**_plotly_base_layout(height), **overrides}


def zones(ctx: DashboardContext, fig: Any, row: int | None = None) -> None:
    """Tint history and forecast behind a time-axis figure, with a rule between."""
    _plotly_zones(ctx.go, fig, _zone_args(ctx.m, ctx.fcst), row=row)


def lede(text: str) -> str:
    """The plain-English sentence or two that opens a section."""
    return f'<p class="lede">{html.escape(text)}</p>'


def h3(text: str) -> str:
    return f"<h3>{html.escape(text)}</h3>"


def note(text: str) -> str:
    """A small explanatory line under a figure or table."""
    return f'<p class="note">{html.escape(text)}</p>'


def grid2(*parts: str) -> str:
    """Two columns on wide screens, stacked on narrow ones."""
    cells = "".join(f"<div>{part}</div>" for part in parts)
    return f'<div class="grid2">{cells}</div>'


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric: Sequence[int] = (),
    scroll: bool = False,
    table_id: str | None = None,
) -> str:
    """A data table with right-aligned, tabular figures in the ``numeric`` columns.

    ``scroll`` caps its height so that long tables scroll inside the page.
    """
    right = set(numeric)

    def cell(tag: str, i: int, text: str) -> str:
        cls = ' class="num"' if i in right else ""
        return f"<{tag}{cls}>{html.escape(text)}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>"
        for row in rows
    )
    ident = f' id="{html.escape(table_id)}"' if table_id else ""
    wrap = "tbl-wrap scroll" if scroll else "tbl-wrap"
    return (
        f'<div class="{wrap}"><table class="tbl"{ident}><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def kv(pairs: Sequence[tuple[str, str]]) -> str:
    """Label and value pairs, for settings and facts."""
    items = "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>"
        for k, v in pairs
    )
    return f'<dl class="kv">{items}</dl>'
