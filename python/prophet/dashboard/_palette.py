"""Colours for the dashboard in its light and dark themes.

The light theme is the editorial palette of :mod:`prophet.plot`. Figures are
always built with the light colours; :data:`DARK_SWAP` maps every light colour
string a dashboard figure may carry to its dark counterpart, and the page
script swaps them when the viewer changes theme. A colour used in a figure
but missing from :data:`DARK_SWAP` stays light in the dark theme, so new
colours belong here and nowhere else (``tests/test_dashboard_page.py`` fails
for any figure colour that is not in the table).

:data:`FIGURE_TEMPLATE` replaces plotly's default template on every embedded
figure, so a property a section leaves unset takes a palette colour too.

The series and dark-chrome steps were checked with the dataviz palette
validator: the three series colours pass the all-pairs colour-blind,
lightness and 3:1 contrast checks against both papers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from prophet.plot import (
    ACCENT,
    FORECAST_TINT,
    GRID,
    HISTORY_TINT,
    INK,
    INK_SOFT,
    OBSERVED,
    OBSERVED_TEXT,
    PAPER,
)

#: ACCENT at the band opacity; the exact string prophet.plot and prophet.visuals use for intervals.
BAND = "rgba(153, 15, 61, 0.25)"
#: The climatology fill of prophet.visuals (ACCENT blended 30% into PAPER).
CLIMATOLOGY_FILL = "#e0b1b8"

#: Up to three series that must be told apart. Slot 1 is the forecast accent.
#: Use them in this order; a fourth series is a sign to facet instead.
SERIES: tuple[str, str, str] = (ACCENT, "#2F6DB5", "#B07A12")
SERIES_DARK: tuple[str, str, str] = ("#E0457A", "#4F8FDB", "#B07E1A")

#: One-hue claret ramp for magnitude (heatmaps). The light end recedes into the paper.
SEQUENTIAL: tuple[tuple[float, str], ...] = (
    (0.0, "#F8E6E2"),
    (0.2, "#EFC0C9"),
    (0.4, "#DE8AA0"),
    (0.6, "#C64F75"),
    (0.8, "#990F3D"),
    (1.0, "#5E0A26"),
)
SEQUENTIAL_DARK: tuple[tuple[float, str], ...] = (
    (0.0, "#2A1C20"),
    (0.2, "#5A2236"),
    (0.4, "#8C2D52"),
    (0.6, "#C2416E"),
    (0.8, "#E0457A"),
    (1.0, "#FAC9D6"),
)

#: Blue to claret through a warm grey, for signed values (residuals, effects). Centre it on zero.
DIVERGING: tuple[tuple[float, str], ...] = (
    (0.0, "#1F5A96"),
    (0.25, "#7FA8D4"),
    (0.5, "#EFE7DD"),
    (0.75, "#DE8AA0"),
    (1.0, "#990F3D"),
)
DIVERGING_DARK: tuple[tuple[float, str], ...] = (
    (0.0, "#78A6E0"),
    (0.25, "#3F6C9E"),
    (0.5, "#3A322C"),
    (0.75, "#8C2D52"),
    (1.0, "#E0457A"),
)

#: Semi-transparent fills for a second and third series' bands.
SERIES_BAND: tuple[str, str] = ("rgba(47, 109, 181, 0.22)", "rgba(176, 122, 18, 0.22)")
SERIES_BAND_DARK: tuple[str, str] = (
    "rgba(79, 143, 219, 0.28)",
    "rgba(176, 126, 26, 0.28)",
)

#: Page chrome in each theme, used by the page CSS as custom properties.
LIGHT: Mapping[str, str] = {
    "paper": PAPER,
    "ink": INK,
    "ink-soft": INK_SOFT,
    "grid": GRID,
    "observed": OBSERVED,
    "observed-text": OBSERVED_TEXT,
    "accent": ACCENT,
    "history-tint": HISTORY_TINT,
    "forecast-tint": FORECAST_TINT,
    "band": BAND,
}
DARK: Mapping[str, str] = {
    "paper": "#1C1714",
    "ink": "#F3ECE4",
    "ink-soft": "#BDB2A8",
    "grid": "#3A322C",
    "observed": "#9D928A",
    "observed-text": "#B3A89F",
    "accent": SERIES_DARK[0],
    "history-tint": "#231D19",
    "forecast-tint": "#2E1D22",
    "band": "rgba(224, 69, 122, 0.28)",
}


def _swap_table(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """Light to dark, failing if one light colour was given two dark steps."""
    table: dict[str, str] = {}
    for light, dark in pairs:
        key = light.upper().replace(" ", "")
        if table.get(key, dark) != dark:
            raise ValueError(f"{light} maps to both {table[key]} and {dark}.")
        table[key] = dark
    return table


#: Every light colour string a dashboard figure may carry, mapped to its dark step.
#: Keys are upper case without spaces; the theme script normalises the same way
#: and also swaps colours inside strings (inline styles in titles and annotations).
DARK_SWAP: Mapping[str, str] = _swap_table(
    [
        *((LIGHT[k], DARK[k]) for k in LIGHT),
        (CLIMATOLOGY_FILL, "#6B3442"),
        *zip(SERIES, SERIES_DARK),
        *zip(SERIES_BAND, SERIES_BAND_DARK),
        *((lo, hi) for (_, lo), (_, hi) in zip(SEQUENTIAL, SEQUENTIAL_DARK)),
        *((lo, hi) for (_, lo), (_, hi) in zip(DIVERGING, DIVERGING_DARK)),
    ]
)


def colorscale(stops: Sequence[tuple[float, str]]) -> list[list[float | str]]:
    """A plotly colorscale from one of the ramps above."""
    return [[position, colour] for position, colour in stops]


def _template_axis() -> dict[str, object]:
    # Plotly's own default draws hairlines in white. PAPER keeps them as quiet as
    # that on the light page and, unlike white, follows the page into the dark theme.
    return {
        "gridcolor": PAPER,
        "linecolor": PAPER,
        "zerolinecolor": PAPER,
        "zerolinewidth": 2,
        "ticks": "",
        "automargin": True,
        "title": {"standoff": 15},
    }


#: The plotly template every embedded figure carries instead of plotly's default.
#: It keeps the default's layout behaviour (automatic margins, closest hover and so
#: on) but every colour in it comes from this module, so a property a section
#: leaves unset still follows the theme. Treat it as read-only.
FIGURE_TEMPLATE: Mapping[str, object] = {
    "layout": {
        "autotypenumbers": "strict",
        "colorway": list(SERIES),
        "font": {"color": INK_SOFT},
        "hovermode": "closest",
        "hoverlabel": {"align": "left"},
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "xaxis": _template_axis(),
        "yaxis": _template_axis(),
        "title": {"x": 0.05},
        "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
        "colorscale": {
            "sequential": colorscale(SEQUENTIAL),
            "diverging": colorscale(DIVERGING),
        },
        "shapedefaults": {"line": {"color": INK_SOFT}},
        "annotationdefaults": {"arrowcolor": INK_SOFT, "arrowhead": 0, "arrowwidth": 1},
    },
    "data": {
        "bar": [{"marker": {"line": {"color": PAPER, "width": 0.5}}}],
        # plotly.js outlines the markers of a scatter that sizes them per point
        # (a bubble chart) in white unless told otherwise, and that default is not
        # a string anywhere in the figure, so the colour swap cannot reach it.
        "scatter": [{"marker": {"line": {"color": PAPER}}}],
        "heatmap": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": colorscale(SEQUENTIAL),
            }
        ],
    },
}
