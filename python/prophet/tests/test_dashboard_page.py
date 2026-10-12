"""The page shell of the dashboard: sections, navigation, key findings, themes and colours.

The section modules are replaced by stubs wherever a test is about the shell
itself, so these tests do not move when a section changes. The colour audit is
the exception: it reads the figures of every section that is really there.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from prophet import forecast, visuals
from prophet.dashboard import SECTION_ORDER, _assets, _palette
from prophet.dashboard._context import DashboardContext, Section, build_context
from prophet.tests.test_visuals import (
    CASES,
    backtest_case,
    fit_case,
    no_warnings,
    weather_labels,
)
from prophet.visuals import dashboard_html

#: Sections written in prophet.visuals itself; the rest are modules of prophet.dashboard.
OWN = ("forecast", "components", "regressors")
MODULES = tuple(key for key in SECTION_ORDER if key not in OWN)


def stub_sections(
    monkeypatch: pytest.MonkeyPatch,
    summaries: Mapping[str, str] | None = None,
    absent: Sequence[str] = (),
) -> list[DashboardContext]:
    """Replace every section module's ``build``; returns the list that collects the contexts it saw."""
    seen: list[DashboardContext] = []
    texts = summaries or {}

    def make(key: str) -> Callable[[DashboardContext], Section | None]:
        def build(ctx: DashboardContext) -> Section | None:
            seen.append(ctx)
            if key in absent:
                return None
            return Section(
                key, key.capitalize(), f"<p>stub {key}</p>", texts.get(key, "")
            )

        return build

    for key in MODULES:
        module = importlib.import_module(f"prophet.dashboard.{key}")
        monkeypatch.setattr(module, "build", make(key))
    return seen


def page_for(tmp_path: Path, case: str = "linear", **kwargs: Any) -> str:
    m, df, fcst = fit_case(case)
    path = dashboard_html(m, fcst, tmp_path / "d.html", df=df, **kwargs)
    return path.read_text(encoding="utf-8")


def section_ids(page: str) -> list[str]:
    return re.findall(r'<section id="([^"]+)">', page)


def nav_targets(page: str) -> list[str]:
    nav = re.search(r"<nav[^>]*>(.*?)</nav>", page, re.S)
    return re.findall(r'href="#([^"]+)"', nav.group(1)) if nav else []


def without_bundle(page: str) -> str:
    from plotly.offline import get_plotlyjs

    return page.replace(get_plotlyjs(), "")


# ---- pure parts ---------------------------------------------------------------------


def test_default_sections_are_all_of_them_in_order() -> None:
    assert visuals._resolve_sections(None, SECTION_ORDER) == SECTION_ORDER


def test_requested_sections_keep_the_given_order() -> None:
    chosen = visuals._resolve_sections(["settings", "forecast"], SECTION_ORDER)
    assert chosen == ("settings", "forecast")
    assert visuals._resolve_sections([], SECTION_ORDER) == ()


def test_unknown_sections_raise_and_name_the_valid_keys() -> None:
    with pytest.raises(ValueError, match="nonsense") as caught:
        visuals._resolve_sections(["forecast", "nonsense"], SECTION_ORDER)
    assert all(key in str(caught.value) for key in SECTION_ORDER)


def test_repeated_sections_raise_and_name_the_valid_keys() -> None:
    with pytest.raises(ValueError, match="more than once") as caught:
        visuals._resolve_sections(["data", "forecast", "data"], SECTION_ORDER)
    assert "'data'" in str(caught.value)
    assert all(key in str(caught.value) for key in SECTION_ORDER)


def test_a_bare_string_is_not_a_list_of_sections() -> None:
    with pytest.raises(ValueError, match="list of keys"):
        visuals._resolve_sections("forecast", SECTION_ORDER)


@pytest.mark.parametrize("theme", ["light", "dark", "auto"])
def test_known_themes_pass(theme: str) -> None:
    visuals._check_theme(theme)


@pytest.mark.parametrize("theme", ["sepia", "", "Dark", None])
def test_other_themes_raise(theme: Any) -> None:
    with pytest.raises(ValueError, match="theme"):
        visuals._check_theme(theme)


def test_findings_are_escaped_and_linked_to_their_section() -> None:
    sections = [
        Section("trend", "The trend", "<p>x</p>", 'Grew 5% <fast> & "steadily"'),
        Section("data", "The data", "<p>y</p>", ""),
        Section("settings", "Model settings", "<p>z</p>", "  Defaults.  "),
    ]
    html_ = visuals._findings_html(sections)
    assert html_.startswith("<h2>Key findings</h2>")
    assert "Grew 5% &lt;fast&gt; &amp; &quot;steadily&quot;" in html_
    assert '<a href="#trend">The trend</a>' in html_
    assert "#data" not in html_  # no summary, no line
    assert "</a>Defaults.</li>" in html_  # trimmed


def test_findings_are_left_out_when_no_section_has_one() -> None:
    assert visuals._findings_html([Section("data", "The data", "<p>y</p>")]) == ""
    assert visuals._findings_html([]) == ""


def test_navigation_lists_the_sections_with_escaped_titles() -> None:
    nav = visuals._nav_html([Section("data", "A & B", "")])
    assert nav == '<nav aria-label="Sections"><a href="#data">A &amp; B</a></nav>'
    assert visuals._nav_html([]) == ""


def test_forecast_rows_are_put_in_date_order_with_their_weather_labels() -> None:
    ds = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    fcst = pd.DataFrame({"ds": ds, "yhat": [3.0, 1.0, 2.0]}, index=[7, 8, 9])
    labels = pd.Series(["climatology", "archive", "forecast"], index=fcst.index)
    ordered, sources = visuals._sorted_by_date(fcst, labels)
    assert list(ordered["yhat"]) == [1.0, 2.0, 3.0]
    assert list(ordered.index) == [0, 1, 2]
    assert sources is not None and list(sources) == [
        "archive",
        "forecast",
        "climatology",
    ]
    same, untouched = visuals._sorted_by_date(ordered, None)
    assert same is ordered and untouched is None


def test_regressor_summary_names_the_strongest_effect() -> None:
    m, _, fcst = fit_case("temperature")
    rows = visuals._regressor_rows(m, fcst, None)
    one = visuals._regressors_summary(rows[:1])
    many = visuals._regressors_summary(rows)
    assert one.startswith(f"Extra regressor {rows[0].effect.name}: ")
    assert many.startswith(
        f"Of {len(rows)} extra regressors, {rows[0].effect.name} matters most"
    )
    assert visuals._regressors_summary([]) == ""
    assert "—" not in one + many


def _regressor_row(
    name: str, headline: str, sentence: str, *, binary: bool = False
) -> visuals._RegressorRow:
    effect = visuals.RegressorEffect(
        name=name,
        mode="additive",
        coef=2e-5,
        coef_lower=2e-5,
        coef_upper=2e-5,
        unit=None,
        binary=binary,
        sentence=sentence,
        headline=headline,
    )
    return visuals._RegressorRow(effect, pd.Series([-21.2, 18.7]), None, None)


def test_regressor_summary_says_how_far_the_forecast_moves_not_what_one_unit_does() -> (
    None
):
    # A regressor on a large scale: the coefficient per unit is tiny, the effect on the forecast is not.
    big = _regressor_row(
        "big_spend",
        "Moves the forecast between -21.2 and +18.7.",
        "each extra unit of big spend has almost no effect on y",
    )
    assert (
        visuals._regressors_summary([big])
        == "Extra regressor big spend: it moves the forecast between -21.2 and +18.7."
    )
    assert visuals._regressors_summary([big, big]) == (
        "Of 2 extra regressors, big spend matters most: it moves the forecast between -21.2 and +18.7."
    )
    promo = _regressor_row(
        "promo",
        "When it is on, it adds 3.2 to the forecast.",
        "when promo is on, y barely changes",
        binary=True,
    )
    assert visuals._regressors_summary([promo]) == (
        "Extra regressor promo: when it is on, it adds 3.2 to the forecast."
    )


def test_a_large_scale_regressor_is_not_called_negligible_in_the_findings() -> None:
    from prophet import Prophet

    df = pd.read_csv(Path(__file__).parent / "data.csv", parse_dates=["ds"])
    rng = np.random.default_rng(0)
    big = rng.uniform(4e6, 6e6, len(df))
    df = df.assign(big=big, y=50 + 0.00002 * big + rng.normal(0, 0.5, len(df)))
    model = Prophet(uncertainty_samples=0)
    model.add_regressor("big")
    model.fit(df)
    future = model.make_future_dataframe(5)
    fcst = model.predict(future.assign(big=rng.uniform(4e6, 6e6, len(future))))
    rows = visuals._regressor_rows(model, fcst, None)
    # The per-unit reading is true and no help here: the regressor moves the forecast by tens.
    assert "almost no effect" in rows[0].effect.sentence
    headline = rows[0].effect.headline
    summary = visuals._regressors_summary(rows)
    assert "almost no effect" not in summary
    assert summary == f"Extra regressor big: it {headline[:1].lower()}{headline[1:]}"


def test_css_is_built_from_the_palette_tokens() -> None:
    css = visuals._PAGE_CSS
    for name, value in _palette.LIGHT.items():
        assert f"--{name}: {value};" in css
    dark_block = css.split(':root[data-theme="dark"]', 1)[1].split("}", 1)[0]
    for name, value in _palette.DARK.items():
        assert f"--{name}: {value};" in dark_block
    assert "color-scheme: dark" in dark_block
    for needle in (
        ".lede",
        ".note",
        ".grid2",
        ".tbl-wrap.scroll",
        ".tbl th",
        "td.num",
        "tabular-nums",
        ".kv",
        ".btn",
        ".topbar nav",
        "pre",
        "@media print",
        "position: sticky",
    ):
        assert needle in css, needle
    assert "prefers-color-scheme" not in css  # only a theme="auto" page carries it
    assert "—" not in css + visuals._AUTO_CSS
    auto = visuals._AUTO_CSS
    assert (
        ':root:not([data-theme="light"])' in auto
        and "prefers-color-scheme: dark" in auto
    )


def _css_number(pattern: str) -> int:
    found = re.search(pattern, visuals._PAGE_CSS)
    assert found is not None, pattern
    return int(found.group(1))


def test_anchor_jumps_use_one_offset_that_matches_the_sticky_bar() -> None:
    css = visuals._PAGE_CSS
    # Padding on the scroller and margin on the target add up, so only one of them may be there.
    assert css.count("scroll-padding-top") == 1 and "scroll-margin-top" not in css
    assert "scroll-padding-top: calc(var(--bar) + " in css
    # The bar is its minimum height plus the one pixel of its border.
    assert (
        _css_number(r"--bar: (\d+)px")
        == _css_number(r"\.topbar-in \{[^}]*min-height: (\d+)px") + 1
    )


def test_long_unbroken_text_wraps_instead_of_widening_the_page() -> None:
    rule = re.search(r"([^{}]+)\{\s*overflow-wrap: anywhere;\s*\}", visuals._PAGE_CSS)
    assert rule is not None
    selectors = {name.strip() for name in rule.group(1).split(",")}
    assert {"h1", ".sub", ".lede", ".note", ".foot", ".findings li"} <= selectors


def test_the_navigation_bar_stays_one_row_and_fades_the_edge_that_is_clipped() -> None:
    css = visuals._PAGE_CSS
    bar = css.split(".topbar nav {", 1)[1].split("}", 1)[0]
    assert (
        "overflow-x: auto" in bar
        and "mask-image" in bar
        and "position: relative" in bar
    )
    assert "flex-wrap" not in bar
    assert ".topbar nav.more-l" in css and ".topbar nav.more-r" in css


@pytest.mark.parametrize("theme", ["light", "dark", "auto"])
def test_a_page_prints_on_the_light_theme_whatever_it_shows_on_screen(
    theme: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    page = page_for(tmp_path, theme=theme, sections=["forecast"])  # type: ignore[arg-type]
    found = re.search(r"<style>(.*?)</style>", page, re.S)
    assert found is not None
    css = found.group(1)
    tail = css[css.rindex("@media print") :]
    # Last in the sheet, so it also beats the system dark block of an auto page.
    assert "prefers-color-scheme" not in tail
    assert css.rfind("prefers-color-scheme") < css.rindex("@media print")
    for name, value in _palette.LIGHT.items():
        assert f"--{name}: {value};" in tail
    assert (
        ':root[data-theme="dark"]' in tail
        and ':root:not([data-theme="light"])' in tail
        and "color-scheme: light" in tail
    )


def test_every_figure_colour_in_the_template_can_be_swapped() -> None:
    colours = _colour_literals(_palette.FIGURE_TEMPLATE)
    assert colours, "the template carries colours"
    assert [c for c in colours if _normal(c) not in _palette.DARK_SWAP] == []


def test_a_scatter_that_sizes_its_markers_per_point_gets_a_themed_outline() -> None:
    go = pytest.importorskip("plotly.graph_objects")
    m, df, fcst = fit_case("linear")
    figure = go.Figure(
        go.Scatter(
            x=[1, 2, 3], y=[3, 1, 2], mode="markers", marker={"size": [6, 10, 14]}
        )
    )
    [(_, layout)] = list(figures_in(build_context(m, fcst, df=df).embed(figure)))
    # plotly.js outlines such markers in white when nothing says otherwise; the colour is
    # not in the figure, so it can only follow the theme through the template.
    outline = layout["template"]["data"]["scatter"][0]["marker"]["line"]["color"]
    assert outline == _palette.PAPER
    assert _normal(outline) in _palette.DARK_SWAP


# ---- the generated page -------------------------------------------------------------


def test_sections_come_in_the_requested_order_and_the_nav_follows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    page = page_for(tmp_path, sections=["settings", "components", "forecast"])
    assert section_ids(page) == ["settings", "components", "forecast"]
    assert nav_targets(page) == ["settings", "components", "forecast"]
    for key, title in (("components", "Components"), ("forecast", "Forecast")):
        assert f'<section id="{key}"><h2>{title}</h2>' in page
    assert 'id="accuracy"' not in page and 'id="data"' not in page


def test_the_default_page_has_every_section_that_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch, absent=["holidays", "accuracy"])
    page = page_for(tmp_path)
    expected = [
        k for k in SECTION_ORDER if k not in ("holidays", "accuracy", "regressors")
    ]
    assert section_ids(page) == expected
    assert nav_targets(page) == expected


def test_a_section_that_does_not_apply_is_logged_and_left_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch, absent=["holidays"])
    with caplog.at_level(logging.INFO, logger="prophet.visuals"):
        page = page_for(tmp_path, sections=["forecast", "holidays"])
    assert section_ids(page) == ["forecast"]
    assert any(
        "'holidays'" in r.getMessage() and "does not apply" in r.getMessage()
        for r in caplog.records
    )


def test_a_section_with_the_wrong_key_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    module = importlib.import_module("prophet.dashboard.data")
    monkeypatch.setattr(module, "build", lambda ctx: Section("other", "Other", ""))
    with pytest.raises(ValueError, match="'other'"):
        page_for(tmp_path, sections=["data"])


def test_unknown_or_repeated_sections_stop_before_anything_is_written(
    tmp_path: Path,
) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    for bad in (["forecast", "nope"], ["forecast", "forecast"]):
        with pytest.raises(ValueError):
            dashboard_html(m, fcst, tmp_path / "d.html", df=df, sections=bad)
    with pytest.raises(ValueError, match="theme"):
        dashboard_html(m, fcst, tmp_path / "d.html", df=df, theme="sepia")  # type: ignore[arg-type]
    assert not (tmp_path / "d.html").exists()


def test_findings_appear_after_the_key_numbers_and_are_escaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(
        monkeypatch,
        summaries={"trend": "Up <b>5%</b> & rising", "data": "No gaps."},
    )
    page = page_for(tmp_path, sections=["forecast", "trend", "data", "settings"])
    head = page.index(">At a glance<")
    findings = page.index("<h2>Key findings</h2>")
    assert head < findings < page.index('<section id="forecast">')
    assert "Up &lt;b&gt;5%&lt;/b&gt; &amp; rising" in page
    assert "Up <b>5%</b>" not in page
    block = page[findings : page.index("</ul>", findings)]
    assert block.count("<li>") == 2  # settings has no summary, forecast neither


def test_no_findings_means_no_heading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    page = page_for(tmp_path, sections=["forecast", "data"])
    assert "Key findings" not in page


@pytest.mark.parametrize(
    ("theme", "attribute"),
    [("light", ' data-theme="light"'), ("dark", ' data-theme="dark"'), ("auto", "")],
)
def test_the_theme_argument_sets_the_starting_theme(
    theme: str, attribute: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    page = page_for(tmp_path, theme=theme, sections=["forecast"])  # type: ignore[arg-type]
    root = re.search(r"<html[^>]*>", page)
    assert root is not None
    assert root.group(0) == f'<html lang="en"{attribute} data-theme-default="{theme}">'
    css = re.search(r"<style>(.*?)</style>", page, re.S)
    assert css is not None
    assert ("prefers-color-scheme" in css.group(1)) == (theme == "auto")
    pressed = "true" if theme == "dark" else "false"
    assert f'id="theme-toggle" aria-pressed="{pressed}"' in page


def test_the_toggle_is_a_labelled_button(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    page = page_for(tmp_path, sections=["forecast"])
    button = re.search(r"<button[^>]*>(.*?)</button>", page, re.S)
    assert button is not None
    assert 'type="button"' in button.group(0) and "aria-pressed=" in button.group(0)
    assert button.group(1) == "Dark theme"
    assert ":focus-visible" in visuals._PAGE_CSS


def test_the_theme_scripts_and_the_swap_table_are_inline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    page = page_for(tmp_path, sections=["forecast"])
    assert "<script src=" not in page and "<link " not in page
    boot, bundle, tracker = (
        page.index(_assets.BOOT_SCRIPT),
        page.index("plotly.js"),
        page.index(_assets.PLOT_TRACKER),
    )
    assert boot < bundle < tracker < page.index("<body>")
    table = re.search(r"createThemeCore\((\{.*?\})\);\n", page)
    assert table is not None
    assert json.loads(table.group(1)) == dict(_palette.DARK_SWAP)
    assert "function applyTheme" in page and "Plotly.react" in page
    assert (
        "dataset.theme" in page
        and "localStorage" in page
        and "prefers-color-scheme" in page
    )
    assert page.rindex("applyTheme") > page.index(
        "</main>"
    )  # the script closes the body


def test_the_swap_table_cannot_close_the_script_element() -> None:
    script = _assets.theme_script({"#AAA": "</script><b>"})
    assert "</script>" not in script
    assert "<\\/script>" in script


def test_backtest_and_cv_reach_the_sections_and_fill_the_metric_tiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    seen = stub_sections(monkeypatch)
    m, df, fcst = fit_case("linear")
    backtest = backtest_case("linear")
    frame = pd.DataFrame({"horizon": [pd.Timedelta(days=1)], "mape": [0.1]})
    page = page_for(
        tmp_path, backtest=backtest, cv=frame, sections=["forecast", "accuracy"]
    )
    assert "How well it works on held-back data" in page and "INSIDE THE BAND" in page
    assert seen and seen[0].backtest is backtest and seen[0].cv is frame
    # Explicit metrics win over the backtest's own.
    metrics = visuals.backtest_metrics(
        [1.0, 2.0], [1.0, 2.0], [0.0] * 2, [3.0] * 2, interval_width=0.8
    )
    again = page_for(
        tmp_path, backtest=backtest, metrics=metrics, sections=["forecast"]
    )
    held_back = '<div class="l">HELD BACK</div><div class="v">{} points</div>'
    assert held_back.format(metrics.n) in again
    assert held_back.format(backtest.metrics.n) not in again
    plain = page_for(tmp_path, sections=["forecast", "accuracy"])
    assert "How well it works on held-back data" not in plain
    assert seen[-1].backtest is None and seen[-1].cv is None


def test_weather_sources_without_regressors_fail_whatever_the_sections(
    tmp_path: Path,
) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("linear")
    with pytest.raises(ValueError, match="no extra regressors"):
        dashboard_html(
            m,
            fcst,
            tmp_path / "d.html",
            sections=["forecast"],
            weather_sources=["archive"] * len(fcst),
        )


def test_the_page_has_no_em_dash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch, summaries={"data": "A finding."})
    page = without_bundle(page_for(tmp_path, theme="auto"))
    assert "—" not in page
    assert "—" not in _assets.THEME_CORE + _assets.BOOT_SCRIPT + _assets.PLOT_TRACKER


def test_unsorted_forecasts_are_drawn_in_date_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    m, df, fcst = fit_case("temperature")
    shuffled = fcst.sample(frac=1.0, random_state=3)
    sources = weather_labels(fcst, df["ds"].max())
    out = dashboard_html(
        m,
        shuffled,
        tmp_path / "d.html",
        df=df,
        weather_sources=sources.loc[shuffled.index].to_numpy().tolist(),
        sections=["forecast", "regressors"],
    )
    ordered = dashboard_html(
        m,
        fcst,
        tmp_path / "e.html",
        df=df,
        weather_sources=sources,
        sections=["forecast", "regressors"],
    )
    note = "Ahead, it comes from the weather forecast for 9 days"
    assert note in out.read_text(encoding="utf-8")
    assert note in ordered.read_text(encoding="utf-8")


# ---- robustness with stubbed sections -------------------------------------------------


SMOKE = [(case, "light") for case in CASES + ["temperature"]] + [
    ("linear", "dark"),
    ("linear", "auto"),
    ("temperature", "dark"),
]


@pytest.mark.parametrize(("case", "theme"), SMOKE)
def test_smoke_over_the_fixture_matrix(
    case: str, theme: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch, summaries={"data": "Stub finding."})
    sources = None
    if case == "temperature":
        m, df, fcst = fit_case(case)
        sources = weather_labels(fcst, df["ds"].max())
    with no_warnings():
        page = page_for(tmp_path, case, weather_sources=sources, theme=theme)  # type: ignore[arg-type]
    expected = [
        k
        for k in SECTION_ORDER
        if k != "regressors" or case in ("regressors", "temperature")
    ]
    assert section_ids(page) == expected
    assert nav_targets(page) == expected
    assert "Stub finding." in page


# ---- the components section ---------------------------------------------------------


def test_panel_labels_become_plain_text() -> None:
    label = (
        "<b>Trend: up 62% since May 2012</b><br>"
        '<span style="font-size:11px;color:#5C534C">With seasonal &amp; holiday effects removed,'
        "<br>the level moved from 41.2 to 66.7.</span>"
    )
    assert visuals._panel_text(label) == (
        "Trend: up 62% since May 2012",
        "With seasonal & holiday effects removed, the level moved from 41.2 to 66.7.",
    )
    assert visuals._panel_text("<b>Weekly</b>") == ("Weekly", "")
    assert visuals._panel_text("no markup at all") == ("no markup at all", "")


def test_panel_references_are_read_from_one_panel_alone() -> None:
    assert visuals._panel_ref("x", 1) == "x" and visuals._panel_ref("x", 2) is None
    assert visuals._panel_ref("y2 domain", 2) == "y domain"
    assert visuals._panel_ref("y2", 1) is None and visuals._panel_ref("y", 2) is None
    assert (
        visuals._panel_ref("paper", 1) is None and visuals._panel_ref(None, 1) is None
    )
    assert visuals._panel_id("xaxis", 1) == "xaxis"
    assert visuals._panel_id("xaxis", 3) == "xaxis3"


@pytest.mark.parametrize(
    "case", ["linear", "logistic", "temperature", "no_regressors_custom_seasonality"]
)
def test_each_component_becomes_a_figure_of_its_own(case: str) -> None:
    go = pytest.importorskip("plotly.graph_objects")
    from prophet.plot import plot_components_plotly

    m, _, fcst = fit_case(case)
    combined = plot_components_plotly(
        m, fcst, title=False, figsize=visuals._PANEL_FIGSIZE
    )
    before = combined.to_json()
    panels = visuals._panel_figures(go, combined)
    axes = [k for k in combined.layout.to_plotly_json() if re.fullmatch(r"xaxis\d*", k)]
    assert len(panels) == len(axes) >= 2
    assert sum(len(figure.data) for _, _, figure in panels) == len(combined.data)
    for head, sentence, figure in panels:
        assert head and "<" not in head + sentence
        layout = figure.layout.to_plotly_json()
        assert not any(re.fullmatch(r"[xy]axis\d+", key) for key in layout)
        assert "domain" not in layout["xaxis"] and "anchor" not in layout["yaxis"]
        assert {t.xaxis for t in figure.data} == {"x"}
        assert {t.yaxis for t in figure.data} == {"y"}
        # The words live in the HTML now, not in the figure, which could only clip them.
        assert not sentence or sentence[:20] not in figure.to_json()
        assert all("<br>" not in (a.text or "") for a in figure.layout.annotations)
    assert combined.to_json() == before  # only read


def test_the_trend_panel_keeps_its_zone_tints_and_labels() -> None:
    go = pytest.importorskip("plotly.graph_objects")
    from prophet.plot import plot_components_plotly

    m, _, fcst = fit_case("linear")
    combined = plot_components_plotly(
        m, fcst, title=False, figsize=visuals._PANEL_FIGSIZE
    )
    trend = visuals._panel_figures(go, combined)[0][2]
    assert len(trend.layout.shapes) == 3  # history tint, forecast tint, boundary line
    assert all(s.xref == "x" and s.yref == "y domain" for s in trend.layout.shapes)
    assert [a.text for a in trend.layout.annotations] == [
        "<b>HISTORY</b>",
        "<b>FORECAST</b>",
    ]


def test_the_components_section_puts_a_heading_and_a_sentence_over_each_panel() -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case("temperature")
    section = visuals._components_section(build_context(m, fcst, df=df))
    assert (section.key, section.title) == ("components", "Components")
    parts = re.findall(r"<h3>|Plotly\.newPlot\(", section.body)
    assert len(parts) >= 8 and parts == ["<h3>", "Plotly.newPlot("] * (len(parts) // 2)
    assert section.body.count('<p class="note">') >= 4
    assert "—" not in section.body


def test_weekday_data_keeps_the_range_breaks_on_the_panels(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    index = pd.bdate_range("2024-03-01", periods=40)
    rng = np.random.default_rng(2)
    series = pd.Series(50 + np.arange(40) * 0.1 + rng.normal(0, 0.5, 40), index=index)
    result = forecast(series, horizon=10, skip_days="weekends", uncertainty_samples=20)
    page = result.dashboard(
        tmp_path / "w.html", backtest=False, sections=["components"]
    ).read_text(encoding="utf-8")
    figures = list(figures_in(page))
    assert len(figures) >= 2
    assert {"bounds": ["sat", "mon"]} in figures[0][1]["xaxis"]["rangebreaks"]


def test_a_forecast_without_future_rows_still_has_a_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    m, df, fcst = fit_case("linear")
    page = dashboard_html(
        m, fcst[fcst["ds"] <= df["ds"].max()], tmp_path / "d.html", df=df
    ).read_text(encoding="utf-8")
    assert "in-sample fit only" in page and section_ids(page)[0] == "forecast"


def test_a_forecast_with_only_future_rows_still_has_a_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    stub_sections(monkeypatch)
    m, df, fcst = fit_case("linear")
    page = dashboard_html(
        m, fcst[fcst["ds"] > df["ds"].max()], tmp_path / "d.html", df=df
    ).read_text(encoding="utf-8")
    assert section_ids(page)[0] == "forecast"


def test_weekday_data_closes_the_gaps_and_short_histories_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    seen = stub_sections(monkeypatch)
    index = pd.bdate_range("2024-03-01", periods=40)
    rng = np.random.default_rng(2)
    series = pd.Series(50 + np.arange(40) * 0.1 + rng.normal(0, 0.5, 40), index=index)
    result = forecast(series, horizon=10, skip_days="weekends", uncertainty_samples=20)
    page = result.dashboard(tmp_path / "w.html", backtest=False).read_text(
        encoding="utf-8"
    )
    assert '"bounds":["sat","mon"]' in page.replace(" ", "")
    assert seen[-1].skip is not None and seen[-1].breaks
    assert section_ids(page) == [k for k in SECTION_ORDER if k != "regressors"]


def test_forecast_result_hands_the_backtest_to_the_dashboard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    captured: dict[str, Any] = {}

    def record(m: Any, fcst: Any, path: Any, **kwargs: Any) -> Path:
        captured.update(kwargs)
        return Path(path)

    monkeypatch.setattr(visuals, "dashboard_html", record)
    series = pd.Series(
        10 + np.sin(np.arange(60) / 4.0), index=pd.date_range("2024-01-01", periods=60)
    )
    result = forecast(series, horizon=7, uncertainty_samples=0)
    run = result.backtest()
    result.dashboard(
        tmp_path / "d.html", backtest=run, theme="dark", sections=["forecast"]
    )
    assert captured["backtest"] is run and "metrics" not in captured
    assert captured["theme"] == "dark" and captured["sections"] == ["forecast"]
    result.dashboard(tmp_path / "d.html", backtest=False)
    assert captured["backtest"] is None


# ---- colours ------------------------------------------------------------------------

TOKEN = re.compile(r"#[0-9a-fA-F]{3,8}(?![0-9a-fA-F])|rgba?\([^)]*\)")
#: Colours that are the same on both themes on purpose.
SHARED = {"TRANSPARENT", "RGBA(0,0,0,0)"}


def _normal(colour: str) -> str:
    return colour.upper().replace(" ", "")


def _colour_literals(node: object) -> list[str]:
    """Every colour written in a plotly spec: hex and rgb(a) anywhere in a string, plus any value under a colour key."""
    found: list[str] = []

    def walk(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, str(k))
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif isinstance(value, str):
            tokens = TOKEN.findall(value)
            found.extend(tokens)
            named = key.lower().endswith("color") or key in ("colorway", "colorscale")
            if named and not tokens:
                found.append(value)

    walk(node)
    return found


def figures_in(page: str) -> Iterator[tuple[list[Any], dict[str, Any]]]:
    """The data and layout of every ``Plotly.newPlot`` call in a page."""
    decoder = json.JSONDecoder()
    for call in re.finditer(r'Plotly\.newPlot\(\s*"[^"]+",\s*', page):
        data, end = decoder.raw_decode(page, call.end())
        layout, _ = decoder.raw_decode(page[end:].lstrip(" ,\n"))
        yield data, layout


@pytest.mark.parametrize("case", [*CASES, "temperature"])
def test_every_colour_in_every_figure_can_be_swapped_to_dark(
    case: str, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    m, df, fcst = fit_case(case)
    sources = weather_labels(fcst, df["ds"].max()) if case == "temperature" else None
    page = page_for(
        tmp_path,
        case,
        weather_sources=sources,
        backtest=backtest_case("linear") if case == "linear" else None,
    )
    figures = list(figures_in(page))
    assert len(figures) >= 2
    stray = {
        colour
        for data, layout in figures
        for colour in _colour_literals([data, layout])
        if _normal(colour) not in _palette.DARK_SWAP and _normal(colour) not in SHARED
    }
    assert stray == set(), (
        f"add these to prophet.dashboard._palette.DARK_SWAP or use palette colours: {sorted(stray)}"
    )


def test_holiday_panels_carry_only_swappable_colours(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    from prophet import Prophet

    df = pd.read_csv(Path(__file__).parent / "data.csv", parse_dates=["ds"])
    holidays = pd.DataFrame(
        {
            "holiday": "launch",
            "ds": pd.to_datetime(["2013-03-01", "2013-09-02", "2014-03-03"]),
            "lower_window": 0,
            "upper_window": 1,
        }
    )
    model = Prophet(holidays=holidays, uncertainty_samples=50).fit(df)
    fcst = model.predict(model.make_future_dataframe(periods=20))
    out = dashboard_html(model, fcst, tmp_path / "h.html", df=df)
    figures = list(figures_in(out.read_text(encoding="utf-8")))
    assert len(figures) >= 2
    stray = {
        colour
        for data, layout in figures
        for colour in _colour_literals([data, layout])
        if _normal(colour) not in _palette.DARK_SWAP and _normal(colour) not in SHARED
    }
    assert stray == set()


def test_the_audit_itself_sees_colours_and_strays() -> None:
    spec = {
        "line": {"color": "#990f3d"},
        "title": {"text": '<span style="color:#5C534C">t</span>'},
        "marker": {"colorscale": [[0, "#F8E6E2"], [1, "navy"]]},
        "fillcolor": "white",
        "x": ["2024-01-01"],
    }
    found = _colour_literals(spec)
    assert {"#990f3d", "#5C534C", "#F8E6E2", "white"} <= set(found)
    assert [c for c in found if _normal(c) not in _palette.DARK_SWAP] == [
        "navy",
        "white",
    ]


# ---- the theme script itself --------------------------------------------------------


def run_js(program: str) -> Any:
    runtime = shutil.which("bun") or shutil.which("node")
    if runtime is None:
        pytest.skip("needs bun or node to run the theme script")
    done = subprocess.run(
        [runtime, "-e", program], capture_output=True, text=True, timeout=60
    )
    if done.returncode:
        pytest.fail(f"{runtime} exited with {done.returncode}:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout)


def test_swapping_colours_to_dark_and_back_is_exact_and_keeps_what_the_viewer_did() -> (
    None
):
    swap = json.dumps(dict(_palette.DARK_SWAP))
    program = (
        _assets.THEME_CORE
        + f"""
const core = createThemeCore({swap});
const big = new Float64Array(5000).fill(1.5);
const dates = Array.from({{length: 300}}, (_, i) => "2024-01-" + String(i % 28 + 1).padStart(2, "0"));
const bdata = {{dtype: "f8", bdata: "AAAAAAAA8D8="}};
const privateKeep = {{note: "#FFF7EC"}};
const spec = {{
  data: [
    {{type: "scatter", x: dates, y: big, visible: true, fillcolor: "rgba(153, 15, 61, 0.25)",
      line: {{color: "#990F3D", width: 2}},
      marker: {{color: [1, 2, 3], colorscale: [[0, "#F8E6E2"], [1, "#5E0A26"]]}}}},
    {{type: "scatter", x: bdata, y: [1, 2, 3], line: {{color: "#2F6DB5"}}, customdata: [[1, "#990F3D"]]}},
  ],
  layout: {{
    paper_bgcolor: "#FFF7EC",
    title: {{text: '<b>t</b> <span style="font-size:11px;color:#5C534C">sub</span>'}},
    xaxis: {{range: ["2024-01-01", "2024-02-01"], linecolor: "#1F1A17", tickfont: {{color: "#5C534C"}}}},
    shapes: [{{fillcolor: "#F6EDE0"}}, {{line: {{color: "#5c534c"}}}}],
    template: {{layout: {{colorway: ["#990F3D", "#2F6DB5", "#B07A12"]}}}},
    _private: privateKeep,
    other: "#123456",
  }},
}};
const original = core.snap(spec, "");
spec.layout.xaxis.range = ["2024-01-05", "2024-01-20"];
spec.data[0].visible = "legendonly";
const dark = core.remap(spec, original, true, "");
const back = core.remap(dark, original, false, "");
const lightKeys = Object.keys({swap});
const skip = new Set(["x", "y", "customdata", "_private"]);
const text = JSON.stringify(dark, (k, v) => skip.has(k) ? "" : v);
const stray = (text.match(/#[0-9a-f]{{3,8}}|rgba?\\([^)]*\\)/gi) || []).filter(t => lightKeys.includes(t.toUpperCase().replace(/\\s+/g, "")));
const originalText = JSON.stringify(original, (k, v) => skip.has(k) ? "" : v);
const backText = JSON.stringify(back, (k, v) => skip.has(k) ? "" : v);
console.log(JSON.stringify({{
  noLightLeft: stray.length === 0,
  darkPaper: dark.layout.paper_bgcolor === {json.dumps(_palette.DARK["paper"])},
  inlineStyle: dark.layout.title.text.includes({json.dumps(_palette.DARK["ink-soft"])}),
  band: dark.data[0].fillcolor === {json.dumps(_palette.DARK["band"])},
  rampStop: dark.data[0].marker.colorscale[0][1] === {json.dumps(_palette.SEQUENTIAL_DARK[0][1])},
  template: dark.layout.template.layout.colorway[0] === {json.dumps(_palette.SERIES_DARK[0])},
  sharedBig: dark.data[0].y === big && dark.data[0].x === dates && dark.data[1].x === bdata,
  numbersUntouched: dark.data[0].marker.color.join() === "1,2,3",
  privateShared: dark.layout._private === privateKeep,
  unknownKept: dark.layout.other === "#123456",
  zoomKept: dark.layout.xaxis.range.join() === "2024-01-05,2024-01-20",
  hiddenKept: dark.data[0].visible === "legendonly",
  copiedNotMutated: dark.layout !== spec.layout && spec.layout.paper_bgcolor === "#FFF7EC",
  exactBack: back.layout.paper_bgcolor === "#FFF7EC" && back.layout.shapes[1].line.color === "#5c534c"
    && back.layout.title.text === original.layout.title.text && back.data[0].fillcolor === "rgba(153, 15, 61, 0.25)",
  backKeepsZoom: back.layout.xaxis.range.join() === "2024-01-05,2024-01-20" && back.data[0].visible === "legendonly",
  swapText: core.swap("a #990f3d b rgba(153,15,61,0.25) c #123456") === "a {_palette.DARK["accent"]} b {_palette.DARK["band"]} c #123456",
}}));
"""
    )
    result = run_js(program)
    assert [name for name, ok in result.items() if not ok] == []


def test_the_scripts_are_valid_javascript() -> None:
    scripts = [
        _assets.BOOT_SCRIPT,
        _assets.PLOT_TRACKER,
        _assets.theme_script(_palette.DARK_SWAP),
    ]
    program = (
        f"const scripts = {json.dumps(scripts)};"
        "scripts.forEach(s => new Function(s));"
        "console.log(JSON.stringify(scripts.length));"
    )
    assert run_js(program) == 3


def test_the_navigation_bar_scrolls_to_a_link_and_hands_the_wheel_back_at_its_ends() -> (
    None
):
    program = (
        _assets.NAV_CORE
        + """
const nav = createNavCore();
console.log(JSON.stringify({
  visibleLinkStays: nav.reveal(100, 200, 0, 500, 40) === 0,
  linkPastTheRightEdge: nav.reveal(450, 560, 0, 500, 40) === 100,
  linkBeforeTheLeftEdge: nav.reveal(10, 90, 300, 500, 40) === 0,
  linkWithRoomStays: nav.reveal(340, 400, 280, 500, 40) === 280,
  wheelScrollsSideways: nav.wheel(0, 44, 20, 0) === 20,
  wheelStopsAtTheEnd: nav.wheel(0, 44, 100, 0) === 44,
  wheelAtTheEndGoesToThePage: nav.wheel(44, 44, 100, 0) === null,
  wheelAtTheStartGoesToThePage: nav.wheel(0, 44, -100, 0) === null,
  nothingToScrollGoesToThePage: nav.wheel(0, 0, 100, 0) === null,
  lineWheelsAreScaled: nav.wheel(10, 400, 3, 1) === 58,
  pageWheelsMoveTheWholeBar: nav.wheel(10, 400, 1, 2) === 400,
}));
"""
    )
    result = run_js(program)
    assert [name for name, ok in result.items() if not ok] == []


SHELL_HARNESS = """
const script = __SCRIPT__;
const handlers = {}, queries = {}, saved = [], classes = new Set(), navHandlers = {}, toggled = {}, scrolls = [];
const relayouts = [];
const root = {
  dataset: {},
  classList: { add: (name) => classes.add(name), remove: (name) => classes.delete(name), toggle() {} },
  getAttribute: (name) => (name === "data-theme-default" ? "dark" : null),
};
const button = { setAttribute() {}, addEventListener() {} };
const plot = {
  data: [{ type: "scatter", x: [1, 2, 3], y: [1, 2, 3], line: { color: "#990F3D" } }],
  layout: { paper_bgcolor: "#FFF7EC" },
};
// Three sections, three links 280 wide in a bar that shows 500 of its 900.
let current = null, observed = null;
const nav = {
  scrollLeft: 0, clientWidth: 500, scrollWidth: 900,
  classList: { toggle: (name, on) => { toggled[name] = on; } },
  addEventListener: (type, fn) => { navHandlers[type] = fn; },
  scrollTo: (options) => { scrolls.push(options); nav.scrollLeft = Math.min(options.left, 400); },
};
const links = ["a", "b", "c"].map((id, index) => ({
  offsetLeft: 300 * index, offsetWidth: 280,
  getAttribute: (name) => (name === "href" ? "#" + id : null),
  setAttribute: (name, value) => { if (name === "aria-current") { current = id; } },
  removeAttribute: () => { current = null; },
}));
const window = {
  localStorage: { getItem: () => null, setItem: (key, value) => saved.push([key, value]) },
  matchMedia: (query) => ({ matches: false, addEventListener: (type, fn) => { queries[query] = fn; } }),
  addEventListener: (type, fn) => { handlers[type] = fn; },
  IntersectionObserver: function (fn) { observed = fn; this.observe = () => {}; },
  __prophetPlots: [],
  Plotly: {
    react: (gd, data, layout) => { gd.data = data; gd.layout = layout; return Promise.resolve(); },
    relayout: (gd, update) => {
      relayouts.push(update);
      Object.entries(update).forEach(([path, value]) => {
        const [head, key] = path.split(".");
        gd.layout[head] = gd.layout[head] || {};
        if (value === null) { delete gd.layout[head][key]; } else { gd.layout[head][key] = value; }
      });
      return Promise.resolve();
    },
  },
};
const document = {
  documentElement: root,
  getElementById: () => button,
  querySelector: (selector) => (selector === ".topbar nav" && __NAV__ ? nav : null),
  querySelectorAll: (selector) =>
    selector === ".js-plotly-plot" ? [plot] : selector.startsWith(".topbar nav a") ? links : [{ id: "a" }, { id: "b" }, { id: "c" }],
};
// A browser has these as globals; the script is handed them as parameters instead.
new Function("window", "document", "console", "IntersectionObserver", script)(
  window, document, console, window.IntersectionObserver
);
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const wheel = (deltaY, extra) =>
  Object.assign({ deltaX: 0, deltaY, deltaMode: 0, ctrlKey: false, prevented: false, preventDefault() { this.prevented = true; } }, extra);
(async () => {
  __SCENARIO__
})();
"""


def shell_scenario(scenario: str, *, nav: bool = True) -> Any:
    """Run ``scenario`` against the theme script with a hand-made page, and return what it prints as JSON."""
    program = (
        SHELL_HARNESS.replace(
            "__SCRIPT__", json.dumps(_assets.theme_script(_palette.DARK_SWAP))
        )
        .replace("__NAV__", "true" if nav else "false")
        .replace("__SCENARIO__", scenario)
    )
    return run_js(program)


def test_narrow_screens_move_legends_under_the_plot_and_wide_ones_put_them_back() -> (
    None
):
    result = shell_scenario(
        """
  plot.layout.legend = { orientation: "h", x: 1, xanchor: "right", y: 1, yanchor: "bottom" };
  await tick();
  const seen = { atStart: relayouts.length };
  queries["(max-width: 640px)"]({ matches: true }); await tick();
  seen.under = Object.assign({}, plot.layout.legend);
  queries["(max-width: 640px)"]({ matches: true }); await tick();
  seen.calls = relayouts.length;
  handlers.beforeprint(); await tick(); handlers.afterprint(); await tick();
  seen.afterRestyle = plot.layout.legend.yref;
  seen.paper = plot.layout.paper_bgcolor;
  queries["(max-width: 640px)"]({ matches: false }); await tick();
  seen.back = plot.layout.legend;
  console.log(JSON.stringify(seen));
"""
    )
    assert result["atStart"] == 0  # a wide screen leaves every figure alone
    assert result["under"] == {
        "orientation": "h",
        "x": 0,
        "xanchor": "left",
        "y": 0,
        "yanchor": "bottom",
        "yref": "container",
    }
    assert result["calls"] == 1  # no second move for the same width
    assert result["afterRestyle"] == "container"  # a theme restyle keeps the move
    assert result["paper"] == _palette.DARK["paper"]
    assert result["back"] == {
        "orientation": "h",
        "x": 1,
        "xanchor": "right",
        "y": 1,
        "yanchor": "bottom",
    }


def test_the_toolbar_is_hidden_on_narrow_screens(tmp_path: Path) -> None:
    page = without_bundle(page_for(tmp_path))
    assert "@media (max-width: 640px) { .modebar-container { display: none" in page


def test_printing_draws_the_figures_light_and_restores_the_viewers_theme() -> None:
    result = shell_scenario(
        """
  await tick();
  const seen = { start: plot.layout.paper_bgcolor };
  handlers.beforeprint(); await tick();
  seen.printing = plot.layout.paper_bgcolor;
  seen.printingLine = plot.data[0].line.color;
  handlers.afterprint(); await tick();
  seen.after = plot.layout.paper_bgcolor;
  seen.afterLine = plot.data[0].line.color;
  queries.print({ matches: true }); await tick();
  seen.byQuery = plot.layout.paper_bgcolor;
  queries.print({ matches: false }); await tick();
  seen.byQueryBack = plot.layout.paper_bgcolor;
  handlers.beforeprint(); handlers.beforeprint(); await tick();
  seen.twice = plot.layout.paper_bgcolor;
  handlers.afterprint(); await tick();
  seen.twiceBack = plot.layout.paper_bgcolor;
  console.log(JSON.stringify({ seen, theme: root.dataset.theme, saved, pending: classes.has("theme-pending") }));
"""
    )
    light, dark = _palette.LIGHT["paper"], _palette.DARK["paper"]
    assert result["seen"] == {
        "start": dark,
        "printing": light,
        "printingLine": "#990F3D",
        "after": dark,
        "afterLine": _palette.DARK_SWAP["#990F3D"],
        "byQuery": light,
        "byQueryBack": dark,
        "twice": light,
        "twiceBack": dark,
    }
    # Printing is not a choice: the theme stays dark and nothing is remembered.
    assert result["theme"] == "dark" and result["saved"] == []
    assert result["pending"] is False


def test_the_navigation_bar_follows_the_section_in_view_and_shows_what_is_clipped() -> (
    None
):
    result = shell_scenario(
        """
  await tick();
  const seen = { fadesAtStart: [toggled["more-l"], toggled["more-r"]] };
  // The last section scrolls into view: its link is marked and brought into the bar.
  observed([{ target: { id: "c" }, isIntersecting: true }]);
  seen.current = current;
  seen.scrolledTo = scrolls.map((s) => s.left);
  navHandlers.scroll();
  seen.fadesAtEnd = [toggled["more-l"], toggled["more-r"]];
  // A link that is already in view does not move the bar; one that is cut off on the left does.
  observed([{ target: { id: "c" }, isIntersecting: false }]);
  observed([{ target: { id: "c" }, isIntersecting: true }]);
  seen.scrollsAfterRepeat = scrolls.length;
  observed([{ target: { id: "b" }, isIntersecting: true }]);
  seen.scrolledAfterB = scrolls.map((s) => s.left);
  // The wheel scrolls the bar sideways until it runs out, then goes to the page.
  nav.scrollLeft = 0;
  const first = wheel(60);
  navHandlers.wheel(first);
  seen.wheelMoved = [nav.scrollLeft, first.prevented];
  nav.scrollLeft = 400;
  const last = wheel(60);
  navHandlers.wheel(last);
  seen.wheelAtEnd = [nav.scrollLeft, last.prevented];
  nav.scrollLeft = 0;
  const sideways = wheel(60, { deltaX: 5 });
  navHandlers.wheel(sideways);
  const zoom = wheel(60, { ctrlKey: true });
  navHandlers.wheel(zoom);
  seen.leftAlone = [sideways.prevented, zoom.prevented, nav.scrollLeft];
  console.log(JSON.stringify(seen));
"""
    )
    assert result == {
        "fadesAtStart": [False, True],
        "current": "c",
        "scrolledTo": [400],
        "fadesAtEnd": [True, False],
        "scrollsAfterRepeat": 1,
        "scrolledAfterB": [400, 260],
        "wheelMoved": [60, True],
        "wheelAtEnd": [400, False],
        "leftAlone": [False, False, 0],
    }


def test_a_page_without_sections_has_no_navigation_bar_and_still_themes() -> None:
    result = shell_scenario(
        """
  await tick();
  console.log(JSON.stringify({ paper: plot.layout.paper_bgcolor, scrolls: scrolls.length }));
""",
        nav=False,
    )
    assert result == {"paper": _palette.DARK["paper"], "scrolls": 0}
