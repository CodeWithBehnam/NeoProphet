"""The inline scripts of the dashboard page, as strings.

Nothing is fetched: the page carries the plotly bundle and these three scripts.

* :data:`BOOT_SCRIPT` runs first in ``<head>``. It settles the theme (the
  viewer's remembered choice, else the ``theme`` argument, else the system
  setting) before the first paint, so the page never flashes the wrong colours.
* :data:`PLOT_TRACKER` runs right after the plotly bundle and remembers the
  promise of every ``Plotly.newPlot`` call that plotly's own embed scripts
  make, so the theme script knows when the figures have been drawn.
* :func:`theme_script` runs at the end of ``<body>``: the toggle button, the
  remembered choice, live system changes, the figure restyling (also for
  printing, which is always on the light theme) and the navigation bar.

Figures are always written with the light colours of :mod:`prophet.dashboard._palette`.
In the dark theme the script maps every light colour string in a drawn figure
to its dark step and calls ``Plotly.react``; going back puts the original
strings in again, so the round trip is exact and anything the viewer did in
between (zoom, hidden series) stays.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

#: localStorage key that remembers the viewer's choice.
STORAGE_KEY = "prophet-dashboard-theme"

BOOT_SCRIPT = """
(function () {
  var root = document.documentElement;
  var mode = root.getAttribute("data-theme-default") || "light";
  try {
    var saved = window.localStorage.getItem("__KEY__");
    if (saved === "light" || saved === "dark") { mode = saved; }
  } catch (error) {
    // Storage can be blocked (private windows, strict settings); the page default then applies.
  }
  if (mode === "auto") {
    mode = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  root.dataset.theme = mode;
  if (mode === "dark") {
    // Figures are drawn with light colours and restyled once drawn; keep them hidden until then.
    root.classList.add("theme-pending");
    window.setTimeout(function () { root.classList.remove("theme-pending"); }, 8000);
  }
})();
""".replace("__KEY__", STORAGE_KEY).strip()

PLOT_TRACKER = """
(function () {
  var plotly = window.Plotly;
  window.__prophetPlots = [];
  if (!plotly || typeof plotly.newPlot !== "function") { return; }
  var newPlot = plotly.newPlot;
  plotly.newPlot = function () {
    var drawing = newPlot.apply(plotly, arguments);
    window.__prophetPlots.push(drawing);
    return drawing;
  };
})();
""".strip()

#: The pure part of the theme script: colour swapping on plain figure specs.
#: ``createThemeCore(swap)`` returns ``{swap, snap, remap}``; it touches no DOM.
THEME_CORE = """
function createThemeCore(SWAP) {
  var TOKEN = /#[0-9a-f]{3,8}(?![0-9a-f])|rgba?\\([^)]*\\)/gi;
  // Keys that hold the data itself. Plotly may hand these back as typed arrays.
  var BIG = {
    x: 1, y: 1, z: 1, customdata: 1, ids: 1, text: 1, hovertext: 1, lat: 1, lon: 1,
    locations: 1, base: 1, open: 1, high: 1, low: 1, close: 1, array: 1, arrayminus: 1,
    values: 1, labels: 1, r: 1, theta: 1, a: 1, b: 1, c: 1
  };

  // Every colour token in a string, so inline styles inside titles and annotations follow too.
  function swap(text) {
    return text.replace(TOKEN, function (token) {
      var dark = SWAP[token.toUpperCase().replace(/\\s+/g, "")];
      return dark === undefined ? token : dark;
    });
  }

  function isObject(node) {
    return node !== null && typeof node === "object";
  }

  // Large numeric data and typed arrays never carry a colour: share them instead of copying.
  function shared(node, key) {
    if (Array.isArray(node)) {
      if (BIG[key] === 1) { return true; }
      var last = node.length - 1;
      return last >= 0 && typeof node[0] === "number" && typeof node[last] === "number";
    }
    return ArrayBuffer.isView(node) || "bdata" in node;
  }

  function isPlain(node) {
    return isObject(node) && !Array.isArray(node);
  }

  // A copy of a spec that shares its big data arrays; later in-place edits by plotly cannot reach it.
  function snap(node, key) {
    if (!isObject(node) || shared(node, key)) { return node; }
    if (Array.isArray(node)) {
      return node.map(function (item) { return snap(item, key); });
    }
    var copy = {};
    Object.keys(node).forEach(function (name) {
      copy[name] = name.charAt(0) === "_" ? node[name] : snap(node[name], name);
    });
    return copy;
  }

  // The current spec with every colour taken from the light original: as it was, or swapped to dark.
  // Everything else (ranges, visibility) stays as the viewer left it.
  function remap(current, original, dark, key) {
    if (typeof current === "string") {
      if (typeof original === "string" && swap(original) !== original) {
        return dark ? swap(original) : original;
      }
      return current;
    }
    if (!isObject(current) || shared(current, key)) { return current; }
    if (Array.isArray(current)) {
      return current.map(function (item, index) {
        return remap(item, Array.isArray(original) ? original[index] : undefined, dark, key);
      });
    }
    var base = isPlain(original) ? original : {};
    var copy = {};
    Object.keys(current).forEach(function (name) {
      copy[name] = name.charAt(0) === "_" ? current[name] : remap(current[name], base[name], dark, name);
    });
    return copy;
  }

  return { swap: swap, snap: snap, remap: remap };
}
""".strip()

#: The pure part of the navigation bar: where it scrolls to. ``createNavCore()``
#: returns ``{reveal, wheel}``; it touches no DOM either.
NAV_CORE = """
function createNavCore() {
  // The scroll position that shows a link (start to end) with some room around it.
  function reveal(start, end, scrollLeft, width, pad) {
    if (start - pad < scrollLeft) { return Math.max(0, start - pad); }
    if (end + pad > scrollLeft + width) { return end + pad - width; }
    return scrollLeft;
  }

  // Where a vertical wheel turn moves a bar that scrolls sideways, or null to leave the turn to the page
  // (nothing to scroll, or the bar is already at that end).
  function wheel(scrollLeft, room, deltaY, deltaMode) {
    if (room <= 0) { return null; }
    var step = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * room : deltaY;
    var next = Math.min(room, Math.max(0, scrollLeft + step));
    return next === scrollLeft ? null : next;
  }

  return { reveal: reveal, wheel: wheel };
}
""".strip()

_THEME_SHELL = """
(function () {
  "use strict";
  var STORAGE_KEY = "__KEY__";
  var core = createThemeCore(__SWAP__);
  var navCore = createNavCore();
  var root = document.documentElement;
  var button = document.getElementById("theme-toggle");
  var fallback = root.getAttribute("data-theme-default") || "light";
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  var originals = new WeakMap();
  var drawn = "light";
  var printing = false;
  var queue = Promise.all(window.__prophetPlots || []).catch(function (error) {
    console.error("Plotly could not draw a figure:", error);
  });

  function remembered() {
    try {
      var saved = window.localStorage.getItem(STORAGE_KEY);
      return saved === "light" || saved === "dark" ? saved : null;
    } catch (error) {
      return null;
    }
  }

  function remember(mode) {
    try {
      window.localStorage.setItem(STORAGE_KEY, mode);
    } catch (error) {
      // The choice then lasts for this visit only.
    }
  }

  function system() {
    return media && media.matches ? "dark" : "light";
  }

  function listen(query, handler) {
    if (query.addEventListener) { query.addEventListener("change", handler); }
    else if (query.addListener) { query.addListener(handler); }
  }

  function redraw(plot, dark) {
    if (!plot.data || !plot.layout) { return Promise.resolve(); }
    var original = originals.get(plot);
    if (!original) {
      original = { data: core.snap(plot.data, ""), layout: core.snap(plot.layout, "") };
      originals.set(plot, original);
    }
    return window.Plotly.react(
      plot,
      core.remap(plot.data, original.data, dark, ""),
      core.remap(plot.layout, original.layout, dark, "")
    );
  }

  // Paper is light whatever the viewer chose on screen; the print style sheet does the same for the page.
  function restyleFigures() {
    var want = !printing && root.dataset.theme === "dark" ? "dark" : "light";
    if (want === drawn) { return undefined; }
    drawn = want;
    var plots = Array.prototype.slice.call(document.querySelectorAll(".js-plotly-plot"));
    return Promise.all(plots.map(function (plot) { return redraw(plot, want === "dark"); }));
  }

  function refresh() {
    queue = queue.then(restyleFigures).catch(function (error) {
      console.error("The dashboard could not restyle a figure:", error);
    }).then(function () { root.classList.remove("theme-pending"); });
    return queue;
  }

  function applyTheme(mode) {
    root.dataset.theme = mode;
    if (button) { button.setAttribute("aria-pressed", mode === "dark" ? "true" : "false"); }
    return refresh();
  }

  function setPrinting(on) {
    if (printing === on) { return; }
    printing = on;
    refresh();
  }

  var choice = remembered();
  applyTheme(choice || (fallback === "auto" ? system() : fallback === "dark" ? "dark" : "light"));

  if (button) {
    button.addEventListener("click", function () {
      var next = root.dataset.theme === "dark" ? "light" : "dark";
      choice = next;
      remember(next);
      applyTheme(next);
    });
  }

  // On a narrow screen the legends wrap into rows that run into the range buttons, so they move
  // under the plot; a wide screen puts back each figure's own place. Queued after the restyles,
  // which keep any layout value that is not a colour, so the two never undo each other.
  var narrow = window.matchMedia ? window.matchMedia("(max-width: 640px)") : null;
  var LEGEND_KEYS = ["x", "xanchor", "y", "yanchor", "yref"];
  var LEGEND_UNDER = { "legend.x": 0, "legend.xanchor": "left", "legend.y": 0, "legend.yanchor": "bottom", "legend.yref": "container" };
  var legendSpots = new WeakMap();
  var legendsUnder = false;

  function legendSpot(plot) {
    var spot = legendSpots.get(plot);
    if (!spot) {
      var legend = plot.layout.legend || {};
      spot = {};
      LEGEND_KEYS.forEach(function (name) {
        spot["legend." + name] = legend[name] === undefined ? null : legend[name];
      });
      legendSpots.set(plot, spot);
    }
    return spot;
  }

  function placeLegends(under) {
    if (under === legendsUnder) { return undefined; }
    legendsUnder = under;
    var plots = Array.prototype.slice.call(document.querySelectorAll(".js-plotly-plot"));
    return Promise.all(plots.map(function (plot) {
      if (!plot.layout || plot.layout.showlegend === false) { return undefined; }
      var spot = legendSpot(plot);
      return window.Plotly.relayout(plot, under ? LEGEND_UNDER : spot);
    }));
  }

  if (narrow) {
    var relegend = function (under) {
      queue = queue.then(function () { return placeLegends(Boolean(under)); }).catch(function (error) {
        console.error("The dashboard could not move a legend:", error);
      });
    };
    relegend(narrow.matches);
    listen(narrow, function (event) { relegend(event.matches); });
  }

  if (media && fallback === "auto") {
    listen(media, function (event) {
      if (!choice) { applyTheme(event.matches ? "dark" : "light"); }
    });
  }

  // Browsers start printing as soon as these handlers return, so the restyle is started at once.
  window.addEventListener("beforeprint", function () { setPrinting(true); });
  window.addEventListener("afterprint", function () { setPrinting(false); });
  var printQuery = window.matchMedia ? window.matchMedia("print") : null;
  if (printQuery) { listen(printQuery, function (event) { setPrinting(event.matches); }); }

  // The navigation bar scrolls sideways when the sections do not fit: fade the clipped edges, turn a
  // wheel over the bar into sideways scrolling, and keep the link of the section in view on show.
  var nav = document.querySelector(".topbar nav");
  var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function fades() {
    var room = nav.scrollWidth - nav.clientWidth;
    nav.classList.toggle("more-l", nav.scrollLeft > 1);
    nav.classList.toggle("more-r", nav.scrollLeft < room - 1);
  }

  function reveal(link) {
    var room = nav.scrollWidth - nav.clientWidth;
    var left = Math.min(room, navCore.reveal(
      link.offsetLeft, link.offsetLeft + link.offsetWidth, nav.scrollLeft, nav.clientWidth, 40
    ));
    if (left !== nav.scrollLeft) { nav.scrollTo({ left: left, behavior: still ? "auto" : "smooth" }); }
  }

  if (nav) {
    fades();
    nav.addEventListener("scroll", fades, { passive: true });
    window.addEventListener("resize", fades);
    window.addEventListener("load", fades);
    nav.addEventListener("wheel", function (event) {
      if (event.deltaX !== 0 || event.ctrlKey) { return; }
      var next = navCore.wheel(nav.scrollLeft, nav.scrollWidth - nav.clientWidth, event.deltaY, event.deltaMode);
      if (next === null) { return; }
      nav.scrollLeft = next;
      event.preventDefault();
    }, { passive: false });
  }

  // Mark the section in view in the navigation bar.
  var links = Array.prototype.slice.call(document.querySelectorAll('.topbar nav a[href^="#"]'));
  if (nav && links.length && "IntersectionObserver" in window) {
    var byId = {};
    links.forEach(function (link) { byId[link.getAttribute("href").slice(1)] = link; });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var link = byId[entry.target.id];
        if (!entry.isIntersecting || !link) { return; }
        links.forEach(function (other) { other.removeAttribute("aria-current"); });
        link.setAttribute("aria-current", "true");
        reveal(link);
      });
    }, { rootMargin: "-15% 0px -75% 0px" });
    Array.prototype.forEach.call(document.querySelectorAll("main section[id]"), function (section) {
      observer.observe(section);
    });
  }
})();
""".strip()


def _json_for_script(value: object) -> str:
    """JSON that is safe inside a ``<script>`` element."""
    return json.dumps(value, sort_keys=True).replace("</", "<\\/")


def theme_script(swap: Mapping[str, str]) -> str:
    """The end-of-body script, with the light to dark colour table ``swap`` inlined."""
    shell = _THEME_SHELL.replace("__KEY__", STORAGE_KEY).replace(
        "__SWAP__", _json_for_script(dict(swap))
    )
    return f"{THEME_CORE}\n{NAV_CORE}\n{shell}"
