"""EXPERIMENTAL MLX backend: batched MAP fitting of Prophet models on Apple Silicon.

Unlike the CmdStanPy backend, this module does NOT call Stan. It re-implements
the objective of ``stan/prophet.stan`` (as run by ``cmdstanpy`` ``optimize``,
which omits the Jacobian of the ``sigma_obs`` constraint) and minimises it with
a batched, damped Newton method, vectorised across many series at once.

Supported: ``growth`` in {"linear", "flat"}, MAP estimation only. Logistic growth
and MCMC raise ``NotImplementedError``.

Objective (negative log posterior, constants dropped as Stan's ``~`` does)::

    f = (k^2 + m^2) / 50                       # k, m ~ normal(0, 5)
      + sum_s |delta_s| / tau                  # delta ~ double_exponential(0, tau)
      + 2 sigma^2                              # sigma ~ normal(0, 0.5), half-normal
      + sum_j beta_j^2 / (2 sigmas_j^2)        # beta ~ normal(0, sigmas)
      + T log sigma + sum_t r_t^2 / (2 sigma^2)
    r = y - (trend * (1 + X_sm beta) + X_sa beta)

Differences from Stan, by design:

* The Laplace prior is handled exactly (no smoothing) with an orthant-wise
  damped Newton method: zero coefficients stay at zero unless the data gradient
  exceeds ``1/tau``, and steps are kept inside the orthant they start in. MLX has
  no L-BFGS and no GPU linear solver, so Newton steps on the ~50 parameters are
  solved with batched LAPACK on the CPU while the O(T) work runs on the GPU.
* The exact cross term between ``sigma_obs`` and the other parameters is dropped
  from the Hessian (Fisher scoring) to keep it positive definite far from the
  optimum. It does not change the fixed point.
* Optimisation runs in float32 on the GPU (MLX has no float64 on the GPU), so
  objective values agree with float64 to about 1e-3 nats. ``device="cpu"`` uses
  float64 for parity testing.
* The convergence test is on the predicted decrease of the objective in nats
  (absolute, default 1e-4 on the GPU), not Stan's relative tolerances. Stan's own
  L-BFGS typically stops 0.1 to 3 nats short of the optimum at the Laplace kinks,
  so this backend usually reaches a slightly better objective than CmdStan.
* The starting value of ``sigma_obs`` is the residual scale at the Prophet
  initial values rather than 1.0; the optimum is unaffected.

mlx is imported lazily so ``import prophet`` works without it.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from prophet.models import IStanBackend

if TYPE_CHECKING:
    from prophet.forecaster import Prophet

logger: logging.Logger = logging.getLogger("prophet.mlx_backend")

Device = Literal["gpu", "cpu"]
FloatArray = npt.NDArray[np.float64]

DEVICE_ENV_VAR = "NEOPROPHET_MLX_DEVICE"
_DEFAULT_DEVICE: Device = "gpu"
# Upper bound on B * T * (2 + S + K) held per chunk, to bound GPU memory.
_DEFAULT_CHUNK_ELEMENTS = 40_000_000
# Padded changepoint location: beyond any scaled time, so its column is all zero.
_PAD_T_CHANGE = 1.0e3
_SOLVE_THREADS = max(1, min(8, os.cpu_count() or 1))
_SOLVE_MIN_PER_THREAD = 16
_SUPPORTED_FIT_KWARGS = ("init", "device", "max_iter", "tol", "chunk_elements")


def _require_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:
        raise ImportError(
            "The MLX backend needs the 'mlx' package (Apple Silicon only). "
            "Install it with: pip install 'prophet[mlx]'"
        ) from e
    return mx


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesInput:
    """Unpadded Stan data and starting values for one series."""

    y: FloatArray  # (T,)
    t: FloatArray  # (T,)
    t_change: FloatArray  # (S,)
    X: FloatArray  # (T, K)
    sigmas: FloatArray  # (K,)
    s_a: FloatArray  # (K,)
    s_m: FloatArray  # (K,)
    tau: float
    trend_indicator: int
    k0: float
    m0: float
    delta0: FloatArray  # (S,)
    beta0: FloatArray  # (K,)


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings. ``tol`` is the predicted objective decrease in nats."""

    device: Device = "gpu"
    max_iter: int = 200
    tol: float | None = None  # None: 1e-4 (float32) or 1e-9 (float64)
    initial_damping: float = 1e-2
    pin_passes: int = 2
    chunk_elements: int = _DEFAULT_CHUNK_ELEMENTS

    @property
    def resolved_tol(self) -> float:
        if self.tol is not None:
            return self.tol
        return 1e-4 if self.device == "gpu" else 1e-9


@dataclass(frozen=True)
class SeriesResult:
    """MAP estimate for one series, unpadded, in Stan's parameterisation."""

    k: float
    m: float
    delta: FloatArray
    beta: FloatArray
    sigma_obs: float
    objective: float  # exact (non-smoothed) negative log posterior
    converged: bool
    iterations: int


@dataclass(frozen=True)
class _Padded:
    """Series padded to a common (T, S, K) and stacked, float64 numpy."""

    y: FloatArray  # (B, T)
    w: FloatArray  # (B, T) 1 for real rows
    t: FloatArray  # (B, T)
    t_change: FloatArray  # (B, S)
    delta_mask: FloatArray  # (B, S)
    X_sa: FloatArray  # (B, T, K)
    X_sm: FloatArray  # (B, T, K)
    sigmas: FloatArray  # (B, K)
    tau: FloatArray  # (B,)
    n_obs: FloatArray  # (B,)
    trend_on: FloatArray  # (B,) 1 for linear, 0 for flat
    x0: FloatArray  # (B, 2 + S + K + 1), last column is log sigma_obs


# --------------------------------------------------------------------------
# Validation and padding (pure numpy)
# --------------------------------------------------------------------------


def series_from_stan(
    stan_init: dict[str, Any], stan_data: dict[str, Any]
) -> SeriesInput:
    """Convert the ``(stan_init, stan_data)`` pair Prophet builds into a SeriesInput."""
    indicator = int(stan_data["trend_indicator"])
    if indicator == 1:
        raise NotImplementedError(
            "The MLX backend does not support logistic growth; use growth='linear' "
            "or 'flat', or the CMDSTANPY backend."
        )
    if indicator not in (0, 2):
        raise ValueError(f"Unknown trend_indicator {indicator}")

    def vec(x: Any) -> FloatArray:
        return np.asarray(x, dtype=np.float64).reshape(-1)

    X = np.asarray(stan_data["X"], dtype=np.float64)
    y, t = vec(stan_data["y"]), vec(stan_data["t"])
    s_a, s_m, sigmas = (
        vec(stan_data["s_a"]),
        vec(stan_data["s_m"]),
        vec(stan_data["sigmas"]),
    )
    t_change = vec(stan_data["t_change"])
    if X.ndim != 2 or X.shape[0] != y.size or t.size != y.size:
        raise ValueError(f"Inconsistent shapes: X {X.shape}, y {y.shape}, t {t.shape}")
    K = X.shape[1]
    if not (s_a.size == s_m.size == sigmas.size == K):
        raise ValueError("s_a, s_m and sigmas must have one entry per column of X")
    if K < 1:
        raise ValueError("Stan model requires K >= 1")
    tau = float(stan_data["tau"])
    if not tau > 0:
        raise ValueError("tau (changepoint_prior_scale) must be positive")
    if np.any(sigmas <= 0):
        raise ValueError("All prior scales (sigmas) must be positive")
    for name, arr in (("y", y), ("t", t), ("X", X), ("t_change", t_change)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in {name}")
    return SeriesInput(
        y=y,
        t=t,
        t_change=t_change,
        X=X,
        sigmas=sigmas,
        s_a=s_a,
        s_m=s_m,
        tau=tau,
        trend_indicator=indicator,
        k0=float(stan_init["k"]),
        m0=float(stan_init["m"]),
        delta0=vec(stan_init["delta"]),
        beta0=vec(stan_init["beta"]),
    )


def _pad(series: Sequence[SeriesInput]) -> _Padded:
    B = len(series)
    T = max(s.y.size for s in series)
    S = max(s.t_change.size for s in series)
    K = max(s.X.shape[1] for s in series)
    y = np.zeros((B, T))
    w = np.zeros((B, T))
    t = np.zeros((B, T))
    tc = np.full((B, S), _PAD_T_CHANGE)
    dmask = np.zeros((B, S))
    X_sa = np.zeros((B, T, K))
    X_sm = np.zeros((B, T, K))
    sig = np.ones((B, K))
    x0 = np.zeros((B, 2 + S + K + 1))
    for b, s in enumerate(series):
        n, ns, nk = s.y.size, s.t_change.size, s.X.shape[1]
        y[b, :n], w[b, :n], t[b, :n] = s.y, 1.0, s.t
        tc[b, :ns], dmask[b, :ns] = s.t_change, 1.0
        X_sa[b, :n, :nk] = s.X * s.s_a
        X_sm[b, :n, :nk] = s.X * s.s_m
        sig[b, :nk] = s.sigmas
        x0[b, 0], x0[b, 1] = s.k0, s.m0
        x0[b, 2 : 2 + ns] = s.delta0
        x0[b, 2 + S : 2 + S + nk] = s.beta0
    return _Padded(
        y=y,
        w=w,
        t=t,
        t_change=tc,
        delta_mask=dmask,
        X_sa=X_sa,
        X_sm=X_sm,
        sigmas=sig,
        tau=np.array([s.tau for s in series]),
        n_obs=np.array([float(s.y.size) for s in series]),
        trend_on=np.array([1.0 if s.trend_indicator == 0 else 0.0 for s in series]),
        x0=x0,
    )


# --------------------------------------------------------------------------
# Exact objective and trend (pure numpy, float64): used for reporting and tests
# --------------------------------------------------------------------------


def linear_or_flat_trend(
    k: float,
    m: float,
    delta: FloatArray,
    t: FloatArray,
    t_change: FloatArray,
    trend_indicator: int,
) -> FloatArray:
    """Trend exactly as ``linear_trend`` / ``flat_trend`` in prophet.stan."""
    if trend_indicator == 2:
        return np.full(t.shape, m)
    A = (t[:, None] >= t_change[None, :]).astype(np.float64)
    return (k + A @ delta) * t + (m + A @ (-t_change * delta))


def _objective(
    s: SeriesInput,
    k: float,
    m: float,
    delta: FloatArray,
    beta: FloatArray,
    sigma_obs: float,
) -> float:
    trend = linear_or_flat_trend(k, m, delta, s.t, s.t_change, s.trend_indicator)
    mu = trend * (1.0 + (s.X * s.s_m) @ beta) + (s.X * s.s_a) @ beta
    r = s.y - mu
    return float(
        (k**2 + m**2) / 50.0
        + np.abs(delta).sum() / s.tau
        + 2.0 * sigma_obs**2
        + np.sum(beta**2 / (2.0 * s.sigmas**2))
        + s.y.size * np.log(sigma_obs)
        + np.sum(r**2) / (2.0 * sigma_obs**2)
    )


def negative_log_posterior(
    stan_data: dict[str, Any],
    k: float,
    m: float,
    delta: FloatArray,
    beta: FloatArray,
    sigma_obs: float,
) -> float:
    """Objective of prophet.stan (``~`` statements, constants dropped), minimised.

    Equals ``-lp__`` from CmdStan optimize (checked on the example datasets),
    with the exact Laplace prior and no Jacobian for ``sigma_obs``.
    """
    init = {"k": k, "m": m, "delta": delta, "beta": beta}
    s = series_from_stan(init, stan_data)
    return _objective(s, k, m, s.delta0, s.beta0, sigma_obs)


# --------------------------------------------------------------------------
# Batched Newton solver (MLX)
# --------------------------------------------------------------------------


@functools.cache
def _solve_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_SOLVE_THREADS, thread_name_prefix="mlx-solve"
    )


def _solve_systems(A: FloatArray, b: FloatArray) -> FloatArray:
    """Solve ``A[i] x = b[i]`` for a stack of small dense systems, using threads.

    MLX has no GPU linear solver; for ~50 unknowns LAPACK on the CPU is the
    right tool, and LAPACK releases the GIL so threads give real parallelism.
    """
    n = A.shape[0]
    if n <= _SOLVE_MIN_PER_THREAD:
        return np.linalg.solve(A, b[:, :, None])[:, :, 0]
    bounds = np.linspace(
        0, n, min(_SOLVE_THREADS, n // _SOLVE_MIN_PER_THREAD) + 1
    ).astype(int)
    out = np.empty_like(b)

    def work(lo: int, hi: int) -> None:
        out[lo:hi] = np.linalg.solve(A[lo:hi], b[lo:hi, :, None])[:, :, 0]

    list(_solve_pool().map(work, bounds[:-1], bounds[1:]))
    return out


def _newton_solve(
    g: FloatArray, H: FloatArray, free: FloatArray, lam: FloatArray
) -> FloatArray:
    """Levenberg-Marquardt step restricted to the coordinates flagged in ``free``."""
    P = g.shape[1]
    eye = np.eye(P)[None]
    Hm = H * free[:, :, None] * free[:, None, :] + (1.0 - free)[:, :, None] * eye
    # Jacobi scaling keeps badly scaled systems well conditioned.
    d = np.maximum(np.sqrt(np.abs(np.diagonal(Hm, axis1=1, axis2=2))), 1e-6)
    Hs = Hm / (d[:, :, None] * d[:, None, :]) + lam[:, None, None] * eye
    return _solve_systems(Hs, -g * free / d) / d


def _orthant_step(
    x: FloatArray,
    g: FloatArray,
    H: FloatArray,
    lam: FloatArray,
    inv_tau: FloatArray,
    dmask: FloatArray,
    Q: int,
    pin_passes: int,
) -> tuple[FloatArray, FloatArray]:
    """Orthant-wise Newton step for the exact Laplace-penalised objective.

    ``g`` and ``H`` come from the smooth part only. Returns the trial point and
    the predicted decrease (half the Newton decrement of the pseudo-gradient).
    """
    delta, gd = x[:, 2:Q], g[:, 2:Q]
    up, down = gd + inv_tau, gd - inv_tau
    at_zero = np.where(up < 0, up, np.where(down > 0, down, 0.0))
    pg_delta = np.where(delta != 0, gd + np.sign(delta) * inv_tau, at_zero) * dmask
    pg = np.concatenate([g[:, :2], pg_delta, g[:, Q:]], axis=1)
    free = (pg != 0).astype(np.float64)
    orth = np.where(delta != 0, np.sign(delta), -np.sign(pg_delta))

    # Coordinates that would leave their orthant are pinned at zero and the step
    # is recomputed for those series only, so the others can compensate.
    pinned = np.zeros(delta.shape, dtype=bool)
    step = np.zeros_like(pg)
    todo = np.arange(x.shape[0])
    for _ in range(pin_passes + 1):
        free_k = free[todo].copy()
        free_k[:, 2:Q] *= ~pinned[todo]
        step[todo] = _newton_solve(pg[todo], H[todo], free_k, lam[todo])
        crossing = ((delta[todo] + step[todo, 2:Q]) * orth[todo] <= 0) & (
            pg_delta[todo] != 0
        )
        newly = crossing & ~pinned[todo]
        pinned[todo] |= crossing
        todo = todo[newly.any(axis=1)]
        if todo.size == 0:
            break
    dec = -0.5 * np.sum(pg * step, axis=1)
    trial = delta + step[:, 2:Q]
    trial = np.where((trial * orth > 0) & ~pinned, trial, 0.0) * dmask
    x_try = np.concatenate(
        [x[:, :2] + step[:, :2], trial, x[:, Q:] + step[:, Q:]], axis=1
    )
    return x_try, dec


def _solve_chunk(
    p: _Padded, cfg: SolverConfig
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Minimise the exact objective for one padded chunk.

    Orthant-wise damped Newton: the smooth part of the objective supplies
    gradient and Hessian, the Laplace prior is handled through its
    pseudo-gradient (a zero coefficient stays at zero unless the data gradient
    exceeds 1/tau) and steps never leave the orthant they started in.

    Returns ``(x, converged, iterations)`` as numpy arrays.
    """
    mx = _require_mlx()
    dt = mx.float32 if cfg.device == "gpu" else mx.float64
    device = mx.gpu if cfg.device == "gpu" else mx.cpu

    B, T = p.y.shape
    S = p.t_change.shape[1]
    K = p.sigmas.shape[1]
    Q = 2 + S
    tol = cfg.resolved_tol

    with mx.stream(device):

        def arr(a: np.ndarray) -> Any:
            return mx.array(a, dtype=dt)

        y, w, t = arr(p.y), arr(p.w), arr(p.t)
        tc, dmask = arr(p.t_change), arr(p.delta_mask)
        X_sa, X_sm = arr(p.X_sa), arr(p.X_sm)
        sigmas, tau, n_obs = arr(p.sigmas), arr(p.tau), arr(p.n_obs)
        trend_on = arr(p.trend_on)
        multiplicative = bool(np.any(p.X_sm != 0.0))

        # G maps (k, m, delta) to the trend: trend = G @ theta. For flat growth
        # the k and delta columns are zeroed, leaving only their priors.
        A = (t[:, :, None] >= tc[:, None, :]).astype(dt)
        on = trend_on[:, None]
        G = mx.concatenate(
            [
                (t * on)[:, :, None],
                mx.ones((B, T, 1), dtype=dt),
                A * (t[:, :, None] - tc[:, None, :]) * on[:, :, None],
            ],
            axis=2,
        )
        Gt = G.transpose(0, 2, 1)
        inv_two_sig2 = 0.5 / (sigmas * sigmas)
        inv_tau = (1.0 / tau)[:, None]
        eye_xi = mx.eye(Q + K, dtype=dt)[None]
        zeros_s = mx.zeros((B, S), dtype=dt)
        d_prior = mx.concatenate(
            [mx.full((B, 2), 1.0 / 25.0, dtype=dt), zeros_s, 1.0 / (sigmas * sigmas)],
            axis=1,
        )
        if not multiplicative:
            # Without multiplicative features the model is linear in (theta, beta),
            # so the Jacobian and its Gram matrix are fixed.
            J_fixed = mx.concatenate([G, X_sa], axis=2)
            Jt_fixed = J_fixed.transpose(0, 2, 1)
            gram = Jt_fixed @ (J_fixed * w[:, :, None])
            mx.eval(J_fixed, gram)

        def residual(x: Any) -> tuple[Any, Any, Any]:
            theta, beta = x[:, :Q], x[:, Q : Q + K]
            if not multiplicative:
                r = (y - (J_fixed @ x[:, : Q + K, None])[:, :, 0]) * w
                return r, None, None
            z = (G @ theta[:, :, None])[:, :, 0]
            M = 1.0 + (X_sm @ beta[:, :, None])[:, :, 0]
            add = (X_sa @ beta[:, :, None])[:, :, 0]
            return (y - (z * M + add)) * w, z, M

        def objective(x: Any) -> Any:
            theta, beta, u = x[:, :Q], x[:, Q : Q + K], x[:, -1]
            r = residual(x)[0]
            return (
                (theta[:, 0] ** 2 + theta[:, 1] ** 2) / 50.0
                + mx.sum(mx.abs(theta[:, 2:]) * dmask, axis=1) * inv_tau[:, 0]
                + 2.0 * mx.exp(2.0 * u)
                + mx.sum(beta * beta * inv_two_sig2, axis=1)
                + n_obs * u
                + 0.5 * mx.exp(-2.0 * u) * mx.sum(r * r, axis=1)
            )

        def smooth_grad_hess(x: Any) -> tuple[Any, Any]:
            """Gradient and Hessian of the objective without the Laplace term."""
            theta, beta, u = x[:, :Q], x[:, Q : Q + K], x[:, -1]
            r, z, M = residual(x)
            hh = mx.exp(-2.0 * u)
            ssr = mx.sum(r * r, axis=1)
            if M is None:
                Jt_ = Jt_fixed
                gram_w = gram
            else:
                J = mx.concatenate(
                    [M[:, :, None] * G, z[:, :, None] * X_sm + X_sa], axis=2
                )
                Jt_ = J.transpose(0, 2, 1)
                gram_w = Jt_ @ (J * w[:, :, None])
            g_xi = -hh[:, None] * (Jt_ @ r[:, :, None])[:, :, 0]  # data term, (B, Q+K)
            prior = mx.concatenate(
                [theta[:, :2] / 25.0, zeros_s, beta / (sigmas * sigmas)], axis=1
            )
            g = mx.concatenate(
                [g_xi + prior, (n_obs + 4.0 * mx.exp(2.0 * u) - hh * ssr)[:, None]],
                axis=1,
            )
            H_xi = hh[:, None, None] * gram_w
            if M is not None:
                # Second derivative of mu couples the trend and the multiplicative betas.
                corr = -hh[:, None, None] * (Gt @ (r[:, :, None] * X_sm))  # (B, Q, K)
                top = mx.concatenate([mx.zeros((B, Q, Q), dtype=dt), corr], axis=2)
                bottom = mx.concatenate(
                    [corr.transpose(0, 2, 1), mx.zeros((B, K, K), dtype=dt)], axis=2
                )
                H_xi = H_xi + mx.concatenate([top, bottom], axis=1)
            H_xi = H_xi + eye_xi * d_prior[:, :, None]
            # The exact sigma cross term, -2 g_xi, vanishes in expectation and can
            # make the Hessian indefinite far from the optimum, so it is dropped
            # (Fisher scoring). It tends to zero at the optimum anyway.
            h_uu = 8.0 * mx.exp(2.0 * u) + 2.0 * hh * ssr
            top = mx.concatenate([H_xi, mx.zeros((B, Q + K, 1), dtype=dt)], axis=2)
            bottom = mx.concatenate(
                [mx.zeros((B, 1, Q + K), dtype=dt), h_uu[:, None, None]], axis=2
            )
            return g, mx.concatenate([top, bottom], axis=1)

        def to_np(a: Any) -> FloatArray:
            mx.eval(a)
            return np.array(a, dtype=np.float64)

        # Start sigma_obs at the residual scale of the initial values.
        x = p.x0.copy()
        r0 = to_np(residual(arr(x))[0])
        x[:, -1] = np.log(np.sqrt(np.maximum(np.sum(r0 * r0, axis=1) / p.n_obs, 1e-12)))

        rel_noise = 2e-7 if cfg.device == "gpu" else 1e-14
        lam = np.full(B, cfg.initial_damping)
        iters = np.zeros(B)
        done = np.zeros(B, dtype=bool)
        inv_tau_np = (1.0 / p.tau)[:, None]
        f = to_np(objective(arr(x)))
        for _ in range(cfg.max_iter):
            g_mx, H_mx = smooth_grad_hess(arr(x))
            g, H = to_np(g_mx), to_np(H_mx)
            x_try, dec = _orthant_step(
                x, g, H, lam, inv_tau_np, p.delta_mask, Q, cfg.pin_passes
            )
            f_try = to_np(objective(arr(x_try)))
            finite = np.isfinite(f_try) & np.isfinite(dec)
            noise = rel_noise * (np.abs(f) + 1.0)
            accept = finite & (dec > 0) & (f_try <= f + noise) & ~done
            done = done | (finite & (np.abs(dec) < tol))
            iters += ~done | accept
            x = np.where(accept[:, None], x_try, x)
            f = np.where(accept, f_try, f)
            if done.all():
                break
            lam = np.where(
                accept, np.maximum(lam / 3.0, 1e-9), np.minimum(lam * 5.0, 1e12)
            )
        return x, done, iters


def _chunk_bounds(sizes: tuple[int, int, int, int], cfg: SolverConfig) -> int:
    B, T, S, K = sizes
    per_series = T * (2 + S + K) + 1
    return max(1, min(B, cfg.chunk_elements // per_series))


def solve_batch(
    series: Sequence[SeriesInput], cfg: SolverConfig | None = None
) -> list[SeriesResult]:
    """MAP estimates for many independent series in one batched optimisation."""
    cfg = cfg or SolverConfig()
    if not series:
        raise ValueError("No series to fit")
    if cfg.device not in ("gpu", "cpu"):
        raise ValueError(f"device must be 'gpu' or 'cpu', got {cfg.device!r}")
    _require_mlx()
    T = max(s.y.size for s in series)
    S = max(s.t_change.size for s in series)
    K = max(s.X.shape[1] for s in series)
    size = _chunk_bounds((len(series), T, S, K), cfg)
    results: list[SeriesResult] = []
    for start in range(0, len(series), size):
        part = series[start : start + size]
        padded = _pad(part)
        x, conv, iters = _solve_chunk(padded, cfg)
        Sp = padded.t_change.shape[1]
        for b, s in enumerate(part):
            ns, nk = s.t_change.size, s.X.shape[1]
            delta = x[b, 2 : 2 + ns].copy()
            beta = x[b, 2 + Sp : 2 + Sp + nk].copy()
            sigma = float(np.exp(x[b, -1]))
            obj = _objective(s, float(x[b, 0]), float(x[b, 1]), delta, beta, sigma)
            results.append(
                SeriesResult(
                    k=float(x[b, 0]),
                    m=float(x[b, 1]),
                    delta=delta,
                    beta=beta,
                    sigma_obs=sigma,
                    objective=obj,
                    converged=bool(conv[b]),
                    iterations=int(iters[b]),
                )
            )
    unconverged = sum(not r.converged for r in results)
    if unconverged:
        logger.warning(
            "%d of %d series did not meet the convergence tolerance",
            unconverged,
            len(results),
        )
    return results


def _result_to_params(r: SeriesResult, s: SeriesInput) -> OrderedDict[str, FloatArray]:
    """Shape a result like ``CmdStanPyBackend.fit``: every entry reshaped to (1, -1)."""
    trend = linear_or_flat_trend(r.k, r.m, r.delta, s.t, s.t_change, s.trend_indicator)
    params: OrderedDict[str, FloatArray] = OrderedDict()
    # lp__ is minus the objective, up to the constants Stan's ``~`` drops.
    params["lp__"] = np.array([-r.objective])
    params["k"] = np.array([r.k])
    params["m"] = np.array([r.m])
    params["delta"] = r.delta
    params["sigma_obs"] = np.array([r.sigma_obs])
    params["beta"] = r.beta
    params["trend"] = trend
    return OrderedDict((name, v.reshape((1, -1))) for name, v in params.items())


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


def _default_device() -> Device:
    value = os.environ.get(DEVICE_ENV_VAR, _DEFAULT_DEVICE)
    if value not in ("gpu", "cpu"):
        raise ValueError(f"{DEVICE_ENV_VAR} must be 'gpu' or 'cpu', got {value!r}")
    return cast(Device, value)


class MLXBackend(IStanBackend):
    """EXPERIMENTAL MAP-only backend that evaluates the Prophet objective with MLX.

    Select the device with ``set_options(device="cpu")`` (float64, for parity
    checks) or the ``NEOPROPHET_MLX_DEVICE`` environment variable; default GPU.
    """

    device: Device
    max_iter: int
    tol: float | None

    def __init__(self) -> None:
        _require_mlx()
        super().__init__()
        self.device = _default_device()
        self.max_iter = 200
        self.tol = None
        self.chunk_elements = _DEFAULT_CHUNK_ELEMENTS

    def set_options(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "device":
                if value not in ("gpu", "cpu"):
                    raise ValueError(f"device must be 'gpu' or 'cpu', got {value!r}")
                self.device = value
            elif key == "max_iter":
                self.max_iter = int(value)
            elif key == "tol":
                self.tol = None if value is None else float(value)
            elif key == "chunk_elements":
                self.chunk_elements = int(value)
            else:
                super().set_options(**{key: value})

    @staticmethod
    def get_type() -> str:
        return "MLX"

    def load_model(self) -> None:
        # There is no compiled model: the objective is evaluated directly.
        return None

    def _config(self, **overrides: Any) -> SolverConfig:
        return SolverConfig(
            device=overrides.get("device", self.device),
            max_iter=int(overrides.get("max_iter", self.max_iter)),
            tol=overrides.get("tol", self.tol),
            chunk_elements=int(overrides.get("chunk_elements", self.chunk_elements)),
        )

    def fit(
        self,
        stan_init: dict[str, float],
        stan_data: dict[str, Any],
        **kwargs: Any,
    ) -> OrderedDict[str, np.ndarray]:
        unknown = sorted(set(kwargs) - set(_SUPPORTED_FIT_KWARGS))
        if unknown:
            raise ValueError(
                f"Unsupported fit options for the MLX backend: {unknown}. "
                f"Supported: {list(_SUPPORTED_FIT_KWARGS)}"
            )
        if "init" in kwargs:
            from prophet.models import CmdStanPyBackend

            stan_init = CmdStanPyBackend.sanitize_custom_inits(
                stan_init, kwargs["init"]
            )
        series = series_from_stan(stan_init, stan_data)
        cfg = self._config(**{k: v for k, v in kwargs.items() if k != "init"})
        (result,) = solve_batch([series], cfg)
        return _result_to_params(result, series)

    def sampling(
        self,
        stan_init: dict[str, float],
        stan_data: dict[str, Any],
        samples: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "The MLX backend only supports MAP estimation (mcmc_samples=0). "
            "Use the CMDSTANPY backend for MCMC."
        )


# --------------------------------------------------------------------------
# Many-series fitting
# --------------------------------------------------------------------------


@functools.cache
def _mlx_prophet_class() -> type[Prophet]:
    from prophet.forecaster import Prophet

    class MLXProphet(Prophet):
        """Prophet whose backend is MLXBackend, avoiding a CmdStan model load."""

        def _load_stan_backend(self, stan_backend: str | None) -> None:
            self.stan_backend = MLXBackend()

    MLXProphet.__module__ = __name__
    return MLXProphet


def __getattr__(name: str) -> Any:
    # Lets pickle resolve the lazily built subclass by reference.
    if name == "MLXProphet":
        return _mlx_prophet_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def fit_many(
    dfs: Sequence[pd.DataFrame],
    *,
    configure: Callable[[Prophet], None] | None = None,
    device: Device | None = None,
    max_iter: int = 200,
    tol: float | None = None,
    chunk_elements: int = _DEFAULT_CHUNK_ELEMENTS,
    **prophet_kwargs: Any,
) -> list[Prophet]:
    """Fit one Prophet model per dataframe in a single batched MAP optimisation.

    Parameters
    ----------
    dfs: one history dataframe (columns ``ds``, ``y``, plus any regressors) per series.
    configure: optional hook called on each fresh model before preprocessing, for
        ``add_regressor``, ``add_seasonality``, ``add_country_holidays`` and so on.
    device: "gpu" (float32) or "cpu" (float64); default from NEOPROPHET_MLX_DEVICE.
    prophet_kwargs: passed to every ``Prophet(...)``. ``growth`` must be linear or
        flat, ``mcmc_samples`` must be 0, and ``stan_backend`` must not be given.

    Returns
    -------
    Fitted models, in input order, ready for ``predict``.
    """
    if not dfs:
        raise ValueError("dfs is empty")
    if "stan_backend" in prophet_kwargs:
        raise ValueError(
            "fit_many always uses the MLX backend; do not pass stan_backend"
        )
    if prophet_kwargs.get("mcmc_samples", 0) > 0:
        raise NotImplementedError(
            "fit_many supports MAP estimation only (mcmc_samples=0)"
        )
    growth = prophet_kwargs.get("growth", "linear")
    if growth == "logistic":
        raise NotImplementedError("fit_many does not support logistic growth")
    if growth not in ("linear", "flat"):
        raise ValueError(f"Unknown growth {growth!r}")

    cls = _mlx_prophet_class()
    models: list[Prophet] = []
    todo: list[tuple[int, SeriesInput, dict[str, Any]]] = []
    for i, df in enumerate(dfs):
        m = cls(**prophet_kwargs)
        if configure is not None:
            configure(m)
        if m.history is not None:
            raise ValueError("configure must not fit the model")
        inputs = m.preprocess(df)
        init = dataclasses.asdict(m.calculate_initial_params(inputs.K))
        data = dict(vars(inputs))
        history = cast(pd.DataFrame, m.history)
        if history["y"].min() == history["y"].max():
            # Same shortcut as Prophet.fit: a constant series needs no optimisation.
            init["sigma_obs"] = 1e-9
            m.params = {name: np.array([v]) for name, v in init.items()}
        else:
            todo.append((i, series_from_stan(init, data), init))
        models.append(m)

    if todo:
        cfg = SolverConfig(
            device=device or _default_device(),
            max_iter=max_iter,
            tol=tol,
            chunk_elements=chunk_elements,
        )
        results = solve_batch([s for _, s, _ in todo], cfg)
        for (i, s, _), r in zip(todo, results, strict=True):
            models[i].params = _result_to_params(r, s)

    for m in models:
        m.stan_fit = None
        # Same tail as Prophet.fit: with no changepoints, fold delta into k.
        if len(cast(pd.Series, m.changepoints)) == 0:
            m.params["k"] = m.params["k"] + m.params["delta"].reshape(-1)
            m.params["delta"] = np.zeros(m.params["delta"].shape).reshape((-1, 1))
    return models
