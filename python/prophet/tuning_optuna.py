"""Optuna search behind ``prophet.tuning.tune(search="optuna")``.

``tune`` imports this module on demand, so Optuna stays an optional
dependency: ``pip install 'neoprophet[optuna]'``.

Trials are proposed through Optuna's ask-and-tell interface and scored by the
same functions as grid search. In parallel mode the process pool scores one
batch at a time: the sampler proposes a batch, the pool scores it, and every
score is told to the study before the next batch is proposed. Batches rather
than a free-running queue keep the proposals reproducible for a given seed
and pool size, because they never depend on which worker finishes first.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import logging
import math
import os
import statistics
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "tune(search='optuna') needs Optuna. Install it with: "
        "pip install 'neoprophet[optuna]'"
    ) from exc

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState, create_trial

from .tuning import (
    FloatRange,
    IntRange,
    SearchDimension,
    Trial,
    _mp_context,
    _score_params,
    _score_params_by_fold,
    _Task,
    space_contains,
    space_size,
)

logger: logging.Logger = logging.getLogger("prophet")

# Completed trials the pruner needs before it compares, and folds it always
# lets run, so one noisy early fold cannot stop a trial.
PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_FOLDS = 1

# Trials in one batch never see each other's scores, so the default pool for
# Optuna is capped to leave at least this many informed rounds.
MIN_ROUNDS = 4

Score = Callable[[Mapping[str, float]], float]
Runner = Callable[[Sequence[_Task]], list[Trial]]


# ---------------------------------------------------------------------------
# Pure conversion
# ---------------------------------------------------------------------------


def to_distribution(dim: SearchDimension) -> BaseDistribution:
    """Optuna distribution for one search dimension."""
    if isinstance(dim, FloatRange):
        return FloatDistribution(float(dim.low), float(dim.high), log=dim.log)
    if isinstance(dim, IntRange):
        return IntDistribution(
            int(dim.low), int(dim.high), log=dim.log, step=int(dim.step)
        )
    return CategoricalDistribution(dim)


def to_distributions(
    space: Mapping[str, SearchDimension],
) -> dict[str, BaseDistribution]:
    return {name: to_distribution(dim) for name, dim in space.items()}


def pool_size(
    n_trials: int,
    with_baseline: bool,
    parallel: bool,
    max_workers: int | None,
    cpu_count: int,
) -> int:
    """Worker processes for a batched search, always at least 1.

    ``max_workers`` is read as in grid search: ``None`` or 0 picks the
    default, a negative value means serial. The default is the CPU count,
    capped at ``n_trials // MIN_ROUNDS``: a batch as large as the whole
    search would leave the sampler no scores to learn from, which is random
    search.
    """
    if not parallel:
        return 1
    default = min(cpu_count, max(1, n_trials // MIN_ROUNDS))
    return max(1, min(n_trials + with_baseline, max_workers or default))


# ---------------------------------------------------------------------------
# Shell: study, process pool and trial loops
# ---------------------------------------------------------------------------


class FoldMedianPruner(optuna.pruners.BasePruner):
    """Stop a trial whose pooled score is worse than the median at this fold.

    Optuna's ``MedianPruner`` compares the best value a trial has reported
    at any step, which suits learning curves that improve step by step.
    Pooled cross-validation scores do not improve with folds: the latest one
    is the trial's best estimate so far, so that is what is compared, against
    the median of completed trials at the same fold. The study minimises, so
    higher is worse. A NaN score is pruned, as it can never win.
    """

    def __init__(self, n_startup_trials: int, n_warmup_steps: int) -> None:
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        step = trial.last_step
        if step is None or step < self._n_warmup_steps:
            return False
        current = trial.intermediate_values[step]
        peers = [
            t.intermediate_values[step]
            for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if step in t.intermediate_values
            and not math.isnan(t.intermediate_values[step])
        ]
        if len(peers) < self._n_startup_trials:
            return False
        return math.isnan(current) or current > statistics.median(peers)


@contextlib.contextmanager
def _quiet_optuna() -> Iterator[None]:
    """Hide Optuna's per-trial INFO lines; ``tune`` logs its own progress."""
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(max(previous, optuna.logging.WARNING))
    try:
        yield
    finally:
        optuna.logging.set_verbosity(previous)


@contextlib.contextmanager
def _runner(workers: int) -> Iterator[Runner]:
    """Score tasks in-process, or in one pool reused by every batch."""
    if workers <= 1:
        yield lambda tasks: [_score_params(t) for t in tasks]
        return
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=_mp_context()
    ) as pool:
        yield lambda tasks: list(pool.map(_score_params, tasks))


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _tell(
    study: optuna.Study, trial: optuna.Trial, outcome: Trial, value: float
) -> None:
    if outcome.pruned:
        study.tell(trial, state=TrialState.PRUNED)
    elif math.isnan(value):
        # Telling NaN as a value would make Optuna warn and fail the trial.
        study.tell(trial, state=TrialState.FAIL)
    else:
        study.tell(trial, value)


def _add_baseline(
    study: optuna.Study,
    distributions: Mapping[str, BaseDistribution],
    space: Mapping[str, SearchDimension],
    outcome: Trial,
    value: float,
    steps: Mapping[int, float],
) -> Trial:
    """Record the scored baseline in the study, if it lies inside the space.

    The sampler then learns from it like any other trial. A baseline outside
    the space stays in ``scores`` but not in the study, which would reject it.
    """
    if not space_contains(space, outcome.params):
        logger.info(
            "Prophet's defaults %s lie outside the search space: scored, "
            "but not added to the Optuna study.",
            outcome.params,
        )
        return outcome
    finite = not math.isnan(value)
    study.add_trial(
        create_trial(
            state=TrialState.COMPLETE if finite else TrialState.FAIL,
            value=value if finite else None,
            params=dict(outcome.params),
            distributions={name: distributions[name] for name in outcome.params},
            intermediate_values=dict(steps),
            user_attrs={"is_default": True},
        )
    )
    number = study.get_trials(deepcopy=False)[-1].number
    return dataclasses.replace(outcome, number=number)


def _run_batched(
    study: optuna.Study,
    distributions: dict[str, BaseDistribution],
    space: Mapping[str, SearchDimension],
    make_task: Callable[[dict[str, Any]], _Task],
    run: Runner,
    *,
    workers: int,
    n_trials: int,
    deadline: float | None,
    baseline: dict[str, Any] | None,
    score: Score,
) -> list[Trial]:
    """Ask a batch, score it, tell every result; repeat until done.

    The baseline shares the first batch so it costs no extra round. At least
    one Optuna trial always runs, even if ``timeout`` is already over.
    """
    results: list[Trial] = []
    pending = baseline
    asked = 0
    while asked < n_trials and (asked == 0 or not _expired(deadline)):
        extra = [] if pending is None else [pending]
        size = min(max(workers - len(extra), 0), n_trials - asked)
        batch = [study.ask(distributions) for _ in range(size)]
        asked += size
        outcomes = run(
            [make_task(params) for params in [*extra, *(t.params for t in batch)]]
        )
        if pending is not None:
            base = dataclasses.replace(outcomes[0], is_default=True)
            results.append(
                _add_baseline(
                    study, distributions, space, base, score(base.metrics), {}
                )
            )
            outcomes = outcomes[1:]
            pending = None
        for trial, outcome in zip(batch, outcomes, strict=True):
            _tell(study, trial, outcome, score(outcome.metrics))
            results.append(dataclasses.replace(outcome, number=trial.number))
        logger.info("Optuna search: %d of %d trials done.", asked, n_trials)
    return results


def _run_pruned(
    study: optuna.Study,
    distributions: dict[str, BaseDistribution],
    space: Mapping[str, SearchDimension],
    make_task: Callable[[dict[str, Any]], _Task],
    *,
    n_trials: int,
    deadline: float | None,
    baseline: dict[str, Any] | None,
    score: Score,
) -> list[Trial]:
    """Score trials one at a time, reporting each fold to the pruner."""
    results: list[Trial] = []
    if baseline is not None:
        steps: dict[int, float] = {}

        def record(step: int, metrics: Mapping[str, float]) -> bool:
            steps[step] = score(metrics)
            return False

        base = dataclasses.replace(
            _score_params_by_fold(make_task(baseline), record), is_default=True
        )
        results.append(
            _add_baseline(study, distributions, space, base, score(base.metrics), steps)
        )
    for asked in range(n_trials):
        if asked > 0 and _expired(deadline):
            break
        trial = study.ask(distributions)

        def should_stop(
            step: int, metrics: Mapping[str, float], trial: optuna.Trial = trial
        ) -> bool:
            trial.report(score(metrics), step)
            return trial.should_prune()

        outcome = _score_params_by_fold(make_task(dict(trial.params)), should_stop)
        _tell(study, trial, outcome, score(outcome.metrics))
        results.append(dataclasses.replace(outcome, number=trial.number))
    pruned = sum(t.pruned for t in results)
    logger.info("Optuna search: %d trials done, %d pruned.", len(results), pruned)
    return results


def run_study(
    make_task: Callable[[dict[str, Any]], _Task],
    space: Mapping[str, SearchDimension],
    *,
    n_trials: int,
    timeout: float | None,
    seed: int,
    parallel: bool,
    max_workers: int | None,
    prune: bool,
    n_folds: int,
    baseline: dict[str, Any] | None,
    score: Score,
) -> tuple[list[Trial], optuna.Study]:
    """Run an Optuna study over ``space`` and return every scored trial.

    ``make_task`` turns a parameter dict into a scoring task and ``score``
    turns metrics into the value to minimise. The baseline, when given, is
    scored as one extra evaluation and flagged ``is_default``.
    """
    distributions = to_distributions(space)
    size = space_size(space)
    if size is not None and n_trials >= size:
        logger.warning(
            "The search space has only %d combinations but n_trials=%d; "
            "search='grid' would score each exactly once.",
            size,
            n_trials,
        )
    # Folds after the warm-up and before the last are the only pruning points.
    if prune and n_folds < PRUNER_WARMUP_FOLDS + 2:
        logger.warning(
            "prune=True has no effect with %d fold(s): pruning needs at least "
            "%d, because the last fold is never a pruning point.",
            n_folds,
            PRUNER_WARMUP_FOLDS + 2,
        )
    deadline = None if timeout is None else time.monotonic() + timeout
    with _quiet_optuna():
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=FoldMedianPruner(
                n_startup_trials=PRUNER_STARTUP_TRIALS,
                n_warmup_steps=PRUNER_WARMUP_FOLDS,
            )
            if prune
            else optuna.pruners.NopPruner(),
        )
        if prune:
            logger.info("Optuna search: up to %d trials with pruning.", n_trials)
            trials = _run_pruned(
                study,
                distributions,
                space,
                make_task,
                n_trials=n_trials,
                deadline=deadline,
                baseline=baseline,
                score=score,
            )
        else:
            workers = pool_size(
                n_trials,
                baseline is not None,
                parallel,
                max_workers,
                os.cpu_count() or 1,
            )
            logger.info(
                "Optuna search: %d trials on %d worker process(es).", n_trials, workers
            )
            with _runner(workers) as run:
                trials = _run_batched(
                    study,
                    distributions,
                    space,
                    make_task,
                    run,
                    workers=workers,
                    n_trials=n_trials,
                    deadline=deadline,
                    baseline=baseline,
                    score=score,
                )
    return trials, study
