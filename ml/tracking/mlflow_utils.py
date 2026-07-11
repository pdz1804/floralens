"""Safe MLflow experiment-tracking wrapper (task requirement: "never let MLflow
being down break a run").

Every training/eval script logs to the SAME local, file/sqlite-backed store
(`infra/mlflow/mlflow.db` + `infra/mlflow/mlartifacts/`) regardless of whether
the `mlflow` docker-compose UI service is running — the tracking URI points
directly at the sqlite file, not at an HTTP server, so logging works with the
UI container down. `mlflow_run()` additionally wraps every MLflow call so
that if the `mlflow` package itself is missing, misconfigured, or the store
is locked/unwritable, the calling script logs a warning and continues with a
no-op tracker instead of crashing a real training/eval run.

One shared experiment name ("floralens"); one run per script invocation
(one run per training, matching the task's "one run per training").
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "floralens"
# Repo-root-relative sqlite store + artifact dir, mirrored by the `mlflow`
# service in infra/docker-compose.yml (mounted at the same infra/mlflow path)
# so the UI and direct-from-script logging always see the same runs.
DEFAULT_TRACKING_URI = "sqlite:///infra/mlflow/mlflow.db"
DEFAULT_ARTIFACT_ROOT = "./infra/mlflow/mlartifacts"


def tracking_uri() -> str:
    """Resolve the MLflow tracking URI: `MLFLOW_TRACKING_URI` env var if set, else the local sqlite default."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _ensure_local_store_dirs(uri: str) -> None:
    """sqlite can't create a db file in a directory that doesn't exist yet;
    docker-compose creates the mounted `infra/mlflow/` dir automatically, but
    a training/eval script run directly (no `docker compose up`) needs it
    created on first use."""
    if not uri.startswith("sqlite:///"):
        return
    db_path = Path(uri[len("sqlite:///") :])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    Path(DEFAULT_ARTIFACT_ROOT).mkdir(parents=True, exist_ok=True)


def _get_or_create_experiment(mlflow_module: Any) -> None:
    """Point the "floralens" experiment at our own artifact dir explicitly.

    Without an explicit `artifact_location`, MLflow's client-side default on
    first creation is `./mlruns/<experiment_id>` *relative to the process's
    CWD* — not `DEFAULT_ARTIFACT_ROOT` — so artifacts silently land outside
    the store this module documents. Creating the experiment ourselves (only
    on first-ever run) pins the artifact root explicitly; `set_experiment`
    is then a no-op lookup for every subsequent run.
    """
    existing = mlflow_module.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is None:
        artifact_root = Path(DEFAULT_ARTIFACT_ROOT).resolve().as_uri()
        mlflow_module.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_root)
    mlflow_module.set_experiment(EXPERIMENT_NAME)


class _NoopMlflow:
    """Swallows every call; used when MLflow itself cannot be initialized."""

    def __getattr__(self, _name: str):
        """Return a no-op callable for any attribute access, so every MLflow call silently does nothing."""

        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop


def _sanitize_metric_key(name: str) -> str:
    """MLflow metric/param keys only allow alphanumerics, `_-. /` — our
    retrieval metric names use `@` (e.g. "recall@5"), so rewrite it to
    "_at_" rather than silently dropping the metric (this module's whole
    point is that a store hiccup shouldn't lose a metric; an invalid name is
    the same failure mode and equally worth avoiding)."""
    return name.replace("@", "_at_")


class _SafeMlflow:
    """Wraps a real mlflow module so any single logging call failure (locked
    db, disk full, artifact-store hiccup, invalid metric name) logs a
    warning instead of raising and aborting the run in progress."""

    _SANITIZE_KEYS_FOR = {"log_metric", "log_metrics", "log_param", "log_params"}

    def __init__(self, module: Any) -> None:
        """Wrap `module` (a real mlflow module or a `_NoopMlflow`) so its calls fail safe."""
        self._module = module

    def __getattr__(self, name: str):
        """Return a fail-safe, key-sanitizing version of `module`'s attribute `name`.

        Non-callable attributes pass through unchanged. Callable attributes
        are wrapped so metric/param keys are sanitized (see
        `_sanitize_metric_key`) and any exception raised by the underlying
        MLflow call is caught, logged as a warning, and swallowed (returns
        `None`) instead of aborting the run in progress.
        """
        attr = getattr(self._module, name)
        if not callable(attr):
            return attr

        def _safe_call(*args: Any, **kwargs: Any) -> Any:
            call_args = args
            if name in self._SANITIZE_KEYS_FOR and args:
                first = args[0]
                if name in ("log_metrics", "log_params") and isinstance(first, dict):
                    first = {_sanitize_metric_key(k): v for k, v in first.items()}
                elif name in ("log_metric", "log_param") and isinstance(first, str):
                    first = _sanitize_metric_key(first)
                call_args = (first, *args[1:])
            try:
                return attr(*call_args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive, store/UI issues
                logger.warning("MLflow call %s(...) failed: %s", name, exc)
                return None

        return _safe_call


@contextmanager
def mlflow_run(
    run_name: str, tags: dict[str, str] | None = None, params: dict[str, Any] | None = None
) -> Iterator[Any]:
    """Start (or no-op) one MLflow run for the duration of the `with` block.

    Usage:
        with mlflow_run("baseline_eval", params={"backbone": ...}) as mlf:
            mlf.log_metric("test_recall_at_5", 0.95)
            mlf.log_artifact("ml/eval/reports/baseline_eval_report.json")
    """
    module: Any
    started = False
    try:
        import mlflow as _mlflow

        uri = tracking_uri()
        _ensure_local_store_dirs(uri)
        _mlflow.set_tracking_uri(uri)
        _get_or_create_experiment(_mlflow)
        _mlflow.start_run(run_name=run_name, tags=tags or {})
        module = _mlflow
        started = True
    except Exception as exc:
        logger.warning(
            "MLflow unavailable (%s); run '%s' proceeds without experiment tracking.",
            exc,
            run_name,
        )
        module = _NoopMlflow()

    safe = _SafeMlflow(module)
    try:
        if params:
            safe.log_params(params)
        yield safe
    finally:
        if started:
            try:
                module.end_run()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("MLflow end_run failed: %s", exc)
