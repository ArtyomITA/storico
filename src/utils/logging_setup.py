"""Centralised logging configuration with Rich console and JSON output."""
from __future__ import annotations

import functools
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar, Union, cast

try:
    import psutil
except ImportError:
    psutil = None

from rich.console import Console
from rich.logging import RichHandler

DEFAULT_LOG_DIR = Path("logs")
ENV_LEVEL_KEYS = ("PIPELINE_LOG_LEVEL", "LOGLEVEL", "LOG_LEVEL")
_JSON_FIELDS = ("stage", "cliente", "event_key", "duration_ms", "rows", "seed", "rss_mb")

F = TypeVar("F", bound=Callable[..., Any])

class JsonLinesFormatter(logging.Formatter):
    """Format log records as JSON lines capturing contextual metadata."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _JSON_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def _coerce_level(candidate: Union[str, int, None]) -> int:
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, str):
        return logging._nameToLevel.get(candidate.upper(), logging.INFO)
    return logging.INFO

_configured = False
_current_log_file: Optional[Path] = None

def configure_logging(
    level: Union[str, int, None] = None,
    *,
    log_dir: Optional[Path] = None,
    console: bool = True,
) -> Path:
    """Configure root logging with Rich console and JSON file handlers.

    Returns the path to the JSON log file used for the current process. Subsequent
    invocations reuse the same handlers while allowing the effective log level to be
    updated programmatically.
    """

    global _configured, _current_log_file

    env_level = next((os.getenv(key) for key in ENV_LEVEL_KEYS if os.getenv(key)), None)
    effective_level = _coerce_level(level if level is not None else env_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    if _configured and _current_log_file:
        for handler in root_logger.handlers:
            handler.setLevel(effective_level)
        return _current_log_file

    log_directory = (log_dir or DEFAULT_LOG_DIR).resolve()
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if console:
        console_handler = RichHandler(
            console=Console(),
            rich_tracebacks=True,
            markup=True,
            show_time=False,
            show_level=True,
            show_path=False,
        )
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonLinesFormatter())
    root_logger.addHandler(file_handler)

    _configured = True
    _current_log_file = log_path
    return log_path

def get_current_log_file() -> Optional[Path]:
    """Return the JSON log file configured for the current process, if any."""

    return _current_log_file

def _collect_meta(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    collected: Dict[str, Any] = {field: None for field in _JSON_FIELDS}
    for source in sources:
        for field in collected:
            if field in source and source[field] is not None:
                collected[field] = source[field]
    return collected

def _rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None

def log_timed(stage_name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator logging execution time (ms) and optional memory footprint."""

    def decorator(func: F) -> F:
        name = stage_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger(func.__module__)
            start = time.perf_counter()
            extra_from_kwargs = {k: kwargs.get(k) for k in ("cliente", "event_key", "rows", "seed") if k in kwargs}
            meta_start = _collect_meta(extra_from_kwargs)
            meta_start.update({"stage": name})
            logger.info("Stage %s started", name, extra=meta_start)
            try:
                result = func(*args, **kwargs)
            except Exception:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                meta = _collect_meta(extra_from_kwargs)
                meta.update({"stage": name, "duration_ms": duration_ms})
                rss_value = _rss_mb()
                if rss_value is not None:
                    meta["rss_mb"] = rss_value
                logger.exception("Stage %s failed", name, extra=meta)
                raise
            else:
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                extra_from_result: Mapping[str, Any]
                if isinstance(result, Mapping):
                    extra_from_result = {k: result.get(k) for k in ("cliente", "event_key", "rows", "seed")}
                else:
                    extra_from_result = {}
                meta = _collect_meta(extra_from_kwargs, extra_from_result)
                meta.update({"stage": name, "duration_ms": duration_ms})
                rss_value = _rss_mb()
                if rss_value is not None:
                    meta["rss_mb"] = rss_value
                logger.info("Stage %s completed", name, extra=meta)
                return result

        return cast(F, wrapper)

    return decorator

__all__ = ["configure_logging", "get_current_log_file", "log_timed"]

