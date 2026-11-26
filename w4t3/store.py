"""Журнал запусков планировщика: накопленные результаты в одном JSON-файле.

Запись несёт время, вид запуска и данные инструмента как есть. Журнал только
дописывается, поэтому отчёт за период видит все запуски, а не последний.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

RUN_KINDS = ("snapshot", "reminder")
# Секунды достаточно: запуски планировщика идут не чаще раза в секунду.
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_LENGTH = len("YYYY-MM-DD")


class StoreError(ValueError):
    """Ошибка журнала: неизвестный вид запуска, битый файл, неверный период."""


@dataclass(frozen=True)
class Run:
    id: int
    at: str
    kind: str
    data: dict[str, Any]


class RunStore:
    """Журнал в JSON-файле. Блокировка защищает от гонки между HTTP-потоками."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def save(self, kind: str, data: Any, now: datetime | None = None) -> Run:
        if kind not in RUN_KINDS:
            raise StoreError(f"Неизвестный вид запуска {kind!r}. Доступны: {', '.join(RUN_KINDS)}")
        if not isinstance(data, dict):
            raise StoreError("Данные запуска должны быть объектом JSON")
        moment = (now or datetime.now()).strftime(TIME_FORMAT)
        with self._lock:
            runs = self._read_all()
            run = Run(len(runs) + 1, moment, kind, data)
            runs.append(run)
            self._write_all(runs)
        return run

    def read(self, start: datetime, end: datetime, kind: str | None = None) -> list[Run]:
        """Записи с start <= время < end, при необходимости одного вида."""
        if kind is not None and kind not in RUN_KINDS:
            raise StoreError(f"Неизвестный вид запуска {kind!r}. Доступны: {', '.join(RUN_KINDS)}")
        with self._lock:
            runs = self._read_all()
        return [
            run for run in runs
            if start <= datetime.strptime(run.at, TIME_FORMAT) < end
            and (kind is None or run.kind == kind)
        ]

    def _read_all(self) -> list[Run]:
        if not self._path.is_file():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("корень файла — не список")
            return [
                Run(int(item["id"]), str(item["at"]), str(item["kind"]), dict(item["data"]))
                for item in raw
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoreError(f"Журнал {self._path} не читается: {exc}") from exc

    def _write_all(self, runs: list[Run]) -> None:
        # Запись во временный файл и замена: обрыв посреди записи не портит журнал.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(
            json.dumps([asdict(run) for run in runs], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self._path)


def parse_period(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    """Период [start, end): даты YYYY-MM-DD или моменты YYYY-MM-DDTHH:MM:SS.

    Конец, заданный датой, включает весь этот день: to=2026-11-27 захватывает 27-е.
    Без границ период открыт: весь журнал.
    """
    lower = _parse_bound(start) if start else datetime.min
    upper_given = _parse_bound(end) if end else datetime.max
    # Сравниваем введённые значения до расширения даты конца до конца дня.
    if lower > upper_given:
        raise StoreError("Начало периода позже конца")
    upper = upper_given + timedelta(days=1) if end and len(end) == DATE_LENGTH else upper_given
    return lower, upper


def _parse_bound(text: str) -> datetime:
    try:
        if len(text) == DATE_LENGTH:
            return datetime.strptime(text, "%Y-%m-%d")
        return datetime.strptime(text, TIME_FORMAT)
    except ValueError as exc:
        raise StoreError(
            f"Неверный формат даты {text!r}: нужны YYYY-MM-DD или YYYY-MM-DDTHH:MM:SS"
        ) from exc
