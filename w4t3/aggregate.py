"""Агрегация журнала запусков: сводка по всем записям периода, а не по последней.

Чистая логика без сети и файлов: получает записи и возвращает цифры. Текст
сводки для человека строит отдельный слой, цифры здесь считаются кодом.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from store import Run
from tracker import STATUSES


@dataclass(frozen=True)
class Report:
    runs_total: int
    snapshots: int
    reminders: int
    first_at: str | None
    last_at: str | None
    # Сколько задач в каждом статусе в среднем за снимок.
    average_by_status: dict[str, float]
    # Сколько раз каждая задача сменила статус между соседними снимками.
    transitions: Counter[str] = field(default_factory=Counter)
    # Ключи задач, которые встретились в снимках периода.
    task_keys: frozenset[str] = frozenset()


def aggregate(runs: list[Run]) -> Report:
    """Сводка по всем записям: снимки дают динамику, напоминания — только счёт."""
    ordered = sorted(runs, key=lambda run: (run.at, run.id))
    snapshots = [run for run in ordered if run.kind == "snapshot"]
    reminders = [run for run in ordered if run.kind == "reminder"]

    totals: Counter[str] = Counter()
    last_status: dict[str, str] = {}
    transitions: Counter[str] = Counter()
    keys: set[str] = set()
    for snapshot in snapshots:
        tasks = _tasks_of(snapshot)
        for task in tasks:
            totals[task["status"]] += 1
            keys.add(task["key"])
            previous = last_status.get(task["key"])
            if previous is not None and previous != task["status"]:
                transitions[f"{previous}→{task['status']}"] += 1
            last_status[task["key"]] = task["status"]

    count = len(snapshots)
    average = {
        status: round(totals[status] / count, 2) if count else 0.0
        for status in STATUSES
    }
    return Report(
        runs_total=len(ordered),
        snapshots=count,
        reminders=len(reminders),
        first_at=ordered[0].at if ordered else None,
        last_at=ordered[-1].at if ordered else None,
        average_by_status=average,
        transitions=transitions,
        task_keys=frozenset(keys),
    )


def _tasks_of(snapshot: Run) -> list[dict[str, Any]]:
    tasks = snapshot.data.get("tasks")
    return tasks if isinstance(tasks, list) else []
