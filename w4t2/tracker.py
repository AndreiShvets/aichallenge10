"""Mock-трекер задач: данные и операции, которые MCP-сервер выставляет агенту.

Данные зашиты в код, а не читаются из файла, чтобы результаты вызовов были
детерминированными: один и тот же запрос всегда возвращает один и тот же
ответ. Создание задачи меняет состояние только в памяти процесса сервера.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUSES = ("open", "in_progress", "done")
PRIORITIES = ("low", "normal", "high")


class TrackerError(ValueError):
    """Ошибка запроса к трекеру: неизвестный ключ, статус или пустое название."""


@dataclass(frozen=True)
class Task:
    key: str
    summary: str
    status: str
    assignee: str
    priority: str


SEED_TASKS = (
    Task("TR-1", "Настроить авторизацию в сервисе отчётов", "done", "Иванова", "high"),
    Task("TR-2", "Исправить расчёт остатков при округлении", "in_progress", "Петров", "high"),
    Task("TR-3", "Обновить README для интеграции с агентом", "open", "Сидорова", "normal"),
    Task("TR-4", "Добавить экспорт отчёта в CSV", "open", "Петров", "low"),
    Task("TR-5", "Перенести журнал вызовов в общий лог", "in_progress", "Иванова", "normal"),
)


class Tracker:
    """Трекер в памяти: список задач и счётчик для новых ключей."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {task.key: task for task in SEED_TASKS}
        self._next_number = len(SEED_TASKS) + 1

    def list_tasks(self, status: str | None = None) -> list[Task]:
        if status is not None and status not in STATUSES:
            raise TrackerError(
                f"Неизвестный статус {status!r}. Доступны: {', '.join(STATUSES)}"
            )
        tasks = sorted(self._tasks.values(), key=lambda task: task.key)
        if status is None:
            return tasks
        return [task for task in tasks if task.status == status]

    def get_task(self, key: str) -> Task:
        task = self._tasks.get(key)
        if task is None:
            raise TrackerError(f"Задача {key} не найдена")
        return task

    def create_task(self, summary: str, assignee: str = "", priority: str = "normal") -> Task:
        summary = summary.strip()
        if not summary:
            raise TrackerError("Название задачи не может быть пустым")
        if priority not in PRIORITIES:
            raise TrackerError(
                f"Неизвестный приоритет {priority!r}. Доступны: {', '.join(PRIORITIES)}"
            )
        key = f"TR-{self._next_number}"
        self._next_number += 1
        task = Task(key, summary, "open", assignee.strip(), priority)
        self._tasks[key] = task
        return task

