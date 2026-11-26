"""Планировщик: фоновый цикл, который по расписанию дёргает инструменты MCP-сервера.

Каждый запуск: опрос трекера (fetch_snapshot), сохранение снимка (save_run) и
напоминание по открытым задачам из того же снимка (save_run, kind=reminder).
Каждые --report-every запусков планировщик читает журнал за время своей работы
(read_runs) и считает сводку по всем записям, а не по последней.

Запуск: python scheduler.py [--mcp-url URL] [--interval SEC] [--report-every N] [--max-runs N]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from aggregate import Report, aggregate
from mcp_http_client import McpError, McpHttpClient
from store import TIME_FORMAT, Run
from summary import SummaryError, build_summarizer

DEFAULT_MCP_URL = "http://127.0.0.1:8765/mcp"
DEFAULT_INTERVAL_SECONDS = 10.0
DEFAULT_REPORT_EVERY = 3


@dataclass(frozen=True)
class Options:
    mcp_url: str = DEFAULT_MCP_URL
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    report_every: int = DEFAULT_REPORT_EVERY
    # None — работать, пока не остановят Ctrl+C.
    max_runs: int | None = None
    # Текст сводки через GigaChat: платный шаг, поэтому по умолчанию выключен.
    with_summary: bool = False


def run_once(client: McpHttpClient) -> tuple[dict[str, Any], dict[str, Any]]:
    """Один запуск: снимок и напоминание. Возвращает их данные для печати."""
    snapshot = json.loads(client.call_tool("fetch_snapshot", {}))
    snapshot_run = json.loads(client.call_tool("save_run", {"kind": "snapshot", "data": snapshot}))
    # Напоминание берёт открытые задачи из того же снимка: второй опрос сдвинул бы трекер.
    open_tasks = [task for task in snapshot["tasks"] if task["status"] == "open"]
    reminder = {"tasks": open_tasks}
    client.call_tool("save_run", {"kind": "reminder", "data": reminder})
    return snapshot_run, reminder


def build_report(client: McpHttpClient, since: str) -> Report:
    """Сводка по всем записям журнала от момента since, а не по последнему запуску."""
    raw = json.loads(client.call_tool("read_runs", {"from": since}))
    runs = [Run(int(item["id"]), item["at"], item["kind"], item["data"]) for item in raw]
    return aggregate(runs)


def format_report(report: Report) -> list[str]:
    lines = [
        f"Сводка: записей {report.runs_total} "
        f"(снимков {report.snapshots}, напоминаний {report.reminders})",
        f"Период записей: {report.first_at} — {report.last_at}",
        "Среднее число задач за снимок: "
        + ", ".join(f"{status} {value}" for status, value in report.average_by_status.items()),
    ]
    if report.transitions:
        moves = ", ".join(f"{move} ×{count}" for move, count in sorted(report.transitions.items()))
        lines.append(f"Смены статусов между снимками: {moves}")
    else:
        lines.append("Смены статусов между снимками: нет")
    return lines


def run_schedule(
    client: McpHttpClient,
    options: Options,
    sleep: Callable[[float], None] = time.sleep,
    out: Callable[[str], None] = print,
    summarizer: Callable[[Report, list[str]], str] | None = None,
) -> int:
    """Цикл планировщика. Ошибка одного запуска не останавливает цикл."""
    since = datetime.now().strftime(TIME_FORMAT)
    run_number = 0
    while options.max_runs is None or run_number < options.max_runs:
        run_number += 1
        try:
            snapshot_run, reminder = run_once(client)
            out(f"[{datetime.now():%H:%M:%S}] запуск {run_number}: снимок №{snapshot_run['id']}, "
                f"открытых задач {len(reminder['tasks'])}")
        except (McpError, ValueError) as exc:
            out(f"[{datetime.now():%H:%M:%S}] запуск {run_number}: ошибка — {exc}")
        if run_number % options.report_every == 0:
            try:
                report = build_report(client, since)
                lines = format_report(report)
                for line in lines:
                    out(line)
                if summarizer is not None:
                    out(f"Текст: {summarizer(report, lines)}")
            except (McpError, ValueError, SummaryError) as exc:
                out(f"Отчёт не собран: {exc}")
        if options.max_runs is None or run_number < options.max_runs:
            sleep(options.interval_seconds)
    return 0


def parse_args(argv: list[str]) -> Options:
    """Разбор --flag=value и --flag value; неизвестный аргумент — ошибка со списком."""
    known = ("--mcp-url", "--interval", "--report-every", "--max-runs", "--summary")
    values: dict[str, str] = {}
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--summary":
            values[arg] = "1"
            continue
        name, sep, inline = arg.partition("=")
        if name not in known:
            raise ValueError(f"Неизвестный аргумент: {arg}. Доступны: {', '.join(known)}")
        if sep:
            values[name] = inline
        elif args:
            values[name] = args.pop(0)
        else:
            raise ValueError(f"Аргумент {name} требует значения")
    options = Options(
        mcp_url=values.get("--mcp-url", DEFAULT_MCP_URL),
        interval_seconds=float(values.get("--interval", DEFAULT_INTERVAL_SECONDS)),
        report_every=int(values.get("--report-every", DEFAULT_REPORT_EVERY)),
        max_runs=int(values["--max-runs"]) if "--max-runs" in values else None,
        with_summary="--summary" in values,
    )
    if options.interval_seconds <= 0 or options.report_every < 1:
        raise ValueError("--interval должен быть больше 0, --report-every — не меньше 1")
    return options


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        options = parse_args(argv)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    client = McpHttpClient(options.mcp_url)
    try:
        info = client.start()
    except McpError as exc:
        print(f"MCP-сервер недоступен: {exc}\nЗапустите: python server.py", file=sys.stderr)
        return 1
    print(f"Подключено к {info.name} {info.version}. Остановка — Ctrl+C.")
    try:
        summarizer = build_summarizer(Path(__file__).resolve().parent) if options.with_summary else None
        return run_schedule(client, options, summarizer=summarizer)
    except SummaryError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Планировщик остановлен.")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
