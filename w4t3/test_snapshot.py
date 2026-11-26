"""Проверки w4t3 без модели и без сети: журнал, агрегация, инструменты сервера.

Журнал пишется во временный файл, инструменты вызываются через тот же
handle_message, что и по HTTP, поэтому проверяется протокол целиком.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from aggregate import aggregate
from gigachat_client import GigaChatError
from server import Services, handle_message
from store import Run, RunStore, StoreError, parse_period
from summary import SummaryError, build_prompt, summarize
from tracker import Tracker


def snapshot_run(run_id: int, at: str, statuses: tuple[str, str]) -> Run:
    tasks = [{"key": "TR-1", "status": statuses[0]}, {"key": "TR-2", "status": statuses[1]}]
    return Run(run_id, at, "snapshot", {"tasks": tasks})


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = RunStore(Path(self._dir.name) / "runs.json")

    def test_runs_accumulate_between_saves(self) -> None:
        self.store.save("snapshot", {"n": 1}, now=datetime(2026, 11, 27, 10, 0))
        second = self.store.save("snapshot", {"n": 2}, now=datetime(2026, 11, 27, 10, 1))
        self.assertEqual(second.id, 2)
        everything = self.store.read(datetime.min, datetime.max)
        self.assertEqual([run.data["n"] for run in everything], [1, 2])

    def test_read_filters_period_and_kind(self) -> None:
        self.store.save("snapshot", {}, now=datetime(2026, 11, 26, 23, 59))
        self.store.save("snapshot", {}, now=datetime(2026, 11, 27, 9, 0))
        self.store.save("reminder", {}, now=datetime(2026, 11, 27, 9, 5))
        start, end = parse_period("2026-11-27", "2026-11-27")
        runs = self.store.read(start, end, kind="snapshot")
        self.assertEqual([run.at for run in runs], ["2026-11-27T09:00:00"])

    def test_unknown_kind_and_non_object_data_are_rejected(self) -> None:
        with self.assertRaises(StoreError):
            self.store.save("weekly", {})
        with self.assertRaises(StoreError):
            self.store.save("snapshot", [1, 2])

    def test_broken_file_is_reported(self) -> None:
        (Path(self._dir.name) / "runs.json").write_text("{не json", encoding="utf-8")
        with self.assertRaises(StoreError):
            self.store.read(datetime.min, datetime.max)


class PeriodTest(unittest.TestCase):
    def test_end_date_includes_whole_day(self) -> None:
        _, end = parse_period(None, "2026-11-27")
        self.assertEqual(end, datetime(2026, 11, 28))

    def test_open_period_has_no_bounds(self) -> None:
        start, end = parse_period(None, None)
        self.assertEqual((start, end), (datetime.min, datetime.max))

    def test_bad_format_and_reversed_period_are_errors(self) -> None:
        with self.assertRaises(StoreError):
            parse_period("27.11.2026", None)
        with self.assertRaises(StoreError):
            parse_period("2026-11-28", "2026-11-27")


class AggregateTest(unittest.TestCase):
    def test_report_counts_every_snapshot_not_the_last(self) -> None:
        runs = [
            snapshot_run(1, "2026-11-27T10:00:00", ("open", "done")),
            snapshot_run(2, "2026-11-27T10:01:00", ("in_progress", "done")),
            snapshot_run(3, "2026-11-27T10:02:00", ("done", "open")),
        ]
        report = aggregate(runs)
        self.assertEqual(report.snapshots, 3)
        self.assertEqual(report.average_by_status["done"], 1.0)
        self.assertEqual(report.average_by_status["open"], 0.67)
        self.assertEqual(report.average_by_status["in_progress"], 0.33)

    def test_transitions_are_counted_between_neighbours(self) -> None:
        runs = [
            snapshot_run(1, "2026-11-27T10:00:00", ("open", "done")),
            snapshot_run(2, "2026-11-27T10:01:00", ("in_progress", "done")),
            snapshot_run(3, "2026-11-27T10:02:00", ("done", "open")),
        ]
        report = aggregate(runs)
        self.assertEqual(report.transitions["open→in_progress"], 1)
        self.assertEqual(report.transitions["in_progress→done"], 1)
        self.assertEqual(report.transitions["done→open"], 1)
        self.assertEqual(report.transitions["done→done"], 0)

    def test_reminders_are_counted_but_not_as_snapshots(self) -> None:
        runs = [
            snapshot_run(1, "2026-11-27T10:00:00", ("open", "done")),
            Run(2, "2026-11-27T10:00:30", "reminder", {"text": "открыто 2"}),
        ]
        report = aggregate(runs)
        self.assertEqual((report.runs_total, report.snapshots, report.reminders), (2, 1, 1))

    def test_empty_period_gives_zero_report(self) -> None:
        report = aggregate([])
        self.assertEqual(report.runs_total, 0)
        self.assertEqual(report.average_by_status["open"], 0.0)
        self.assertIsNone(report.first_at)


class FakeChat:
    """Заглушка клиента GigaChat: запоминает промпт и отвечает заданным образом."""

    def __init__(self, reply: dict | Exception) -> None:
        self.reply = reply
        self.prompt = ""

    def get_access_token(self) -> str:
        return "token"

    def chat(self, access_token: str, model: str, prompt: str, max_tokens: int | None = None) -> dict:
        self.prompt = prompt
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class SummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        runs = [snapshot_run(1, "2026-11-27T10:00:00", ("open", "done"))]
        self.report = aggregate(runs)
        self.lines = ["Снимков 1"]

    def test_prompt_carries_computed_numbers(self) -> None:
        self.assertIn("Всего записей: 1, снимков: 1.", build_prompt(self.report, self.lines))

    def test_summary_is_model_text(self) -> None:
        chat = FakeChat({"choices": [{"message": {"content": " Задач стало меньше. "}}]})
        self.assertEqual(summarize(chat, self.report, self.lines), "Задач стало меньше.")
        self.assertIn("Снимков 1", chat.prompt)

    def test_missing_choices_and_transport_errors_are_summary_errors(self) -> None:
        with self.assertRaises(SummaryError):
            summarize(FakeChat({"choices": []}), self.report, self.lines)
        with self.assertRaises(SummaryError):
            summarize(FakeChat(GigaChatError("503")), self.report, self.lines)


class ServerToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.services = Services(Tracker(), RunStore(Path(self._dir.name) / "runs.json"))

    def call(self, name: str, arguments: dict) -> tuple[bool, str]:
        response = handle_message(self.services, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        result = response["result"]
        return result["isError"], result["content"][0]["text"]

    def test_tools_are_registered_with_schemas(self) -> None:
        response = handle_message(self.services, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = [tool["name"] for tool in response["result"]["tools"]]
        self.assertEqual(names, ["fetch_snapshot", "save_run", "read_runs"])

    def test_each_fetch_changes_state_and_runs_accumulate(self) -> None:
        _, first = self.call("fetch_snapshot", {})
        _, second = self.call("fetch_snapshot", {})
        self.assertNotEqual(json.loads(first)["tasks"], json.loads(second)["tasks"])

        for payload in (first, second):
            is_error, _ = self.call("save_run", {"kind": "snapshot", "data": json.loads(payload)})
            self.assertFalse(is_error)
        _, stored = self.call("read_runs", {})
        self.assertEqual(len(json.loads(stored)), 2)

    def test_fetch_with_status_filter(self) -> None:
        _, text = self.call("fetch_snapshot", {"status": "open"})
        statuses = {task["status"] for task in json.loads(text)["tasks"]}
        self.assertTrue(statuses <= {"open"})

    def test_domain_errors_come_back_as_tool_results(self) -> None:
        is_error, text = self.call("save_run", {"kind": "weekly", "data": {}})
        self.assertTrue(is_error)
        self.assertIn("weekly", text)
        is_error, _ = self.call("read_runs", {"from": "27.11.2026"})
        self.assertTrue(is_error)


if __name__ == "__main__":
    unittest.main()
