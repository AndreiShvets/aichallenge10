from __future__ import annotations

import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class MemoryDialogTest(unittest.TestCase):
    def test_updates_context_and_done(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = iter([
                "/remember имя=Марина long",
                "Срок: 1 мая 2026",
                "Статус: согласование",
                "Версия: 1.0",
                "Подход: локальная обработка",
                "Срок перенесли с 1 мая 2026 на 15 мая 2026",
                "Какой сейчас срок?",
                "Статус теперь выполнение",
                "Какой сейчас статус?",
                "Версия теперь 2.0",
                "Какая сейчас версия?",
                "Подход теперь обработка на сервере",
                "Какой сейчас подход?",
                "/state go execution confirm",
                "/state go validation",
                "/state go done confirm",
                "/exit",
            ])
            expected = {
                "срок": "15 мая 2026",
                "статус": "выполнение",
                "версия": "2.0",
                "подход": "обработка на сервере",
            }
            stale = {
                "срок": "1 мая 2026",
                "статус": "согласование",
                "версия": "1.0",
                "подход": "локальная обработка",
            }
            checked: list[tuple[str, str]] = []

            def chat_response(client, model, messages, body, query_number, step_number):
                query = messages[-1]["content"]
                answer = "Принято."
                if query.endswith("?"):
                    key = next(key for key in expected if key in query.lower())
                    actual = "\n".join(str(message.get("content", "")) for message in messages)
                    keys = list(expected)
                    for changed_key in keys[:keys.index(key) + 1]:
                        self.assertNotIn(stale[changed_key], actual)
                    match = re.search(r"(?m)^- " + re.escape(key) + r": (.+)$", actual)
                    self.assertIsNotNone(match, actual)
                    answer = match.group(1)
                    checked.append((key, answer))
                return {"choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

            with (patch.object(main, "PROFILE_DIR", root / "profiles"),
                  patch.object(main, "OUTPUT_DIR", root),
                  patch.object(main, "load_env_file"),
                  patch.object(main, "GigaChatClient") as client,
                  patch.object(main, "print_gigachat_balance"),
                  patch.object(main, "_chat_response", side_effect=chat_response),
                  patch.object(main, "ask_command", side_effect=lambda *args: next(commands)),
                  patch.dict("os.environ", {"GIGACHAT_CREDENTIALS": "test"}),
                  patch("sys.argv", ["main.py", "--env", str(root / "unused.env"),
                                      "--no-extract", "--no-compress"]),
                  contextlib.redirect_stdout(io.StringIO()),
                  contextlib.redirect_stderr(io.StringIO())):
                client.return_value.get_access_token.return_value = "token"
                result = main.main()

            self.assertEqual(result, 0)
            self.assertEqual(dict(checked), expected)
            self.assertEqual(len(checked), 4)
            memory = main.load_memory(root / "profiles/default/work.json",
                                      root / "profiles/default/long.json")
            self.assertEqual(memory.state.stage, "done")
            self.assertFalse(memory.working.facts)
            self.assertEqual(memory.long.facts["имя"].value, "Марина")


if __name__ == "__main__":
    unittest.main()
