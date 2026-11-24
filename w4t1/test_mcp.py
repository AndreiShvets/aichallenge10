"""Проверки w4t1 без платных запросов: заглушка MCP-сервера и клиент модели.

Заглушка сервера — настоящий процесс по stdio JSON-RPC, поэтому протокол
mcp_client проверяется тем же способом, каким его использует filesystem-сервер.
Клиент GigaChat подменяется скриптовым классом: он отдаёт заранее заготовленные
ответы и сохраняет тела запросов — по ним видно, что схемы обоих источников
ушли модели в одном массиве functions.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from main import AppError, History, McpConnection, run_agent
from mcp_client import McpError, McpStdioClient

# Заглушка MCP-сервера: минимальный диалект протокола — initialize, tools/list,
# tools/call; уведомления молча пропускаются, неизвестные методы — ошибка JSON-RPC.
FAKE_SERVER_SCRIPT = """
import json, sys

TOOLS = [
    {"name": "echo_upper", "description": "Возвращает текст прописными.",
     "inputSchema": {"type": "object",
                     "properties": {"text": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "boom", "description": "Всегда ошибается.",
     "inputSchema": {"type": "object", "properties": {}}},
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05",
                  "serverInfo": {"name": "fake-mcp", "version": "0.1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        if params.get("name") == "echo_upper":
            result = {"content": [{"type": "text",
                                   "text": params["arguments"]["text"].upper()}],
                      "isError": False}
        else:
            result = {"content": [{"type": "text", "text": "всегда ошибка"}],
                      "isError": True}
    else:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                     "error": {"code": -32601,
                                               "message": "Method not found"}}) + "\\n")
        sys.stdout.flush()
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                                 "result": result}) + "\\n")
    sys.stdout.flush()
"""


class FakeChatClient:
    """Скриптовый клиент модели: отдаёт заготовки, копит тела запросов."""

    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)
        self.bodies: list[dict] = []
        self.log_context: dict = {}

    def chat_messages(self, token: str | None, body: dict) -> dict:
        self.bodies.append(json.loads(json.dumps(body)))
        if not self._replies:
            raise AppError("Заготовленные ответы кончились")
        return self._replies.pop(0)


def function_call_reply(name: str, arguments: dict) -> dict:
    return {
        "choices": [{"message": {"function_call": {"name": name,
                                                   "arguments": arguments}}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def text_reply(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class McpProtocolTest(unittest.TestCase):
    """Рукопожатие, список инструментов и вызов через живой процесс-заглушку."""

    def setUp(self) -> None:
        self.client = McpStdioClient([sys.executable, "-c", FAKE_SERVER_SCRIPT])
        self.info = self.client.start()
        self.tools = self.client.list_tools()

    def tearDown(self) -> None:
        self.client.close()

    def test_initialize_reports_server(self) -> None:
        self.assertEqual(self.info.name, "fake-mcp")
        self.assertEqual(self.info.version, "0.1.0")

    def test_tools_listed_with_schemas(self) -> None:
        self.assertEqual([tool.name for tool in self.tools], ["echo_upper", "boom"])
        schema = self.tools[0].schema()
        self.assertEqual(schema["name"], "echo_upper")
        self.assertEqual(schema["parameters"]["required"], ["text"])

    def test_call_returns_text_block(self) -> None:
        self.assertEqual(self.client.call_tool("echo_upper", {"text": "готово"}),
                         "ГОТОВО")

    def test_tool_error_raises(self) -> None:
        with self.assertRaises(McpError):
            self.client.call_tool("boom", {})


class MixedAgentTest(unittest.TestCase):
    """Смешанный набор: схемы обоих источников в запросе, вызовы — по веткам."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._mcp = McpStdioClient([sys.executable, "-c", FAKE_SERVER_SCRIPT])
        self.info = self._mcp.start()
        self.addCleanup(self._mcp.close)
        self.connection = McpConnection(self._mcp, self.info, self._mcp.list_tools())

    def test_schemas_merge_both_sources(self) -> None:
        names = [schema["name"] for schema in self.connection.schemas()]
        self.assertEqual(names, ["decimal_to_hex", "echo_upper", "boom"])

    def test_agent_uses_local_and_mcp_tools(self) -> None:
        chat = FakeChatClient([
            function_call_reply("decimal_to_hex", {"number": 255}),
            function_call_reply("echo_upper", {"text": "готово"}),
            text_reply("255 = 0xff, ГОТОВО"),
        ])
        run = run_agent(
            "Переведи 255 и прокричи готово", chat, "GigaChat",
            Path(self.tmp.name) / "tool_calls.log", History("старт"),
            self.connection, context=(1, "Переведи 255 и прокричи готово"),
        )
        self.assertEqual([step.source for step in run.steps], ["local", "mcp"])
        self.assertEqual(run.steps[0].result, "0xff")
        self.assertEqual(run.steps[1].result, "ГОТОВО")
        self.assertEqual(run.answer, "255 = 0xff, ГОТОВО")
        # Тело первого запроса: один массив functions из схем обоих источников.
        sent_functions = chat.bodies[0]["functions"]
        self.assertEqual([schema["name"] for schema in sent_functions],
                         ["decimal_to_hex", "echo_upper", "boom"])

    def test_unknown_tool_does_not_crash_run(self) -> None:
        chat = FakeChatClient([
            function_call_reply("no_such_tool", {}),
            text_reply("обойдусь"),
        ])
        run = run_agent(
            "вызови неизвестное", chat, "GigaChat",
            Path(self.tmp.name) / "tool_calls.log", History("старт"),
            self.connection, context=(1, "вызови неизвестное"),
        )
        self.assertIn("не найден", run.steps[0].error)
        self.assertEqual(run.answer, "обойдусь")

    def test_mcp_tool_error_reaches_model(self) -> None:
        chat = FakeChatClient([
            function_call_reply("boom", {}),
            text_reply("тогда без него"),
        ])
        run = run_agent(
            "сломай инструмент", chat, "GigaChat",
            Path(self.tmp.name) / "tool_calls.log", History("старт"),
            self.connection, context=(1, "сломай инструмент"),
        )
        self.assertTrue(run.steps[0].error.startswith("MCP:"))
        self.assertEqual(run.answer, "тогда без него")


if __name__ == "__main__":
    unittest.main()
