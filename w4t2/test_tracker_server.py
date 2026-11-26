"""Проверки w4t2 без модели и без сети: трекер, обработчики MCP и HTTP-клиент.

HTTP-клиент проверяется против сервера, поднятого в том же процессе на
свободном порту, — так видно и протокол, и транспорт, без внешних зависимостей.
"""

from __future__ import annotations

import threading
import unittest

from mcp_http_client import McpError, McpHttpClient
from server import handle_message, serve
from tracker import TrackerError, Tracker


def rpc(tracker: Tracker, method: str, params: dict | None = None, request_id: int = 1) -> dict:
    response = handle_message(tracker, {"jsonrpc": "2.0", "id": request_id,
                                        "method": method, "params": params or {}})
    assert response is not None
    return response


class TrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = Tracker()

    def test_list_filters_by_status(self) -> None:
        keys = [task.key for task in self.tracker.list_tasks("open")]
        self.assertEqual(keys, ["TR-3", "TR-4"])

    def test_unknown_status_is_error(self) -> None:
        with self.assertRaises(TrackerError):
            self.tracker.list_tasks("closed")

    def test_create_assigns_next_key_and_defaults(self) -> None:
        task = self.tracker.create_task("Новая задача")
        self.assertEqual((task.key, task.status, task.priority), ("TR-6", "open", "normal"))

    def test_create_rejects_empty_summary_and_bad_priority(self) -> None:
        with self.assertRaises(TrackerError):
            self.tracker.create_task("   ")
        with self.assertRaises(TrackerError):
            self.tracker.create_task("Задача", priority="urgent")

    def test_get_missing_key_is_error(self) -> None:
        with self.assertRaises(TrackerError):
            self.tracker.get_task("TR-404")


class ServerProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = Tracker()

    def test_initialize_and_tools_list(self) -> None:
        info = rpc(self.tracker, "initialize")["result"]["serverInfo"]
        self.assertEqual(info["name"], "mock-tracker")
        names = [tool["name"] for tool in rpc(self.tracker, "tools/list")["result"]["tools"]]
        self.assertEqual(names, ["list_tasks", "get_task", "create_task"])

    def test_tools_list_exposes_schema_without_handler(self) -> None:
        tools = rpc(self.tracker, "tools/list")["result"]["tools"]
        create = next(tool for tool in tools if tool["name"] == "create_task")
        self.assertEqual(create["inputSchema"]["required"], ["summary"])
        self.assertNotIn("handler", create)

    def test_tool_call_returns_text_content(self) -> None:
        result = rpc(self.tracker, "tools/call",
                     {"name": "get_task", "arguments": {"key": "TR-2"}})["result"]
        self.assertFalse(result["isError"])
        self.assertIn('"assignee": "Петров"', result["content"][0]["text"])

    def test_domain_error_is_tool_result_not_rpc_error(self) -> None:
        response = rpc(self.tracker, "tools/call",
                       {"name": "get_task", "arguments": {"key": "TR-404"}})
        self.assertNotIn("error", response)
        self.assertTrue(response["result"]["isError"])

    def test_unknown_tool_and_method_are_rpc_errors(self) -> None:
        self.assertEqual(rpc(self.tracker, "tools/call", {"name": "nope"})["error"]["code"], -32602)
        self.assertEqual(rpc(self.tracker, "resources/list")["error"]["code"], -32601)

    def test_notification_gets_no_response(self) -> None:
        self.assertIsNone(handle_message(self.tracker, {"jsonrpc": "2.0",
                                                        "method": "notifications/initialized"}))


class HttpClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd = serve("127.0.0.1", 0, Tracker())
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self._stop)
        port = self.httpd.server_address[1]
        self.client = McpHttpClient(f"http://127.0.0.1:{port}/mcp")

    def _stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_handshake_and_tools(self) -> None:
        self.assertEqual(self.client.start().name, "mock-tracker")
        self.assertEqual([tool.name for tool in self.client.list_tools()],
                         ["list_tasks", "get_task", "create_task"])

    def test_call_and_create_round_trip(self) -> None:
        self.client.start()
        created = self.client.call_tool("create_task",
                                        {"summary": "Проверка", "priority": "high"})
        self.assertIn('"key": "TR-6"', created)
        self.assertIn('"priority": "high"', created)

    def test_tool_error_raises_mcp_error(self) -> None:
        self.client.start()
        with self.assertRaises(McpError):
            self.client.call_tool("get_task", {"key": "TR-404"})

    def test_unreachable_server_raises_mcp_error(self) -> None:
        self._stop()
        with self.assertRaises(McpError):
            self.client.start()


if __name__ == "__main__":
    unittest.main()
