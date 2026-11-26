"""MCP-сервер поверх HTTP: отдаёт инструменты mock-трекера агенту.

Протокол — JSON-RPC 2.0. Клиент шлёт каждое сообщение POST-запросом на
`/mcp`; ответ приходит JSON-телом, уведомления получают 202 без тела.
Сервер не хранит сессий: каждый запрос самодостаточен, а состояние трекера
живёт в процессе сервера.

Запуск: python server.py [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from tracker import PRIORITIES, STATUSES, Task, Tracker, TrackerError

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "mock-tracker", "version": "0.1.0"}
MCP_PATH = "/mcp"
MAX_BODY_BYTES = 1_000_000

# Коды ошибок JSON-RPC из спецификации.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def task_view(task: Task) -> dict[str, str]:
    return {
        "key": task.key,
        "summary": task.summary,
        "status": task.status,
        "assignee": task.assignee,
        "priority": task.priority,
    }


def list_tasks_tool(tracker: Tracker, arguments: dict[str, Any]) -> str:
    status = arguments.get("status")
    tasks = tracker.list_tasks(status)
    return json.dumps([task_view(task) for task in tasks], ensure_ascii=False)


def get_task_tool(tracker: Tracker, arguments: dict[str, Any]) -> str:
    key = arguments.get("key")
    if not isinstance(key, str) or not key:
        raise TrackerError("Параметр key обязателен и должен быть строкой")
    return json.dumps(task_view(tracker.get_task(key)), ensure_ascii=False)


def create_task_tool(tracker: Tracker, arguments: dict[str, Any]) -> str:
    summary = arguments.get("summary")
    if not isinstance(summary, str):
        raise TrackerError("Параметр summary обязателен и должен быть строкой")
    task = tracker.create_task(
        summary,
        assignee=str(arguments.get("assignee", "")),
        priority=str(arguments.get("priority", "normal")),
    )
    return json.dumps(task_view(task), ensure_ascii=False)


# Описание инструментов уходит клиенту в tools/list: по нему модель решает,
# что вызвать и с какими аргументами, поэтому описания пишутся для модели.
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_tasks",
        "description": "Возвращает задачи трекера. Фильтр status ограничивает список "
                       "одним статусом.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(STATUSES),
                    "description": "Статус задач: open, in_progress или done.",
                },
            },
        },
        "handler": list_tasks_tool,
    },
    {
        "name": "get_task",
        "description": "Возвращает одну задачу по её ключу, например TR-2.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Ключ задачи, например TR-2."},
            },
            "required": ["key"],
        },
        "handler": get_task_tool,
    },
    {
        "name": "create_task",
        "description": "Создаёт задачу в статусе open и возвращает её с новым ключом.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Название задачи."},
                "assignee": {"type": "string", "description": "Исполнитель, необязательно."},
                "priority": {
                    "type": "string",
                    "enum": list(PRIORITIES),
                    "description": "Приоритет: low, normal или high. По умолчанию normal.",
                },
            },
            "required": ["summary"],
        },
        "handler": create_task_tool,
    },
)
TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS}


def rpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def call_tool(tracker: Tracker, params: dict[str, Any]) -> dict[str, Any]:
    """Исполняет инструмент. Ошибка предметной области уходит результатом с isError,
    а не JSON-RPC ошибкой: модель должна увидеть текст и выбрать другой путь."""
    name = params.get("name")
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        raise KeyError(name)
    arguments = params.get("arguments") or {}
    try:
        text = tool["handler"](tracker, arguments)
    except TrackerError as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle_message(tracker: Tracker, message: Any) -> dict[str, Any] | None:
    """Ответ на одно JSON-RPC сообщение; None — уведомление, ответа не требует."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return rpc_error(None, INVALID_REQUEST, "Ожидается объект JSON-RPC 2.0")
    method = message.get("method")
    if "id" not in message:
        return None
    request_id = message["id"]
    if not isinstance(method, str):
        return rpc_error(request_id, INVALID_REQUEST, "Отсутствует метод")
    params = message.get("params") or {}
    if method == "initialize":
        return rpc_result(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "tools/list":
        public = [{key: value for key, value in tool.items() if key != "handler"}
                  for tool in TOOLS]
        return rpc_result(request_id, {"tools": public})
    if method == "tools/call":
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return rpc_error(request_id, INVALID_PARAMS, "Нужен параметр name")
        if params["name"] not in TOOLS_BY_NAME:
            return rpc_error(request_id, INVALID_PARAMS,
                             f"Инструмент {params['name']} не найден")
        return rpc_result(request_id, call_tool(tracker, params))
    return rpc_error(request_id, METHOD_NOT_FOUND, f"Метод {method} не поддерживается")


def make_handler(tracker: Tracker) -> type[BaseHTTPRequestHandler]:
    class McpHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — имя задано http.server
            if self.path != MCP_PATH:
                self._send_json(404, rpc_error(None, INVALID_REQUEST, "Нет такого пути"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._send_json(413, rpc_error(None, INVALID_REQUEST, "Тело запроса неверной длины"))
                return
            raw = self.rfile.read(length)
            try:
                message = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, rpc_error(None, PARSE_ERROR, "Тело не является JSON"))
                return
            response = handle_message(tracker, message)
            if response is None:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send_json(200, response)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # Журнал HTTP-запросов сервера не нужен: вызовы видны в ответах клиента.
            return

    return McpHandler


def serve(host: str, port: int, tracker: Tracker | None = None) -> ThreadingHTTPServer:
    """Создаёт сервер, но не запускает его: вызывающий решает, когда крутить цикл."""
    return ThreadingHTTPServer((host, port), make_handler(tracker or Tracker()))


def main(argv: list[str]) -> int:
    host, port = "127.0.0.1", 8765
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg in ("--host", "--port") and args:
            value = args.pop(0)
            if arg == "--host":
                host = value
            else:
                port = int(value)
        else:
            print(f"Неизвестный аргумент: {arg}", file=sys.stderr)
            return 1
    httpd = serve(host, port)
    print(f"MCP mock-трекер слушает http://{host}:{httpd.server_address[1]}{MCP_PATH}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
