"""MCP-сервер поверх HTTP: трекер для сбора данных и журнал запусков для планировщика.

Протокол — JSON-RPC 2.0. Клиент шлёт каждое сообщение POST-запросом на
`/mcp`; ответ приходит JSON-телом, уведомления получают 202 без тела.
Сервер не хранит сессий: состояние трекера живёт в процессе сервера, журнал
запусков — в JSON-файле, переживает перезапуск.

Запуск: python server.py [--host 127.0.0.1] [--port 8765] [--runs PATH] [--verbose]
--verbose печатает каждый полученный запрос и отправленный ответ.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from store import RUN_KINDS, Run, RunStore, StoreError, parse_period
from tracker import PRIORITIES, STATUSES, Task, Tracker, TrackerError

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "snapshot-server", "version": "0.1.0"}
MCP_PATH = "/mcp"
MAX_BODY_BYTES = 1_000_000
DEFAULT_RUNS_PATH = Path(__file__).resolve().parent / "runs.json"

# Коды ошибок JSON-RPC из спецификации.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

# Ошибки, которые показываются модели текстом результата (isError), а не JSON-RPC.
DOMAIN_ERRORS = (TrackerError, StoreError)


@dataclass
class Services:
    """Всё, что инструменты используют: трекер для данных и журнал для хранения."""

    tracker: Tracker
    store: RunStore


def task_view(task: Task) -> dict[str, str]:
    return {
        "key": task.key,
        "summary": task.summary,
        "status": task.status,
        "assignee": task.assignee,
        "priority": task.priority,
    }


def run_view(run: Run) -> dict[str, Any]:
    return {"id": run.id, "at": run.at, "kind": run.kind, "data": run.data}


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _optional_str(arguments: dict[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TrackerError(f"Параметр {name} должен быть строкой")
    return value


def fetch_snapshot_tool(services: Services, arguments: dict[str, Any]) -> str:
    """Опрос трекера: каждый вызов — новый момент, поэтому состояние между вызовами меняется."""
    status = _optional_str(arguments, "status")
    services.tracker.tick()
    tasks = services.tracker.list_tasks(status)
    return _dump({
        "taken_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": [task_view(task) for task in tasks],
    })


def save_run_tool(services: Services, arguments: dict[str, Any]) -> str:
    kind = arguments.get("kind")
    if not isinstance(kind, str):
        raise StoreError(f"Параметр kind обязателен. Доступны: {', '.join(RUN_KINDS)}")
    data = arguments.get("data")
    if not isinstance(data, dict):
        raise StoreError("Параметр data обязателен и должен быть объектом")
    return _dump(run_view(services.store.save(kind, data)))


def read_runs_tool(services: Services, arguments: dict[str, Any]) -> str:
    start, end = parse_period(_optional_str(arguments, "from"), _optional_str(arguments, "to"))
    runs = services.store.read(start, end, _optional_str(arguments, "kind"))
    return _dump([run_view(run) for run in runs])


# Описание инструментов уходит клиенту в tools/list: по нему модель решает,
# что вызвать и с какими аргументами, поэтому описания пишутся для модели.
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "fetch_snapshot",
        "description": "Опрашивает трекер и возвращает текущие задачи с момента опроса. "
                       "Каждый вызов — новый момент: состояние задач меняется между вызовами.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(STATUSES),
                    "description": "Только задачи с этим статусом. Без фильтра — все задачи.",
                },
            },
        },
        "handler": fetch_snapshot_tool,
    },
    {
        "name": "save_run",
        "description": "Сохраняет результат запуска в журнал и возвращает запись с номером.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": list(RUN_KINDS),
                    "description": "Вид запуска: snapshot — сбор данных, reminder — напоминание.",
                },
                "data": {
                    "type": "object",
                    "description": "Результат запуска: например, ответ fetch_snapshot.",
                },
            },
            "required": ["kind", "data"],
        },
        "handler": save_run_tool,
    },
    {
        "name": "read_runs",
        "description": "Возвращает накопленные записи журнала за период, от старых к новым. "
                       "Период задаётся датами: from включительно, to — весь этот день.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Начало периода, YYYY-MM-DD."},
                "to": {"type": "string", "description": "Конец периода, YYYY-MM-DD."},
                "kind": {
                    "type": "string",
                    "enum": list(RUN_KINDS),
                    "description": "Только записи этого вида. Без фильтра — все виды.",
                },
            },
        },
        "handler": read_runs_tool,
    },
)
TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS}


def rpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def call_tool(services: Services, params: dict[str, Any]) -> dict[str, Any]:
    """Исполняет инструмент. Ошибка предметной области уходит результатом с isError,
    а не JSON-RPC ошибкой: модель должна увидеть текст и выбрать другой путь."""
    tool = TOOLS_BY_NAME[params["name"]]
    arguments = params.get("arguments") or {}
    try:
        text = tool["handler"](services, arguments)
    except DOMAIN_ERRORS as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle_message(services: Services, message: Any) -> dict[str, Any] | None:
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
        return rpc_result(request_id, call_tool(services, params))
    return rpc_error(request_id, METHOD_NOT_FOUND, f"Метод {method} не поддерживается")


def _show(direction: str, text: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {direction}\n{text}", flush=True)


def make_handler(services: Services, verbose: bool = False) -> type[BaseHTTPRequestHandler]:
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
            if verbose:
                _show("← запрос", raw.decode("utf-8", errors="replace"))
            try:
                message = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, rpc_error(None, PARSE_ERROR, "Тело не является JSON"))
                return
            response = handle_message(services, message)
            if response is None:
                if verbose:
                    _show("→ ответ HTTP 202", "(уведомление, тела нет)")
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send_json(200, response)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if verbose:
                _show(f"→ ответ HTTP {status}", body.decode("utf-8"))
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # Журнал HTTP-запросов сервера не нужен: вызовы видны в ответах клиента.
            return

    return McpHandler


def serve(host: str, port: int, services: Services | None = None,
          verbose: bool = False) -> ThreadingHTTPServer:
    """Создаёт сервер, но не запускает его: вызывающий решает, когда крутить цикл."""
    services = services or Services(Tracker(), RunStore(DEFAULT_RUNS_PATH))
    return ThreadingHTTPServer((host, port), make_handler(services, verbose))


def main(argv: list[str]) -> int:
    # Консоль Windows по умолчанию не в UTF-8: без этого кириллица в --verbose портится.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    host, port, verbose, runs_path = "127.0.0.1", 8765, False, DEFAULT_RUNS_PATH
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--verbose":
            verbose = True
        elif arg in ("--host", "--port", "--runs") and args:
            value = args.pop(0)
            if arg == "--host":
                host = value
            elif arg == "--port":
                port = int(value)
            else:
                runs_path = Path(value)
        else:
            print(f"Неизвестный аргумент: {arg}. Доступны: --host, --port, --runs, --verbose",
                  file=sys.stderr)
            return 1
    httpd = serve(host, port, Services(Tracker(), RunStore(runs_path)), verbose=verbose)
    print(f"MCP-сервер слушает http://{host}:{httpd.server_address[1]}{MCP_PATH}, журнал: {runs_path}")
    if verbose:
        print("Режим --verbose: печатаются полученные запросы и отправленные ответы.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
