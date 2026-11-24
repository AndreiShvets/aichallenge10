"""Мини-клиент MCP поверх stdio на стандартной библиотеке.

Транспорт stdio в протоколе MCP — это JSON-RPC 2.0 построчно по stdin/stdout
поднятого серверного процесса. Соединение устанавливается рукопожатием
`initialize`, после него клиент запрашивает список инструментов `tools/list`
и вызывает их через `tools/call`. Официальный SDK не ставится: правила
репозитория запрещают зависимости, а нужная часть протокола укладывается
в один файл стандартной библиотеки.

Сервер по умолчанию — официальный `@modelcontextprotocol/server-filesystem`
через `npx`: инструменты этого сервера агент не писал сам, как и требует
задача. На Windows `npx` — это .cmd-файл, который `Popen` без оболочки
не запускает, поэтому команда заворачивается в `cmd /c`.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Версия протокола, которую понимает клиент; сервер отвечает своей
# согласованной версией в результате initialize.
PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "w4t1-agent", "version": "1.0.0"}
# Ждать ответ сервера дольше этого — считать соединение зависшим.
RESPONSE_TIMEOUT_SECONDS = 60.0
# На initialize таймаут шире: под ним npx скачивает пакет сервера при первом
# запуске, и это легитимно занимает минуты на медленной сети.
STARTUP_TIMEOUT_SECONDS = 180.0
# Сколько ждать добровольного выхода сервера после закрытия stdin.
KILL_TIMEOUT_SECONDS = 5.0
FILESYSTEM_SERVER_PACKAGE = "@modelcontextprotocol/server-filesystem"


class McpError(RuntimeError):
    """Ошибка MCP-соединения, протокола или вызова инструмента на сервере."""


@dataclass(frozen=True)
class McpServerInfo:
    """Что сервер рассказал о себе в ответе initialize."""

    name: str
    version: str


@dataclass(frozen=True)
class McpTool:
    """Инструмент, объявленный сервером MCP.

    Реализации у клиента нет: имя и аргументы уезжают на сервер в `tools/call`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        """Схема для поля `functions` запроса к модели (GigaChat/OpenAI-вид)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


def default_server_command(root: Path) -> list[str]:
    """Команда запуска filesystem-сервера с доступом только к `root`."""
    command = ["npx", "-y", FILESYSTEM_SERVER_PACKAGE, str(root)]
    if os.name == "nt":
        # CreateProcess не ищет .cmd по PATH; cmd /c — документированный обход.
        return ["cmd", "/c", *command]
    return command


class McpStdioClient:
    """Соединение с MCP-сервером: подъём процесса и JSON-RPC по его трубам.

    Ответы читает фоновый поток в очередь: `readline` без таймаута завис бы
    навсегда на замолчавшем сервере, а `select` на трубах в Windows не работает.
    """

    def __init__(self, command: list[str]) -> None:
        self._command = command
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._next_id = 0

    def start(self) -> McpServerInfo:
        """Поднимает сервер и проводит рукопожатие initialize."""
        try:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stderr не перехватывается: прогресс первого скачивания npx
                # и ошибки запуска видны в консоли и не забивают трубу.
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise McpError(
                f"Не удалось запустить MCP-сервер {self._command}: {exc}"
            ) from exc
        threading.Thread(target=self._read_lines, daemon=True).start()
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
            timeout=STARTUP_TIMEOUT_SECONDS,
        )
        info = result.get("serverInfo") or {}
        server = McpServerInfo(
            str(info.get("name", "")), str(info.get("version", ""))
        )
        self._notify("notifications/initialized", {})
        return server

    def list_tools(self) -> list[McpTool]:
        """Список инструментов сервера, с учётом пагинации по курсору."""
        tools: list[McpTool] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = self._request("tools/list", params)
            for item in result.get("tools") or []:
                tools.append(McpTool(
                    name=str(item.get("name", "")),
                    description=str(item.get("description", "")),
                    input_schema=item.get("inputSchema") or {},
                ))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Вызов инструмента на сервере; возвращает текстовые блоки ответа.

        isError у результата означает ошибку самого инструмента: её нельзя
        терять, модель должна увидеть текст ошибки и выбрать другой путь.
        """
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(
            str(block.get("text", ""))
            for block in result.get("content") or []
            if block.get("type") == "text"
        )
        if result.get("isError"):
            raise McpError(text or f"Инструмент {name} вернул ошибку без текста")
        return text

    def close(self) -> None:
        """Закрывает stdin и ждёт выход сервера; зависший — убивает."""
        process = self._process
        if process is None:
            return
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.wait(timeout=KILL_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        finally:
            # Трубы закрываются в любом случае: незакрытый stdout оставляет
            # ResourceWarning и держит дескриптор до сборщика мусора.
            for stream in (process.stdout, process.stdin):
                if stream and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._process = None

    def _request(self, method: str, params: dict[str, Any],
                 timeout: float = RESPONSE_TIMEOUT_SECONDS) -> dict[str, Any]:
        """Запрос с ожиданием ответа по id; уведомления и чужие запросы мимо."""
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                    "params": params})
        deadline = timeout
        while True:
            try:
                message = json.loads(self._lines.get(timeout=deadline))
            except queue.Empty as exc:
                raise McpError(
                    f"MCP-сервер не ответил на {method} за {timeout:g} с"
                ) from exc
            except json.JSONDecodeError:
                continue  # мусорная строка в stdout сервера — не ответ
            if message.get("id") != request_id or (
                "result" not in message and "error" not in message
            ):
                # Уведомления пропускаются; запрос от сервера (id + method)
                # надо отклонить, иначе сервер будет ждать ответ сам.
                if "method" in message and "id" in message:
                    self._send({"jsonrpc": "2.0", "id": message["id"], "error": {
                        "code": -32601, "message": "Method not found",
                    }})
                continue
            if "error" in message:
                error = message["error"] or {}
                raise McpError(
                    f"MCP-сервер отклонил {method}: "
                    f"{error.get('code')} {error.get('message')}"
                )
            return message.get("result") or {}

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise McpError("MCP-сервер не запущен")
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise McpError(f"Не удалось отправить сообщение MCP-серверу: {exc}") from exc

    def _read_lines(self) -> None:
        """Цикл фонового потока: stdout сервера — построчная очередь ответов."""
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put("")  # метка конца потока: readline вернёт пустую строку
