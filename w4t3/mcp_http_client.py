"""Клиент MCP поверх HTTP: тот же протокол, что в w4t1, другой транспорт.

Каждое JSON-RPC сообщение уходит отдельным POST на endpoint сервера. Ответ
на запрос — JSON-тело; на уведомление сервер отвечает 202 без тела. Сессий
нет: сервер не выдаёт идентификатора сессии, поэтому клиент его не хранит.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = "2025-03-26"
CLIENT_INFO = {"name": "w4t2-agent", "version": "1.0.0"}
# Сетевой таймаут на один POST: локальный сервер отвечает за миллисекунды,
# а зависший ответ не должен держать агента бесконечно.
REQUEST_TIMEOUT_SECONDS = 10.0


class McpError(RuntimeError):
    """Ошибка MCP-соединения, протокола или вызова инструмента на сервере."""


@dataclass(frozen=True)
class McpServerInfo:
    name: str
    version: str


@dataclass(frozen=True)
class McpTool:
    """Инструмент, объявленный сервером: имя, описание и JSON-схема входа."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        """Схема для поля `functions` запроса к модели."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


class McpHttpClient:
    """Соединение с MCP-сервером по HTTP: рукопожатие, список и вызов инструментов."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._next_id = 0

    def start(self) -> McpServerInfo:
        """Проверяет доступность сервера рукопожатием initialize."""
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        self._notify("notifications/initialized", {})
        info = result.get("serverInfo") or {}
        return McpServerInfo(str(info.get("name", "")), str(info.get("version", "")))

    def list_tools(self) -> list[McpTool]:
        result = self._request("tools/list", {})
        return [
            McpTool(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                input_schema=item.get("inputSchema") or {},
            )
            for item in result.get("tools") or []
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Вызов инструмента; ошибка инструмента (isError) поднимается McpError."""
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text = "".join(
            str(block.get("text", ""))
            for block in result.get("content") or []
            if block.get("type") == "text"
        )
        if result.get("isError"):
            raise McpError(text or f"Инструмент {name} вернул ошибку без текста")
        return text

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        message = self._post({"jsonrpc": "2.0", "id": self._next_id,
                              "method": method, "params": params})
        if message is None:
            raise McpError(f"MCP-сервер не вернул тело на запрос {method}")
        if "error" in message:
            error = message["error"] or {}
            raise McpError(f"MCP-сервер отклонил {method}: "
                           f"{error.get('code')} {error.get('message')}")
        return message.get("result") or {}

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params})

    def _post(self, message: dict[str, Any]) -> dict[str, Any] | None:
        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # 400/413 сервер присылает JSON-RPC ошибкой в теле — её и покажем.
            raw = exc.read()
            if not raw:
                raise McpError(f"MCP-сервер ответил HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise McpError(f"Нет связи с MCP-сервером {self._url}: {exc}") from exc
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise McpError(f"MCP-сервер вернул не JSON: {exc}") from exc
