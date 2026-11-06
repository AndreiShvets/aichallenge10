"""Клиент шлюза Veai: OpenAI-совместимый список моделей и чат.

Документация: https://veai.ru/docs/veai/connect/api
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

BASE_URL = "https://api.veai.ru/openai/api"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODELS_URL = f"{BASE_URL}/models"
DEFAULT_TIMEOUT = 180.0


class VeaiError(RuntimeError):
    """Ошибка обращения к шлюзу Veai."""


class VeaiClient:
    def __init__(self, api_key: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request(MODELS_URL, method="GET")
        models = data.get("data")
        if not isinstance(models, list):
            raise VeaiError(f"Неожиданный формат списка моделей: {data}")
        return models

    def chat(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Отправляет один запрос в чат. Возвращает ответ API целиком.

        Схема передаётся в формате OpenAI. json_object шлюз отклоняет с HTTP 400:
        без схемы он не гарантирует JSON.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop:
            body["stop"] = stop
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
        return self._request(
            CHAT_URL, method="POST", data=json.dumps(body, ensure_ascii=False).encode()
        )

    def _request(
        self, url: str, method: str, data: bytes | None = None
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            hint = ""
            if exc.code == 401:
                hint = " Проверьте ключ VEAI_CREDENTIALS."
            elif exc.code == 429:
                hint = " Закончились минуты подписки Veai."
            elif exc.code == 404:
                hint = f" Модель не найдена, сверьте имя со списком {MODELS_URL}."
            raise VeaiError(f"HTTP {exc.code} при запросе {url}:{hint} {detail}") from exc
        except urllib.error.URLError as exc:
            raise VeaiError(f"Не удалось подключиться к {url}: {exc.reason}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VeaiError(f"Ответ {url} не является JSON: {raw[:200]!r}") from exc
