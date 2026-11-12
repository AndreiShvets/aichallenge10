"""Клиент GigaChat API: авторизация по ключу, список моделей, баланс и чат.

Документация: https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import tempfile
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_TIMEOUT = 120.0
# Токен живёт 30 минут; запас на то, чтобы не поймать протухание посреди запроса.
TOKEN_EXPIRY_MARGIN_SECONDS = 60.0
TOKEN_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
TOKEN_CACHE_FILE = TOKEN_CACHE_DIR / "gigachat_token.json"
CERT_DIR = Path(__file__).resolve().parent / "certs"
CA_CERT_FILES = ("russian_trusted_root.cer", "russian_trusted_sub_ca.cer")


class GigaChatError(RuntimeError):
    """Ошибка обращения к GigaChat API."""


# Что записать о каждом HTTP-обмене: время, шаг, метод, URL, запрос, статус,
# ответ. Пишется и успешный ответ, и ошибка HTTP, и отказ сети.
HttpLogger = Callable[[dict[str, Any]], None]


class GigaChatClient:
    def __init__(
        self,
        credentials: str,
        scope: str = DEFAULT_SCOPE,
        timeout: float = DEFAULT_TIMEOUT,
        ca_cert_files: tuple[Path, ...] = (),
        token_cache_file: Path | None = None,
        http_logger: HttpLogger | None = None,
    ) -> None:
        self.credentials = credentials
        self.scope = scope
        self.timeout = timeout
        self.token_cache_file = token_cache_file
        self.token_source = "нет"
        self.http_logger = http_logger
        # Номер запроса в сеансе и шаг цикла, которые идут в HTTP-лог: по ним
        # видно, какое сообщение какого диалога описывает строка.
        self.log_context: dict[str, Any] | None = None
        self._ssl_context = _build_ssl_context(ca_cert_files)

    def get_access_token(self, use_cache: bool = True) -> str:
        self.token_source = "новый"
        if use_cache and self.token_cache_file is not None:
            cached = _read_cached_token(self.token_cache_file, self.credentials, self.scope)
            if cached is not None:
                self.token_source = "из кэша"
                return cached
        return self._fetch_access_token()

    def _fetch_access_token(self) -> str:
        body = urllib.parse.urlencode({"scope": self.scope}).encode()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.credentials}",
        }
        data = self._request(OAUTH_URL, method="POST", headers=headers, data=body)
        token = data.get("access_token")
        if not token:
            raise GigaChatError(f"Ответ авторизации не содержит access_token: {data}")
        if self.token_cache_file is not None:
            _write_cached_token(
                self.token_cache_file, token, data.get("expires_at"), self.credentials, self.scope
            )
        return token

    def list_models(self, access_token: str) -> list[dict[str, Any]]:
        url = f"{API_BASE_URL}/models"
        data = self._request(url, method="GET", headers=_auth_headers(access_token))
        models = data.get("data")
        if not isinstance(models, list):
            raise GigaChatError(f"Неожиданный формат списка моделей: {data}")
        return models

    def get_balance(self, access_token: str) -> list[dict[str, Any]]:
        """Остаток токенов по моделям. Доступен только для пакетов токенов."""
        url = f"{API_BASE_URL}/balance"
        try:
            data = self._request(url, method="GET", headers=_auth_headers(access_token))
        except GigaChatError as exc:
            if "HTTP 403" in str(exc):
                raise GigaChatError(
                    "Остаток токенов недоступен: метод работает только при покупке "
                    "пакетов токенов (при pay-as-you-go API отвечает 403)."
                ) from exc
            raise
        balance = data.get("balance")
        if not isinstance(balance, list):
            raise GigaChatError(f"Неожиданный формат баланса: {data}")
        return balance

    def chat(
        self,
        access_token: str,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Отправляет один запрос в чат. Возвращает ответ API целиком.

        reasoning_effort — уровень рассуждений. API принимает low, medium и high,
        на max отвечает HTTP 422 «Unknown reasoning_effort».
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop:
            body["stop"] = stop
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return self.chat_messages(access_token, body)

    def chat_messages(self, access_token: str, body: dict[str, Any]) -> dict[str, Any]:
        """Отправляет готовое тело запроса /chat/completions.

        Нужен для диалога с функциями: там в теле не только модель и сообщения,
        но и массив functions и режим function_call, а история сообщений растёт
        после каждого вызова инструмента.
        """
        headers = {**_auth_headers(access_token), "Content-Type": "application/json"}
        url = f"{API_BASE_URL}/chat/completions"
        context = body.pop("context", None)
        return self._post_chat(
            url, headers, json.dumps(body, ensure_ascii=False).encode(), context
        )

    def _post_chat(
        self,
        url: str,
        headers: dict[str, str],
        body: bytes,
        log_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(url, method="POST", headers=headers, data=body,
                             log_context=log_context)

    def _request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        data: bytes | None = None,
        log_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        started = time.time()
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._ssl_context
            ) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            detail = raw.decode("utf-8", "replace").strip()
            self._log_http(method, url, headers, data, exc.code, raw,
                           time.time() - started, log_context)
            hint = ""
            if exc.code == 401:
                hint = " Проверьте ключ авторизации GIGACHAT_CREDENTIALS."
            elif exc.code in (400, 403) and "oauth" in url:
                hint = f" Проверьте, что ключ выдан для scope {self.scope}."
            raise GigaChatError(f"HTTP {exc.code} при запросе {url}:{hint} {detail}") from exc
        except urllib.error.URLError as exc:
            self._log_http(method, url, headers, data, None, b"",
                           time.time() - started, log_context, str(exc.reason))
            raise GigaChatError(f"Не удалось подключиться к {url}: {exc.reason}") from exc
        self._log_http(method, url, headers, data, status, raw, time.time() - started,
                       log_context)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GigaChatError(f"Ответ {url} не является JSON: {raw[:200]!r}") from exc

    def _log_http(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None,
        status: int | None,
        raw: bytes,
        seconds: float,
        log_context: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        """Отдаёт обмен вызывающему коду, который решает, куда его записать.

        Тело — как есть: вызов этого метода не должен ронять запрос к API,
        поэтому телом занимается сам логгер.
        """
        if self.http_logger is None:
            return
        self.http_logger({
            "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
            **(self.log_context or {}),
            **(log_context or {}),
            "method": method,
            "url": url,
            "request_headers": _redact_headers(headers),
            "request_body": _redact_body(data.decode("utf-8", "replace")) if data else "",
            "status": status,
            "response_body": _redact_body(raw.decode("utf-8", "replace")),
            "seconds": round(seconds, 3),
            "error": error,
        })


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Копия заголовков без секретов: токен нужен только в самом запросе."""
    # Заголовок Basic в запросе авторизации — это и есть ключ GIGACHAT_CREDENTIALS.
    return {
        key: (
            "<скрыт>"
            if key.lower() == "authorization" and value.startswith("Basic ")
            else "Bearer <скрыт>" if key.lower() == "authorization" else value
        )
        for key, value in headers.items()
    }


def _redact_body(text: str) -> str:
    """Убирает из тела ответа выданный access_token.

    Лог должен быть безопасно пересылать и хранить, а токен живёт 30 минут
    и даёт доступ к API от имени ключа.
    """
    if not text:
        return text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict) and "access_token" in payload:
        payload["access_token"] = "<скрыт>"
        return json.dumps(payload, ensure_ascii=False)
    return text


def _auth_headers(access_token: str) -> dict[str, str]:
    return {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}


def _build_ssl_context(ca_cert_files: tuple[Path, ...]) -> ssl.SSLContext:
    # Python не доверяет корневому сертификату НУЦ Минцифры, которым подписан
    # сертификат GigaChat. Если положить оба PEM-сертификата в один файл,
    # OpenSSL на Windows не разбирает его из-за CRLF, поэтому каждый файл
    # загружается отдельно.
    context = ssl.create_default_context()
    for cert_file in ca_cert_files:
        if not cert_file.is_file():
            raise GigaChatError(f"Сертификат не найден: {cert_file}")
        context.load_verify_locations(cadata=cert_file.read_text(encoding="utf-8"))
    return context


def default_ca_cert_files() -> tuple[Path, ...]:
    return tuple(CERT_DIR / name for name in CA_CERT_FILES)


def _read_cached_token(cache_file: Path, credentials: str, scope: str) -> str | None:
    """Возвращает токен из файла, если он не протух и выдан под те же условия."""
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # ValueError покрывает и JSONDecodeError, и UnicodeDecodeError.
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("scope") != scope or cached.get("credentials_hash") != _hash(credentials):
        return None
    token = cached.get("access_token")
    expires_at = cached.get("expires_at")
    if not isinstance(token, str) or not isinstance(expires_at, (int, float)):
        return None
    if expires_at / 1000 - TOKEN_EXPIRY_MARGIN_SECONDS <= time.time():
        return None
    return token


def _write_cached_token(
    cache_file: Path,
    token: str,
    expires_at: Any,
    credentials: str,
    scope: str,
) -> None:
    if not isinstance(expires_at, (int, float)):
        return
    payload = {
        "access_token": token,
        "expires_at": int(expires_at),
        "scope": scope,
        "credentials_hash": _hash(credentials),
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Пишем через временный файл, чтобы оборванный запуск не оставил битый кэш.
        fd, tmp_path = tempfile.mkstemp(dir=cache_file.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
        os.replace(tmp_path, cache_file)
    except OSError as exc:
        raise GigaChatError(f"Не удалось сохранить токен в {cache_file}: {exc}") from exc


def _hash(value: str) -> str:
    # Сам ключ в кэше не нужен: храним только хеш для проверки, что токен от того же ключа.
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
