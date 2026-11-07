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


class GigaChatClient:
    def __init__(
        self,
        credentials: str,
        scope: str = DEFAULT_SCOPE,
        timeout: float = DEFAULT_TIMEOUT,
        ca_cert_files: tuple[Path, ...] = (),
        token_cache_file: Path | None = None,
    ) -> None:
        self.credentials = credentials
        self.scope = scope
        self.timeout = timeout
        self.token_cache_file = token_cache_file
        self.token_source = "нет"
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
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Отправляет один запрос в чат. Возвращает ответ API целиком.

        reasoning_effort — уровень рассуждений. API принимает low, medium и high,
        на max отвечает HTTP 422 «Unknown reasoning_effort».
        temperature — температура сэмплирования, от 0 до 2.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop:
            body["stop"] = stop
        # Именно is not None: температура 0.0 — валидное значение, а `if temperature` его теряет.
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        headers = {**_auth_headers(access_token), "Content-Type": "application/json"}
        url = f"{API_BASE_URL}/chat/completions"
        return self._post_chat(url, headers, json.dumps(body, ensure_ascii=False).encode())

    def _post_chat(self, url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
        return self._request(url, method="POST", headers=headers, data=body)

    def _request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        data: bytes | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._ssl_context
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            hint = ""
            if exc.code == 401:
                hint = " Проверьте ключ авторизации GIGACHAT_CREDENTIALS."
            elif exc.code in (400, 403) and "oauth" in url:
                hint = f" Проверьте, что ключ выдан для scope {self.scope}."
            raise GigaChatError(f"HTTP {exc.code} при запросе {url}:{hint} {detail}") from exc
        except urllib.error.URLError as exc:
            raise GigaChatError(f"Не удалось подключиться к {url}: {exc.reason}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GigaChatError(f"Ответ {url} не является JSON: {raw[:200]!r}") from exc


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
