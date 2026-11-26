"""Текст сводки через GigaChat: модель пересказывает цифры, которые уже посчитал код.

Модель не считает: числа приходят в промпт готовыми из aggregate.Report, так
что текст не может выдумать статистику, которой нет в журнале.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol

from aggregate import Report
from gigachat_client import (
    TOKEN_CACHE_FILE,
    GigaChatClient,
    GigaChatError,
    default_ca_cert_files,
)

GIGACHAT_CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"
GIGACHAT_SCOPE_ENV = "GIGACHAT_SCOPE"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_MODEL = "GigaChat"
SUMMARY_MAX_TOKENS = 400


class SummaryError(RuntimeError):
    """Сводку не удалось получить: нет ключа или модель не ответила."""


class ChatClient(Protocol):
    def get_access_token(self) -> str: ...

    def chat(self, access_token: str, model: str, prompt: str,
             max_tokens: int | None = None) -> dict: ...


def build_prompt(report: Report, lines: list[str]) -> str:
    """Промпт с готовыми строками отчёта: модель видит только то, что посчитал код."""
    facts = "\n".join(f"- {line}" for line in lines)
    return (
        "Ниже сводка по журналу запусков сборщика задач трекера. Напиши короткий "
        "связный текст на русском: 2–3 предложения о динамике. Используй только "
        "приведённые цифры, ничего не придумывай. Числа пиши обычными цифрами, "
        "без формул и LaTeX.\n\n"
        f"Всего записей: {report.runs_total}, снимков: {report.snapshots}.\n"
        f"{facts}"
    )


def summarize(client: ChatClient, report: Report, lines: list[str]) -> str:
    try:
        token = client.get_access_token()
        response = client.chat(token, DEFAULT_MODEL, build_prompt(report, lines),
                               max_tokens=SUMMARY_MAX_TOKENS)
    except GigaChatError as exc:
        raise SummaryError(f"GigaChat не ответил: {exc}") from exc
    choices = response.get("choices") or []
    if not choices:
        raise SummaryError(f"Ответ GigaChat без choices: {response}")
    return str((choices[0].get("message") or {}).get("content") or "").strip()


def find_env_file(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise SummaryError(f"Файл .env не найден ни в {start}, ни в его родительских папках.")


def build_summarizer(start: Path) -> Callable[[Report, list[str]], str]:
    """Клиент GigaChat из .env, который ищется вверх от start."""
    credentials = load_credentials(find_env_file(start))
    scope = os.environ.get(GIGACHAT_SCOPE_ENV, "").strip() or DEFAULT_SCOPE
    client = GigaChatClient(
        credentials=credentials,
        scope=scope,
        ca_cert_files=default_ca_cert_files(),
        token_cache_file=TOKEN_CACHE_FILE,
    )

    def summarizer(report: Report, lines: list[str]) -> str:
        return summarize(client, report, lines)

    return summarizer


def load_credentials(env_file: Path) -> str:
    """Читает ключ из .env, не перетирая уже заданные переменные окружения."""
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    credentials = os.environ.get(GIGACHAT_CREDENTIALS_ENV, "").strip()
    if not credentials:
        raise SummaryError(f"В {env_file} не задан {GIGACHAT_CREDENTIALS_ENV}.")
    return credentials
