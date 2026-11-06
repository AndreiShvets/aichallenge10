"""Запрос к GigaChat или шлюзу Veai: модели, ввод промпта, выбор модели, ответ.

Сценарий:
1. список моделей обоих провайдеров;
2. ввод текста запроса и четырёх необязательных параметров: описание формата,
   лимит токенов, stop-последовательность и JSON-схема;
3. выбор провайдера — GigaChat или Veai;
4. список моделей выбранного провайдера только для чата, выбор модели;
5. четыре прогона одного запроса и проверка каждого ответа на валидность JSON:
   без ограничений, с ограничениями, со схемой, с ограничениями и схемой.

Где провайдер не поддерживает смысл параметра (всё, кроме GigaChat функции
и Veai схемы), прогон не отправляется и вместо ответа печатается
«Не поддерживается».
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gigachat_client import (
    TOKEN_CACHE_FILE,
    GigaChatClient,
    GigaChatError,
    default_ca_cert_files,
)
from veai_client import VeaiClient, VeaiError

GIGACHAT_CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"
GIGACHAT_FALLBACK_CREDENTIALS_ENV = "GIGACHAT_API_KEY"
VEAI_CREDENTIALS_ENV = "VEAI_CREDENTIALS"
DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_PROVIDER = "GigaChat"
VEAI_PROVIDER = "Veai"
PROJECT_DIR = Path(__file__).resolve().parent
# Ключи не хранятся в репозитории, поэтому ищем .env от папки задачи вверх по дереву.
ENV_SEARCH_START = PROJECT_DIR


class AppError(RuntimeError):
    """Ошибка сценария: неверный ввод или отказ провайдера."""


@dataclass(frozen=True)
class JsonSchemaField:
    """Поле схемы: имя, тип JSON Schema и описание для модели."""

    name: str
    type: str
    description: str

    def to_schema(self) -> dict[str, Any]:
        return {"type": self.type, "description": self.description}


@dataclass(frozen=True)
class RequestParams:
    """Четыре необязательных параметра, которые вводит пользователь."""

    format_description: str | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    schema_fields: tuple[JsonSchemaField, ...] = ()

    @property
    def has_limits(self) -> bool:
        return bool(self.max_tokens or self.stop)

    @property
    def has_schema(self) -> bool:
        return bool(self.schema_fields)

    def json_schema(self) -> dict[str, Any] | None:
        if not self.schema_fields:
            return None
        return {
            "type": "object",
            "properties": {f.name: f.to_schema() for f in self.schema_fields},
            "required": [f.name for f in self.schema_fields],
            "additionalProperties": False,
        }

    def format_note(self) -> str:
        """Описание формата ответа для промпта.

        Все прогоны должны просить JSON, иначе проверка на валидность JSON
        сравнивает несравнимое; описание пользователя добавляется к этой основе.
        """
        note = "Ответь одним JSON-объектом без пояснений и без markdown."
        if self.format_description:
            note += f" Формат ответа: {self.format_description}"
        return note

    def schema_note(self) -> str:
        fields = ", ".join(f'"{f.name}" ({f.type})' for f in self.schema_fields)
        return f"JSON должен строго соответствовать схеме, поля: {fields}."


def find_env_file(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise AppError(f"Файл .env не найден ни в {start}, ни в его родительских папках.")


def parse_args(argv: list[str]) -> tuple[Path, bool]:
    """Возвращает (путь к .env, использовать ли кэш токена)."""
    env_path: Path | None = None
    use_cache = True
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--no-cache":
            use_cache = False
        elif arg == "--env":
            if not args:
                raise AppError("После --env требуется путь к файлу .env.")
            env_path = Path(args.pop(0)).expanduser()
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env=") :]).expanduser()
        else:
            raise AppError(f"Неизвестный аргумент: {arg}. Доступны: --env PATH, --no-cache.")
    return env_path or find_env_file(ENV_SEARCH_START), use_cache


def load_env_file(path: Path) -> None:
    """Читает .env в os.environ, не перетирая уже заданные переменные."""
    if not path.is_file():
        raise AppError(f"Файл с ключом не найден: {path}")
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    template = "  ".join(f"{{:<{w}}}" for w in widths)
    print(template.format(*headers))
    print(template.format(*("-" * w for w in widths)))
    for row in rows:
        print(template.format(*row))


def chat_models(models: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    """Оставляет модели, в которые можно отправить текстовый запрос.

    В GigaChat тип модели указан в поле type, у Veai типа нет, а способность
    принимать текст описывают input_modalities и output_modalities.
    """
    if provider == GIGACHAT_PROVIDER:
        return [m for m in models if str(m.get("type", "")) == "chat"]
    return [
        m
        for m in models
        if "text" in (m.get("input_modalities") or ["text"])
        and "text" in (m.get("output_modalities") or ["text"])
    ]


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError as exc:
        raise AppError("Ввод прерван: не получены ответы на вопросы.") from exc


def ask_text(title: str) -> str:
    print(f"\n{title} (пустая строка завершает ввод):")
    lines: list[str] = []
    while True:
        try:
            line = input("> " if not lines else "| ")
        except EOFError:
            line = ""
        if not line.strip():
            if lines:
                return "\n".join(lines).strip()
            print("Введите хотя бы одну строку.")
            continue
        lines.append(line.rstrip())


def ask_choice(title: str, options: list[str], allow_numbers: bool = True) -> str:
    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        raw = ask("Ваш выбор: ")
        if not raw:
            print("Введите номер варианта.")
            continue
        if allow_numbers and raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(options):
                return options[index - 1]
            print(f"Номер должен быть от 1 до {len(options)}.")
            continue
        # Дополнительно принимаем точное имя модели: так удобнее для длинных id.
        if raw in options:
            return raw
        print("Такого варианта нет.")
        continue


def ask_optional(title: str, hint: str = "пустая строка — не задавать") -> str | None:
    print(f"\n{title} ({hint}):")
    value = ask("> ")
    return value or None


def ask_max_tokens() -> int | None:
    while True:
        raw = ask_optional("3. Ограничение объёма ответа в токенах")
        if raw is None:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Введите целое число больше 0.")


def ask_stop() -> list[str] | None:
    while True:
        raw = ask_optional("4. Stop sequence (несколько — через запятую)")
        if raw is None:
            return None
        sequences = [part.strip() for part in raw.split(",") if part.strip()]
        if sequences:
            return sequences
        print("Введите хотя бы одну последовательность.")


def ask_schema_fields() -> tuple[JsonSchemaField, ...]:
    while True:
        raw = ask_optional(
            "5. Строгая JSON-схема ответа",
            "пустая строка — без схемы, иначе поля через запятую: name:type[:описание]",
        )
        if raw is None:
            return ()
        fields: list[JsonSchemaField] = []
        for part in raw.split(","):
            pieces = [piece.strip() for piece in part.strip().split(":", 2)]
            if len(pieces) < 2 or not pieces[0]:
                print(f"Не удалось разобрать «{part.strip()}»: нужен вид name:type[:описание].")
                fields = []
                break
            name, field_type = pieces[0], pieces[1]
            description = pieces[2] if len(pieces) > 2 and pieces[2] else name
            fields.append(JsonSchemaField(name=name, type=field_type, description=description))
        if fields:
            return tuple(fields)


def ask_params() -> RequestParams:
    print("\nПараметры запроса (все необязательные):")
    return RequestParams(
        format_description=ask_optional("2. Явное описание формата ответа"),
        max_tokens=ask_max_tokens(),
        stop=ask_stop(),
        schema_fields=ask_schema_fields(),
    )


def describe_params(params: RequestParams) -> None:
    print("\nЗаданы параметры:")
    print(f"  описание формата: {params.format_description or '—'}")
    print(f"  max_tokens: {params.max_tokens or '—'}")
    print(f"  stop: {', '.join(params.stop) if params.stop else '—'}")
    print(
        "  JSON-схема: "
        + (
            ", ".join(f"{f.name}:{f.type}" for f in params.schema_fields)
            if params.schema_fields
            else "—"
        )
    )


def model_column(value: Any, limit: int = 60) -> str:
    return str(value) if value is not None else "-"


def print_models(models: list[dict[str, Any]], provider: str) -> None:
    if provider == GIGACHAT_PROVIDER:
        rows = [
            (str(m.get("id", "")), str(m.get("type", "")), str(m.get("owned_by", "")))
            for m in models
        ]
        print_table(("ID", "Тип", "Владелец"), rows)
        return
    rows = [
        (
            str(m.get("id", "")),
            model_column(m.get("owned_by")),
            model_column(m.get("context_window")),
            model_column(m.get("max_output_tokens")),
        )
        for m in models
    ]
    print_table(("ID", "Владелец", "Контекст", "Макс. ответ"), rows)


def print_balance(balance: list[dict[str, Any]]) -> None:
    rows = [
        (str(item.get("usage", "")), f"{int(item.get('value') or 0):,}".replace(",", " "))
        for item in balance
    ]
    print_table(("Модель", "Остаток токенов"), rows)


def extract_answer(response: dict[str, Any], provider: str) -> str:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AppError(f"Ответ {provider} без choices/message: {response}") from exc
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        # Некоторые модели отдают content массивом частей вида {"type": "text", "text": ...}.
        content = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return str(content or "").strip()


def extract_finish_reason(response: dict[str, Any]) -> str:
    """Причина остановки генерации: по ней видно, сработал ли max_tokens."""
    try:
        reason = response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return ""
    return str(reason or "")


def describe_usage(response: dict[str, Any]) -> str:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return ""
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    return (
        f"Токены: запрос {prompt_tokens}, ответ {completion_tokens}, всего {total_tokens}"
    )


def select_provider(providers: list[str]) -> str:
    return ask_choice("\nКуда отправить запрос? Выберите провайдера:", providers)


@dataclass(frozen=True)
class RunCase:
    """Один прогон запроса: какие параметры в нём участвуют."""

    number: int
    title: str
    with_limits: bool
    with_schema: bool


RUN_CASES = (
    RunCase(1, "без ограничений", with_limits=False, with_schema=False),
    RunCase(2, "с ограничениями, без строгой схемы", with_limits=True, with_schema=False),
    RunCase(3, "со строгой схемой, без ограничений", with_limits=False, with_schema=True),
    RunCase(4, "с ограничениями и строгой схемой", with_limits=True, with_schema=True),
)

# Проверено запросами к API: GigaChat принимает stop, но не применяет его,
# а строгую схему в v1/chat/completions не поддерживает. Для stop прогон всё равно
# отправляется, а о потере смысла сообщается после ответа.
UNSUPPORTED: dict[str, dict[str, str]] = {
    GIGACHAT_PROVIDER: {
        "schema": "GigaChat API не поддерживает строгую JSON-схему на этом тарифе \
(в документации схема описана для v2/chat/completions, она требует B2B-доступа)",
    },
    VEAI_PROVIDER: {},
}

# Параметры, которые провайдер принимает, но не применяет: прогон отправляется,
# а после ответа печатается предупреждение, что часть смысла потеряна.
IGNORED_PARAMS: dict[str, dict[str, str]] = {
    GIGACHAT_PROVIDER: {
        "stop": "GigaChat API принимает stop, но не применяет его: ответ по stop не обрывается. \
Действовал только max_tokens",
    },
    VEAI_PROVIDER: {},
}


def build_prompt(
    base_prompt: str, params: RequestParams, case: RunCase, provider: str
) -> str:
    """Собирает промпт прогона: описание формата есть везде, схема — где просят."""
    parts = [base_prompt]
    if case.with_schema and provider == GIGACHAT_PROVIDER:
        # Схему GigaChat не принимает, поэтому передаём её словами.
        parts.append(params.schema_note())
    parts.append(params.format_note())
    return "\n\n".join(parts)


def unsupported_reason(
    provider: str, params: RequestParams, case: RunCase
) -> str | None:
    reasons = UNSUPPORTED.get(provider, {})
    if case.with_schema and "schema" in reasons:
        return reasons["schema"]
    return None


def ignored_warnings(provider: str, params: RequestParams, case: RunCase) -> list[str]:
    notes: list[str] = []
    if case.with_limits and params.stop:
        reason = IGNORED_PARAMS.get(provider, {}).get("stop")
        if reason:
            notes.append(reason)
    return notes


def check_json(text: str) -> str:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        hint = ""
        if text.startswith("```"):
            hint = " (модель обернула ответ в markdown-блок ```)"
        return f"Нет: {exc.msg} (позиция {exc.pos}){hint}"
    return "Да, ответ — валидный JSON"


def send_request(
    provider: str,
    model: str,
    prompt: str,
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
    params: RequestParams,
    case: RunCase,
) -> dict[str, Any]:
    max_tokens = params.max_tokens if case.with_limits else None
    stop = params.stop if case.with_limits else None
    try:
        if provider == GIGACHAT_PROVIDER:
            token = gigachat.get_access_token(use_cache=use_cache)
            return gigachat.chat(token, model, prompt, max_tokens=max_tokens, stop=stop)
        return veai.chat(
            model,
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            json_schema=params.json_schema() if case.with_schema else None,
        )
    except (GigaChatError, VeaiError) as exc:
        raise AppError(f"{provider}: {exc}") from exc


def run_cases(
    provider: str,
    model: str,
    prompt: str,
    params: RequestParams,
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
) -> None:
    for case in RUN_CASES:
        print("\n" + "=" * 78)
        print(f"Прогон {case.number} из {len(RUN_CASES)}: {case.title}")
        print("=" * 78)

        reason = unsupported_reason(provider, params, case)
        if reason is not None:
            print(f"Не поддерживается: {reason}")
            continue

        full_prompt = build_prompt(prompt, params, case, provider)
        print(f"Отправка запроса в {provider} ({model})...")
        started = time.time()
        response = send_request(
            provider, model, full_prompt, gigachat, veai, use_cache, params, case
        )
        elapsed = time.time() - started

        answer = extract_answer(response, provider)
        finish_reason = extract_finish_reason(response)
        print(f"\nОтвет ({response.get('model') or model}), {elapsed:.1f} с:")
        print(answer or "(пустой ответ)")
        usage = describe_usage(response)
        if usage:
            print(f"\n{usage}")
        if answer:
            print(f"\nВалидный JSON: {check_json(answer)}")
        else:
            print("\nВалидный JSON: нет — ответ пустой")
        if finish_reason:
            print(f"finish_reason: {finish_reason}")
        for note in ignored_warnings(provider, params, case):
            print(f"\nВНИМАНИЕ: {note}")


def list_all_models(
    gigachat: GigaChatClient, veai: VeaiClient, use_cache: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Exception | None, Exception | None]:
    """Собирает модели обоих провайдеров и возвращает ошибки, не прерывая сценарий.

    Если один провайдер недоступен, второй всё равно можно использовать.
    """
    gigachat_models: list[dict[str, Any]] = []
    veai_models: list[dict[str, Any]] = []
    gigachat_error: Exception | None = None
    veai_error: Exception | None = None

    print(f"\n{VEAI_PROVIDER}: запрос списка моделей...")
    try:
        veai_models = veai.list_models()
    except VeaiError as exc:
        veai_error = exc

    print(f"{GIGACHAT_PROVIDER}: авторизация и запрос списка моделей...")
    try:
        token = gigachat.get_access_token(use_cache=use_cache)
        gigachat_models = gigachat.list_models(token)
    except GigaChatError as exc:
        gigachat_error = exc

    return gigachat_models, veai_models, gigachat_error, veai_error


def show_provider_models(
    provider: str,
    gigachat_models: list[dict[str, Any]],
    veai_models: list[dict[str, Any]],
    gigachat_error: Exception | None,
    veai_error: Exception | None,
) -> list[dict[str, Any]]:
    models, error = (
        (gigachat_models, gigachat_error)
        if provider == GIGACHAT_PROVIDER
        else (veai_models, veai_error)
    )
    if error is not None:
        raise AppError(f"{provider} недоступен: {error}")

    available = chat_models(models, provider)
    if not available:
        raise AppError(f"{provider} не вернул ни одной модели для чата.")
    # В алфавитном порядке: иначе модели Veai идут вперемешку семействами.
    available = sorted(available, key=lambda m: str(m.get("id", "")).lower())

    print(f"\n{provider}: моделей для чата — {len(available)}")
    print_models(available, provider)
    return available


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и портит кириллицу.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        env_file, use_cache = parse_args(sys.argv[1:])
        load_env_file(env_file)

        gigachat_credentials = os.environ.get(
            GIGACHAT_CREDENTIALS_ENV
        ) or os.environ.get(GIGACHAT_FALLBACK_CREDENTIALS_ENV)
        if not gigachat_credentials:
            raise AppError(
                f"В {env_file} не найден ключ {GIGACHAT_CREDENTIALS_ENV} "
                f"(или {GIGACHAT_FALLBACK_CREDENTIALS_ENV})."
            )
        veai_key = os.environ.get(VEAI_CREDENTIALS_ENV, "").strip()
        if not veai_key:
            raise AppError(f"В {env_file} не найден ключ {VEAI_CREDENTIALS_ENV}.")

        scope = os.environ.get("GIGACHAT_SCOPE", "").strip() or DEFAULT_GIGACHAT_SCOPE
        gigachat = GigaChatClient(
            credentials=gigachat_credentials,
            scope=scope,
            ca_cert_files=default_ca_cert_files(),
            token_cache_file=TOKEN_CACHE_FILE,
        )
        veai = VeaiClient(api_key=veai_key)

        print(f"Ключи: {env_file}")
        gigachat_models, veai_models, gigachat_error, veai_error = list_all_models(
            gigachat, veai, use_cache
        )

        print(f"\nДоступно моделей: {GIGACHAT_PROVIDER} — {len(gigachat_models)}, "
              f"{VEAI_PROVIDER} — {len(veai_models)}")
        if gigachat_models:
            print(f"\n{GIGACHAT_PROVIDER}:")
            print_models(gigachat_models, GIGACHAT_PROVIDER)
        else:
            print(f"\n{GIGACHAT_PROVIDER}: список не получен — {gigachat_error}")
        if veai_models:
            print(f"\n{VEAI_PROVIDER}:")
            print_models(veai_models, VEAI_PROVIDER)
        else:
            print(f"\n{VEAI_PROVIDER}: список не получен — {veai_error}")

        try:
            balance = gigachat.get_balance(gigachat.get_access_token(use_cache=use_cache))
        except GigaChatError as exc:
            balance = []
            print(f"\nОстаток токенов GigaChat недоступен: {exc}")
        else:
            print("\nОстаток токенов GigaChat:")
            if balance:
                print_balance(balance)
            else:
                print("  API вернул пустой баланс.")

        prompt = ask_text("\n1. Введите текст запроса")
        params = ask_params()
        describe_params(params)

        # Выбор провайдера предлагаем только из тех, кто ответил списком моделей.
        reachable = []
        if gigachat_error is None:
            reachable.append(GIGACHAT_PROVIDER)
        if veai_error is None:
            reachable.append(VEAI_PROVIDER)
        if not reachable:
            raise AppError("Ни один провайдер не ответил, отправлять запрос некуда.")
        if len(reachable) == 1:
            provider = reachable[0]
            print(f"\nДоступен только {provider}, отправляю запрос туда.")
        else:
            provider = select_provider(reachable)
            if provider not in reachable:
                raise AppError(f"{provider} недоступен: список моделей не получен.")

        available = show_provider_models(
            provider, gigachat_models, veai_models, gigachat_error, veai_error
        )
        model_ids = [str(m.get("id", "")) for m in available]
        model = ask_choice(f"\nКакую модель {provider} использовать?", model_ids)

        run_cases(provider, model, prompt, params, gigachat, veai, use_cache)
        print("\nГотово: выполнены все прогоны.")
        return 0
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
