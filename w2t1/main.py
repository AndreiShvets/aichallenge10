"""Простой агент на GigaChat: модель сама выбирает инструмент под запрос.

Сценарий:
1. читает ключ GigaChat и список запросов (по умолчанию — три проверочных);
2. отправляет запрос модели вместе с описанием трёх инструментов;
3. если модель просит вызвать инструмент, выполняет функцию локально, пишет
   вызов в журнал и возвращает результат модели;
4. повторяет шаги 3–4, пока модель не ответит текстом, и печатает ответ.

Инструменты — обычные функции: бросок кости, текущее время, перевод числа
в шестнадцатеричную систему. Модель их не выполняет, она только просит
вызвать функцию с аргументами, а результат уходит обратно в диалог.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gigachat_client import (
    TOKEN_CACHE_FILE,
    GigaChatClient,
    GigaChatError,
    default_ca_cert_files,
)

GIGACHAT_CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"
GIGACHAT_FALLBACK_CREDENTIALS_ENV = "GIGACHAT_API_KEY"
DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
# Самая дешёвая чат-модель GigaChat: 65 ₽ за 1M токенов против 500 у Pro
# и 650 у Max (ai_models_prices.json).
DEFAULT_MODEL = "GigaChat"
PROJECT_DIR = Path(__file__).resolve().parent
ENV_SEARCH_START = PROJECT_DIR
# Результаты пишутся в корень проектов, рядом с tasks.json.
OUTPUT_DIR = PROJECT_DIR.parent
DEFAULT_LOG_FILE = PROJECT_DIR / "tool_calls.log"

# Запросов на один вопрос. Агент с тремя инструментами укладывается в два шага;
# лимит нужен, чтобы модель, зациклившаяся на вызовах, не тратила токены.
MAX_STEPS = 6

TIME_FORMAT = "%Y-%m-%d %H:%M:%S %Z"
SYSTEM_PROMPT = (
    "Ты агент с инструментами. Если для ответа нужен случайный бросок, текущее "
    "время или перевод числа в шестнадцатеричную систему, вызови подходящую "
    "функцию. Не выдумывай результаты броска и не считай системы счисления "
    "сам — вызывай функцию. Отвечай кратко, по-русски, называя полученное число."
)

# Три проверки агента: без инструментов, с одним и с двумя инструментами подряд.
DEFAULT_QUERIES = (
    "Сколько будет 17 умножить на 23? Ответь одним числом.",
    "Брось игральную кость с 20 гранями и скажи, что выпало.",
    "Брось игральную кость с 8 гранями и переведи выпавшее число "
    "в шестнадцатеричную систему счисления.",
)


class AppError(RuntimeError):
    """Ошибка сценария: неверный ввод, ключ или отказ GigaChat."""


@dataclass(frozen=True)
class Tool:
    """Инструмент агента — обычная функция без данных о себе внутри.

    name, description и parameters уходят в запрос к модели, поэтому описание
    живёт рядом с функцией и правится в одном месте.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def roll_dice(sides: int) -> int:
    """Случайное число от 1 до sides включительно."""
    if sides < 1:
        raise ValueError(f"У кости должно быть хотя бы 1 грань, получено {sides}")
    return random.randint(1, sides)


def current_time() -> str:
    """Текущие дата и время пользователя в его часовом поясе."""
    return datetime.now().astimezone().strftime(TIME_FORMAT)


def decimal_to_hex(number: int) -> str:
    """Шестнадцатеричная запись целого числа, например 255 -> 0xff."""
    return hex(number)


TOOLS = (
    Tool(
        "roll_dice",
        "Бросает игральную кость и возвращает случайное число от 1 до числа "
        "граней включительно.",
        {
            "type": "object",
            "properties": {
                "sides": {
                    "type": "integer",
                    "description": "Количество граней кости, например 6 или 20.",
                }
            },
            "required": ["sides"],
        },
        roll_dice,
    ),
    Tool(
        "current_time",
        "Возвращает текущие дату и время пользователя. Параметров нет.",
        {"type": "object", "properties": {}},
        current_time,
    ),
    Tool(
        "decimal_to_hex",
        "Переводит целое число из десятичной системы счисления "
        "в шестнадцатеричную.",
        {
            "type": "object",
            "properties": {
                "number": {
                    "type": "integer",
                    "description": "Целое число в десятичной системе, например 255.",
                }
            },
            "required": ["number"],
        },
        decimal_to_hex,
    ),
)
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


@dataclass(frozen=True)
class CallLogEntry:
    at: str
    query: str
    step: int
    tool: str
    arguments: str
    result: str
    error: str = ""


@dataclass(frozen=True)
class Step:
    """Один шаг агента: либо просьба вызвать инструмент, либо готовый ответ."""

    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    text: str = ""


@dataclass(frozen=True)
class AgentRun:
    query: str
    steps: list[Step]
    answer: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def tools(self) -> list[str]:
        return [step.tool for step in self.steps]


@dataclass(frozen=True)
class Options:
    env_path: Path
    queries: list[str]
    model: str
    log_file: Path
    use_cache: bool


def find_env_file(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise AppError(f"Файл .env не найден ни в {start}, ни в его родительских папках.")


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


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError as exc:
        raise AppError("Ввод прерван: не получен запрос.") from exc


def parse_args(argv: list[str]) -> Options:
    env_path: Path | None = None
    query: str | None = None
    model = DEFAULT_MODEL
    log_file = DEFAULT_LOG_FILE
    use_cache = True
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--no-cache":
            use_cache = False
        elif arg in ("--env", "--query", "--model", "--log"):
            if not args:
                raise AppError(f"После {arg} требуется значение.")
            value = args.pop(0)
            if arg == "--env":
                env_path = Path(value).expanduser()
            elif arg == "--query":
                query = value
            elif arg == "--model":
                model = value
            else:
                log_file = Path(value).expanduser()
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env=") :]).expanduser()
        elif arg.startswith("--query="):
            query = arg[len("--query=") :]
        elif arg.startswith("--model="):
            model = arg[len("--model=") :]
        elif arg.startswith("--log="):
            log_file = Path(arg[len("--log=") :]).expanduser()
        else:
            raise AppError(
                f"Неизвестный аргумент: {arg}. Доступны: --env PATH, --query TEXT, "
                "--model NAME, --log PATH, --no-cache."
            )
    queries = [query] if query else list(DEFAULT_QUERIES)
    return Options(
        env_path or find_env_file(ENV_SEARCH_START), queries, model, log_file, use_cache
    )


def log_call(entry: CallLogEntry, log_file: Path) -> None:
    """Дописывает вызов инструмента в журнал, по строке на вызов."""
    line = json.dumps(
        {
            "time": entry.at,
            "query": entry.query,
            "step": entry.step,
            "tool": entry.tool,
            "arguments": entry.arguments,
            "result": entry.result,
            "error": entry.error,
        },
        ensure_ascii=False,
    )
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except OSError as exc:
        raise AppError(f"Не удалось записать журнал вызовов {log_file}: {exc}") from exc


def call_tool(tool: Tool, arguments: dict[str, Any], query: str, step: int,
              log_file: Path) -> Step:
    """Выполняет инструмент и пишет вызов в журнал — и успешный, и неуспешный."""
    at = datetime.now().astimezone().strftime(TIME_FORMAT)
    try:
        result = tool.function(**arguments)
    except (TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_call(CallLogEntry(at, query, step, tool.name, _format_arguments(arguments),
                              "", error), log_file)
        return Step(tool=tool.name, arguments=arguments, error=error)
    text = str(result)
    log_call(CallLogEntry(at, query, step, tool.name, _format_arguments(arguments), text),
             log_file)
    return Step(tool=tool.name, arguments=arguments, result=text)


def _format_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False) if arguments else "—"


def _function_results(step: Step) -> str:
    """Результат функции для модели.

    API разбирает content сообщения роли function как JSON и отвечает 422
    на голую строку вроде `0xff`, поэтому результат заворачивается в объект.
    """
    if step.error:
        return json.dumps({"error": step.error}, ensure_ascii=False)
    return json.dumps({"result": step.result}, ensure_ascii=False)


def _tokens(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage") or {}
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def run_agent(
    query: str,
    client: GigaChatClient,
    access_token: str,
    model: str,
    log_file: Path,
) -> AgentRun:
    """Цикл агента: запрос → вызов инструментов → ответ, не больше MAX_STEPS шагов."""
    started = time.time()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    steps: list[Step] = []
    prompt_tokens = completion_tokens = 0

    for step_number in range(1, MAX_STEPS + 1):
        body = {
            "model": model,
            "messages": messages,
            "functions": [tool.schema() for tool in TOOLS],
            "function_call": "auto",
        }
        try:
            response = client.chat_messages(access_token, body)
        except GigaChatError as exc:
            raise AppError(f"GigaChat не ответил: {exc}") from exc
        sent, received = _tokens(response)
        prompt_tokens += sent
        completion_tokens += received

        choices = response.get("choices") or []
        if not choices:
            raise AppError(f"Ответ GigaChat без choices: {response}")
        message = choices[0].get("message") or {}
        function_call = message.get("function_call")
        if not function_call:
            return AgentRun(
                query, steps, (message.get("content") or "").strip(),
                time.time() - started, prompt_tokens, completion_tokens,
            )

        name = str(function_call.get("name", ""))
        raw_arguments = function_call.get("arguments") or {}
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            # Неизвестную функцию выполнять нечем, но и ронять прогон не за что:
            # говорим модели об ошибке и даём ей шанс ответить иначе.
            step = Step(tool=name, arguments=raw_arguments,
                        error=f"Инструмент {name} не найден")
            log_call(CallLogEntry(
                datetime.now().astimezone().strftime(TIME_FORMAT), query, step_number,
                name, _format_arguments(raw_arguments), "", step.error,
            ), log_file)
        elif not isinstance(raw_arguments, dict):
            step = Step(tool=name, error="Аргументы функции — не объект JSON")
        else:
            step = call_tool(tool, raw_arguments, query, step_number, log_file)
        steps.append(step)

        messages.append(
            {"role": "assistant", "content": "", "function_call": function_call}
        )
        messages.append(
            {"role": "function", "name": name, "content": _function_results(step)}
        )

    # Лимит шагов выбирался, чтобы не платить за зациклившиеся вызовы, но сам
    # по себе он не ошибка: показываем, на чём агент остановился.
    return AgentRun(
        query, steps, "", time.time() - started, prompt_tokens, completion_tokens
    )


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


def print_run(position: int, total: int, run: AgentRun) -> None:
    print(f"\n[{position}/{total}] {run.query}")
    if not run.steps:
        print("  инструменты: не вызывались")
    for number, step in enumerate(run.steps, start=1):
        outcome = step.error or step.result
        print(
            f"  шаг {number}: {step.tool}({_format_arguments(step.arguments)})"
            f" → {'ошибка: ' if step.error else ''}{outcome}"
        )
    if run.answer:
        print(f"  ответ: {run.answer}")
    else:
        print(f"  ответ: не получен за {MAX_STEPS} шагов")
    print(
        f"  время: {run.seconds:.1f} с, токены: {run.prompt_tokens} вход + "
        f"{run.completion_tokens} выход"
    )


def print_summary(runs: list[AgentRun]) -> None:
    print("\n" + "=" * 78)
    print("ИТОГ: какой запрос обошелся без инструментов, а какой вызвал их")
    print("=" * 78)
    rows: list[tuple[str, ...]] = []
    for number, run in enumerate(runs, start=1):
        rows.append((
            str(number),
            _shorten(run.query, 52),
            str(len(run.steps)),
            ", ".join(run.tools) or "—",
            "да" if run.answer else "нет",
        ))
    print_table(("#", "Запрос", "Вызовов", "Инструменты", "Ответ"), rows)
    # Три проверочных запроса идут первыми: ожидаемое число вызовов видно
    # сразу, без чтения каждого блока.
    expected = [0, 1, 2]
    checks = min(len(runs), len(expected))
    if checks:
        print()
    for number in range(1, checks + 1):
        run, want = runs[number - 1], expected[number - 1]
        mark = "+" if len(run.steps) == want else "−"
        print(f"  {mark} запрос {number}: ожидалось вызовов {want}, "
              f"получено {len(run.steps)}")
    if len(runs) > checks:
        print(f"  без сверки: запросы {checks + 1}–{len(runs)}")


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def export(runs: list[AgentRun], options: Options, base: Path) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = OUTPUT_DIR / f"{base}_{stamp}.json"
    csv_path = OUTPUT_DIR / f"{base}_{stamp}.csv"
    payload = {
        "meta": {
            "model": options.model,
            "max_steps": MAX_STEPS,
            "tools": [tool.schema() for tool in TOOLS],
            "log_file": str(options.log_file),
            "created": datetime.now().astimezone().strftime(TIME_FORMAT),
        },
        "results": [
            {
                "query": run.query,
                "answer": run.answer,
                "tools": run.tools,
                "seconds": round(run.seconds, 2),
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "steps": [
                    {
                        "tool": step.tool,
                        "arguments": step.arguments,
                        "result": step.result,
                        "error": step.error,
                    }
                    for step in run.steps
                ],
            }
            for run in runs
        ],
    }
    try:
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(
                ("Запрос", "Инструменты", "Вызовов", "Ответ", "Секунд",
                 "Вход", "Выход", "Ошибка")
            )
            for run in runs:
                errors = "; ".join(step.error for step in run.steps if step.error)
                writer.writerow(
                    (run.query, ", ".join(run.tools), len(run.steps), run.answer,
                     f"{run.seconds:.1f}", run.prompt_tokens,
                     run.completion_tokens, errors)
                )
    except OSError as exc:
        raise AppError(f"Не удалось сохранить результаты: {exc}") from exc
    return json_path, csv_path


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и портит кириллицу.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        options = parse_args(sys.argv[1:])
        load_env_file(options.env_path)
        credentials = os.environ.get(
            GIGACHAT_CREDENTIALS_ENV
        ) or os.environ.get(GIGACHAT_FALLBACK_CREDENTIALS_ENV)
        if not credentials:
            raise AppError(
                f"В {options.env_path} не найден ключ {GIGACHAT_CREDENTIALS_ENV} "
                f"(или {GIGACHAT_FALLBACK_CREDENTIALS_ENV})."
            )
        scope = os.environ.get("GIGACHAT_SCOPE", "").strip() or DEFAULT_GIGACHAT_SCOPE
        client = GigaChatClient(
            credentials=credentials,
            scope=scope,
            ca_cert_files=default_ca_cert_files(),
            token_cache_file=TOKEN_CACHE_FILE,
        )

        print(f"Ключи: {options.env_path}")
        print(f"Модель: {options.model}")
        print("Инструменты: " + ", ".join(tool.name for tool in TOOLS))
        print(f"Журнал вызовов: {options.log_file}")
        print(f"Запросов: {len(options.queries)}")

        access_token = client.get_access_token(options.use_cache)
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except GigaChatError as exc:
        print(f"Ошибка авторизации: {exc}", file=sys.stderr)
        return 1

    runs: list[AgentRun] = []
    for number, query in enumerate(options.queries, start=1):
        try:
            run = run_agent(
                query, client, access_token, options.model, options.log_file
            )
        except AppError as exc:
            # Ошибка одного запроса не роняет прогон: она видна в отчёте,
            # остальные запросы выполняются дальше.
            print(f"\n[{number}/{len(options.queries)}] {query}")
            print(f"  ошибка: {exc}")
            runs.append(AgentRun(query, [], "", 0.0))
            continue
        print_run(number, len(options.queries), run)
        runs.append(run)

    print_summary(runs)
    try:
        json_path, csv_path = export(runs, options, "w2t1_" + options.model)
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    print(f"\nРезультаты: {json_path.name}, {csv_path.name}")
    print(f"Журнал вызовов: {options.log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
