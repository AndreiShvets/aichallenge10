"""Агент на GigaChat с инструментами и памятью разговора.

Сценарий:
1. читает ключ GigaChat и историю прошлого диалога, если она сохранена;
2. спрашивает запрос у пользователя;
3. отправляет модели всю историю разговора вместе с описанием трёх инструментов;
4. если модель просит вызвать инструмент, выполняет функцию локально, пишет
   вызов в журнал и возвращает результат модели;
5. повторяет шаг 4, пока модель не ответит текстом, печатает ответ и сколько
   раз за цикл понадобился каждый инструмент;
6. дописывает запрос и ответ в историю, сохраняет её на диск и спрашивает
   следующий запрос — до команды /exit; /clear стирает историю с диска.

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
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gigachat_client import (
    TOKEN_CACHE_FILE,
    GigaChatClient,
    GigaChatError,
    HttpLogger,
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
# Журналы: вызовы инструментов и HTTP-обмены с GigaChat. Оба копятся между
# запусками; в git не попадают — это результаты работы, а не исходники.
DEFAULT_LOG_FILE = PROJECT_DIR / "tool_calls.log"
DEFAULT_HTTP_LOG_FILE = PROJECT_DIR / "http.log"
# история разговора: без неё каждый запуск начинал бы беседу заново.
DEFAULT_HISTORY_FILE = PROJECT_DIR / "history.json"
# Версия формата файла истории: по ней видно, когда формат менялся.
HISTORY_VERSION = 1
# Отступ в JSON-записи HTTP-лога: один обмен читается глазами, вложенные
# сообщения и тела ответов стоят на своих строках.
HTTP_LOG_INDENT = 2

# Запросов на один вопрос. Агент с тремя инструментами укладывается в два шага;
# лимит нужен, чтобы модель, зациклившаяся на вызовах, не тратила токены.
MAX_STEPS = 6
# Команды диалога.
EXIT_COMMAND = "/exit"
CLEAR_COMMAND = "/clear"

TIME_FORMAT = "%Y-%m-%d %H:%M:%S %Z"
SYSTEM_PROMPT = (
    "Ты агент с инструментами. Если для ответа нужен случайный бросок, текущее "
    "время или перевод числа в шестнадцатеричную систему, вызови подходящую "
    "функцию. Не выдумывай результаты броска и не считай системы счисления "
    "сам — вызывай функцию. Отвечай кратко, по-русски, называя полученное число."
)

# Три проверки агента: без инструментов, с одним и с двумя инструментами подряд.
# На них сохранены выводы прошлого прогона и по ним проверяется работа диалога.
EXAMPLE_QUERIES = (
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
class ChatTurn:
    """Один обмен разговора: что спросил пользователь и что ответил агент.

    Шаги сохраняются только для отчёта, в модель они не уходят: модели нужен
    лишь текст прошлых сообщений, а просьбы вызвать функцию в историю
    не переносятся — они относились к своему запросу.
    """

    query: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    at: str = ""

    @property
    def tools(self) -> list[str]:
        return [step.tool for step in self.steps]


@dataclass(frozen=True)
class History:
    """История разговора: начало беседы и все её обмены по порядку."""

    started: str
    turns: list[ChatTurn] = field(default_factory=list)

    def messages(self) -> list[dict[str, Any]]:
        """История в виде сообщений для API: пара user/assistant на каждый обмен."""
        messages: list[dict[str, Any]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.query})
            if turn.answer:
                messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def extended(self, turn: ChatTurn) -> History:
        return History(self.started, [*self.turns, turn])


@dataclass(frozen=True)
class Options:
    env_path: Path
    model: str
    log_file: Path
    http_log_file: Path
    history_file: Path
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
    model = DEFAULT_MODEL
    log_file = DEFAULT_LOG_FILE
    http_log_file = DEFAULT_HTTP_LOG_FILE
    history_file = DEFAULT_HISTORY_FILE
    use_cache = True
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--no-cache":
            use_cache = False
        elif arg in ("--env", "--model", "--log", "--http-log", "--history"):
            if not args:
                raise AppError(f"После {arg} требуется значение.")
            value = args.pop(0)
            if arg == "--env":
                env_path = Path(value).expanduser()
            elif arg == "--model":
                model = value
            elif arg == "--log":
                log_file = Path(value).expanduser()
            elif arg == "--history":
                history_file = Path(value).expanduser()
            else:
                http_log_file = Path(value).expanduser()
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env=") :]).expanduser()
        elif arg.startswith("--model="):
            model = arg[len("--model=") :]
        elif arg.startswith("--log="):
            log_file = Path(arg[len("--log=") :]).expanduser()
        elif arg.startswith("--history="):
            history_file = Path(arg[len("--history=") :]).expanduser()
        elif arg.startswith("--http-log="):
            http_log_file = Path(arg[len("--http-log=") :]).expanduser()
        else:
            raise AppError(
                f"Неизвестный аргумент: {arg}. Доступны: --env PATH, "
                "--model NAME, --log PATH, --http-log PATH, --history PATH, "
                "--no-cache."
            )
    return Options(
        env_path or find_env_file(ENV_SEARCH_START), model, log_file,
        http_log_file, history_file, use_cache,
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


def http_logger(log_file: Path) -> HttpLogger:
    """Создаёт логгер HTTP-обменов для клиента GigaChat.

    В отличие от журнала инструментов, сбой записи не роняет запрос к API:
    без лога ответ пользователю всё равно нужен. Предупреждение печатается
    один раз, чтобы цикл агента не утонул в одинаковых строках.
    """
    warned = False

    def log(entry: dict[str, Any]) -> None:
        nonlocal warned
        # Один обмен — один JSON-объект с отступами: файл читается глазами
        # (тела сообщений и ответов стоят на своих строках), но остаётся
        # склеиваемым потоком объектов, а не одним массивом.
        record = json.dumps({
            "time": entry["time"],
            "query_number": entry.get("query_number"),
            "step": entry.get("step"),
            "method": entry["method"],
            "url": entry["url"],
            "request_headers": entry["request_headers"],
            "request_body": _parse_json(entry["request_body"]),
            "status": entry["status"],
            "response_body": _parse_json(entry["response_body"]),
            "seconds": entry["seconds"],
            "error": entry["error"],
        }, ensure_ascii=False, indent=HTTP_LOG_INDENT)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write(record + "\n")
        except OSError as exc:
            if not warned:
                warned = True
                print(f"Предупреждение: не удалось записать {log_file}: {exc}",
                      file=sys.stderr)

    return log


def _parse_json(text: str) -> Any:
    """Тело для лога в виде объекта, если это JSON, иначе — строкой."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


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
    history: History,
    on_step: Callable[[Step, int], None] | None = None,
    context: tuple[int, str] = (0, ""),
) -> AgentRun:
    """Цикл агента: запрос → вызов инструментов → ответ, не больше MAX_STEPS шагов.

    history — прошлые обмены разговора: они уходят модели перед новым запросом,
    поэтому она продолжает тему, а не начинает беседу заново.
    on_step вызывается на каждый выполненный инструмент: по нему вызывающий
    код показывает шаги по ходу работы, а не только после ответа.
    context — номер запроса в сеансе и текст запроса: они уходят в HTTP-лог,
    чтобы в нём было видно, какой шаг какого диалога описывает строка.
    """
    query_number, user_query = context
    started = time.time()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history.messages(),
        {"role": "user", "content": user_query},
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
        client.log_context = {
            "query_number": query_number,
            "query": user_query,
            "step": step_number,
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
        if on_step is not None:
            on_step(step, step_number)

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


def print_event(step: Step, number: int) -> None:
    """Строка о вызове инструмента по ходу цикла."""
    print(
        f"  шаг {number}: {step.tool}({_format_arguments(step.arguments)})"
        f" → {'ошибка: ' if step.error else ''}{step.error or step.result}",
        flush=True,
    )


def print_answer(run: AgentRun) -> None:
    """Ответ и сколько раз за цикл был использован каждый инструмент."""
    if not run.steps:
        print("  инструменты: не вызывались")
    if run.answer:
        print(f"\nОтвет: {run.answer}")
    else:
        print(f"\nОтвет: не получен за {MAX_STEPS} шагов")
    counts = Counter(run.tools)
    print("Использовано инструментов:")
    for tool in TOOLS:
        print(f"  {tool.name}: {counts.get(tool.name, 0)}")
    other = sorted(set(counts) - {tool.name for tool in TOOLS})
    for name in other:
        print(f"  {name} (нет такого инструмента): {counts[name]}")
    print(
        f"Время: {run.seconds:.1f} с, токены: {run.prompt_tokens} вход + "
        f"{run.completion_tokens} выход"
    )


def print_help(history: History) -> None:
    print(f"Введите запрос и нажмите Enter. {EXIT_COMMAND} — выход, "
          f"{CLEAR_COMMAND} — забыть разговор.")
    print("Например:")
    for query in EXAMPLE_QUERIES:
        print(f"  {query}")
    if history.turns:
        print(f"Продолжается разговор из {len(history.turns)} обменов "
              f"(начат {history.started}).")


def save_history(history: History, history_file: Path) -> None:
    """Пишет историю разговора целиком: файл — состояние, а не журнал."""
    payload = {
        "version": HISTORY_VERSION,
        "started": history.started,
        "saved": datetime.now().astimezone().strftime(TIME_FORMAT),
        "turns": [
            {
                "at": turn.at,
                "query": turn.query,
                "answer": turn.answer,
                "tools": turn.tools,
                "steps": [
                    {
                        "tool": step.tool,
                        "arguments": step.arguments,
                        "result": step.result,
                        "error": step.error,
                    }
                    for step in turn.steps
                ],
            }
            for turn in history.turns
        ],
    }
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AppError(f"Не удалось сохранить историю {history_file}: {exc}") from exc


def load_history(history_file: Path) -> History:
    """Читает сохранённый разговор; битый файл — ошибка, а не тихая потеря."""
    if not history_file.is_file():
        return History(datetime.now().astimezone().strftime(TIME_FORMAT))
    try:
        payload = json.loads(history_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError(
            f"Не удалось прочитать историю {history_file}: {exc}. "
            f"Удалите файл или введите {CLEAR_COMMAND} после запуска."
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("turns"), list):
        raise AppError(f"Файл истории {history_file} не похож на историю разговора.")
    turns: list[ChatTurn] = []
    for item in payload["turns"]:
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise AppError(f"В истории {history_file} есть запись без запроса.")
        steps = [
            Step(
                tool=str(step.get("tool", "")),
                arguments=step.get("arguments") or {},
                result=str(step.get("result", "")),
                error=str(step.get("error", "")),
            )
            for step in item.get("steps") or []
            if isinstance(step, dict)
        ]
        turns.append(ChatTurn(
            query=item["query"],
            answer=str(item.get("answer", "")),
            steps=steps,
            at=str(item.get("at", "")),
        ))
    return History(str(payload.get("started", "")), turns)


def clear_history(history_file: Path) -> None:
    """Забыть разговор: файла нет — уже забыт."""
    try:
        history_file.unlink(missing_ok=True)
    except OSError as exc:
        raise AppError(f"Не удалось удалить историю {history_file}: {exc}") from exc


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
            http_logger=http_logger(options.http_log_file),
        )

        print(f"Ключи: {options.env_path}")
        print(f"Модель: {options.model}")
        print("Инструменты: " + ", ".join(tool.name for tool in TOOLS))
        print(f"HTTP-лог: {options.http_log_file}")
        print(f"Журнал вызовов: {options.log_file}")
        print(f"История: {options.history_file}")
        history = load_history(options.history_file)

        access_token = client.get_access_token(options.use_cache)
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except GigaChatError as exc:
        print(f"Ошибка авторизации: {exc}", file=sys.stderr)
        return 1

    runs: list[AgentRun] = []
    # Номер запроса в сеансе: считается по всем введённым, включая закончившиеся
    # ошибкой — в HTTP-логе номера должны идти подряд.
    received = 0
    print()
    print_help(history)
    while True:
        try:
            query = ask("\nЗапрос (или /exit): ")
        except AppError as exc:
            # Конец ввода (Ctrl+D, закрытый поток) — тот же выход, что и /exit.
            print(f"\n{exc}")
            break
        command = query.lower()
        if command == EXIT_COMMAND:
            break
        if command == CLEAR_COMMAND:
            try:
                clear_history(options.history_file)
            except AppError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
                continue
            history = History(datetime.now().astimezone().strftime(TIME_FORMAT))
            print("История разговора очищена — следующий запрос начнёт беседу заново.")
            continue
        if not query:
            continue

        received += 1
        print("\nЦикл агента")
        try:
            run = run_agent(
                query, client, access_token, options.model, options.log_file,
                history, on_step=print_event, context=(received, query),
            )
        except AppError as exc:
            # Отказ по одному запросу не закрывает диалог: причину видно сразу,
            # а следующий запрос можно задать снова.
            print(f"  ошибка: {exc}")
            continue
        runs.append(run)
        print_answer(run)
        # Обмен сохраняется сразу после ответа: даже оборванный запуск
        # не теряет уже полученные ответы.
        history = history.extended(ChatTurn(
            query=query,
            answer=run.answer,
            steps=run.steps,
            at=datetime.now().astimezone().strftime(TIME_FORMAT),
        ))
        try:
            save_history(history, options.history_file)
        except AppError as exc:
            print(f"  предупреждение: {exc}", file=sys.stderr)
        print(f"  в истории разговора: {len(history.turns)}")

    return finish(runs, history, options)


def finish(runs: list[AgentRun], history: History, options: Options) -> int:
    """Итог диалога: куда легла история и что с ней будет дальше."""
    print()
    if history.turns:
        print(f"В истории разговора: {len(history.turns)} обменов "
              f"({options.history_file.name})")
        print("Следующий запуск продолжит этот разговор, "
              f"{CLEAR_COMMAND} его сотрёт.")
    else:
        print("История разговора пуста.")
    if not runs:
        print("Запросов не было — сохранять нечего.")
        return 0
    counts: Counter[str] = Counter()
    for run in runs:
        counts.update(run.tools)
    print(f"Запросов за сеанс: {len(runs)}")
    print("Вызовов инструментов за сеанс:")
    for tool in TOOLS:
        print(f"  {tool.name}: {counts.get(tool.name, 0)}")
    try:
        json_path, csv_path = export(runs, options, "w2t2_" + options.model)
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    print(f"Результаты: {json_path.name}, {csv_path.name}")
    print(f"Журнал вызовов: {options.log_file}")
    print(f"HTTP-лог: {options.http_log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
