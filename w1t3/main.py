"""Сравнение моделей на наборе задач: 2 уровня рассуждений × 4 способа решения.

Сценарий:
1. читает задачи из tasks.json (по умолчанию — в корне проектов, на уровень выше);
2. показывает модели GigaChat и Veai и просит выбрать провайдера и модель;
3. для каждой задачи и каждого уровня рассуждений решает её четырьмя способами:
   напрямую, с «решай пошагово», по промпту, который модель составила сама,
   и через группу экспертов;
4. сверяет извлечённый из ответа ответ с эталоном и печатает итоговые таблицы.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
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
# Наборы задач лежат в корне проектов, чтобы ими пользовались и следующие задачи.
DEFAULT_TASKS_FILE = PROJECT_DIR.parent / "tasks.json"
SHORT_TASKS_FILE = PROJECT_DIR.parent / "tasks_short.json"
ENV_SEARCH_START = PROJECT_DIR

# Уровни рассуждений. Значение — что отправлять провайдеру.
# GigaChat принимает reasoning_effort со значениями low/medium/high (max отклоняется
# с HTTP 422 «Unknown reasoning_effort»), поэтому максимальный уровень — high.
GIGACHAT_REASONING_EFFORT = {"низкий": "low", "максимальный": "high"}
VEAI_REASONING_EFFORT = {"низкий": "low", "максимальный": "max"}
REASONING_LEVELS = ("низкий", "максимальный")


class AppError(RuntimeError):
    """Ошибка сценария: неверный ввод, задачи или отказ провайдера."""


@dataclass(frozen=True)
class Task:
    id: str
    topic: str
    difficulty: int
    answer_type: str
    answer: Any
    accepted: tuple[str, ...]
    question: str

    @property
    def expected(self) -> str:
        return str(self.answer)


@dataclass(frozen=True)
class Method:
    key: str
    title: str


METHODS = (
    Method("direct", "напрямую"),
    Method("steps", "решай пошагово"),
    Method("self_prompt", "промпт от модели"),
    Method("experts", "группа экспертов"),
)


@dataclass
class RunResult:
    task_id: str
    reasoning: str
    method: str
    answer: str
    correct: bool
    seconds: float
    error: str = ""
    # Что фактически ушло в API и что вернулось — для проверки гипотез:
    # доходит ли reasoning_effort и работает ли кэш промпта.
    effort: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    precached_tokens: int | None = None
    # Veai возвращает цепочку рассуждений отдельным полем message.reasoning_content;
    # его длина в символах видна в таблице диагностики.
    reasoning_chars: int = 0

    @property
    def status(self) -> str:
        if self.error:
            return "ошибка"
        return "+" if self.correct else "−"


@dataclass(frozen=True)
class ChatReply:
    """Ответ модели вместе с метаданными, которые нужно показать в таблице."""

    content: str
    effort: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    precached_tokens: int | None = None
    reasoning_chars: int = 0


def find_env_file(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise AppError(f"Файл .env не найден ни в {start}, ни в его родительских папках.")


def parse_args(argv: list[str]) -> tuple[Path, Path | None, bool]:
    """Возвращает (путь к .env, путь к файлу задач или None, использовать ли кэш).

    None вместо пути означает, что набор задач ещё не выбран и его надо спросить
    у пользователя интерактивно.
    """
    env_path: Path | None = None
    tasks_path: Path | None = None
    use_cache = True
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--no-cache":
            use_cache = False
        elif arg == "--full":
            tasks_path = DEFAULT_TASKS_FILE
        elif arg == "--short":
            tasks_path = SHORT_TASKS_FILE
        elif arg in ("--env", "--tasks"):
            if not args:
                raise AppError(f"После {arg} требуется путь.")
            value = Path(args.pop(0)).expanduser()
            if arg == "--env":
                env_path = value
            else:
                tasks_path = value
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env=") :]).expanduser()
        elif arg.startswith("--tasks="):
            tasks_path = Path(arg[len("--tasks=") :]).expanduser()
        else:
            raise AppError(
                f"Неизвестный аргумент: {arg}. Доступны: --env PATH, --tasks PATH, "
                "--full, --short, --no-cache."
            )
    return env_path or find_env_file(ENV_SEARCH_START), tasks_path, use_cache


def choose_tasks_file() -> Path:
    """Спрашивает, какой набор задач запускать. Пустой ввод — полный набор."""
    try:
        short_count = len(load_tasks(SHORT_TASKS_FILE))
    except AppError:
        short_count = 0
    try:
        full_count = len(load_tasks(DEFAULT_TASKS_FILE))
    except AppError:
        full_count = 0

    options: list[tuple[str, Path]] = []
    if short_count:
        options.append((f"сокращённый — {short_count} задач, быстрый ({SHORT_TASKS_FILE.name})", SHORT_TASKS_FILE))
    if full_count:
        options.append((f"полный — {full_count} задач, долгий ({DEFAULT_TASKS_FILE.name})", DEFAULT_TASKS_FILE))
    if not options:
        raise AppError(
            f"Не найдено ни {SHORT_TASKS_FILE.name}, ни {DEFAULT_TASKS_FILE.name}."
        )
    default_is_full = bool(full_count)

    print("\nКакой прогон запускать?")
    for index, (title, _) in enumerate(options, start=1):
        print(f"  {index}. {title}")
    print("  Enter — полный набор" if default_is_full else "  Enter — сокращённый набор")
    while True:
        raw = ask("Ваш выбор: ")
        if not raw:
            return options[-1][1] if default_is_full else options[0][1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print(f"Введите номер от 1 до {len(options)} или нажмите Enter.")


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


def load_tasks(path: Path) -> list[Task]:
    if not path.is_file():
        raise AppError(f"Файл задач не найден: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise AppError(f"{path} не является корректным JSON: {exc}") from exc

    raw_tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise AppError(f"В {path} нет непустого списка tasks.")

    tasks: list[Task] = []
    for index, raw in enumerate(raw_tasks, start=1):
        if not isinstance(raw, dict):
            raise AppError(f"Задача №{index} в {path} описана не объектом.")
        missing = {"id", "question", "answer", "answer_type"} - set(raw)
        if missing:
            raise AppError(f"В задаче №{index} нет полей: {', '.join(sorted(missing))}.")
        accepted = raw.get("accepted") or [str(raw["answer"])]
        tasks.append(
            Task(
                id=str(raw["id"]),
                topic=str(raw.get("topic", "")),
                difficulty=int(raw.get("difficulty", 0)),
                answer_type=str(raw["answer_type"]),
                answer=raw["answer"],
                accepted=tuple(str(item) for item in accepted),
                question=str(raw["question"]),
            )
        )
    return tasks


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return re.sub(r"[\s\-‐‑–—]+", " ", text).strip()


def squeeze(text: str) -> str:
    return re.sub(r"[\s\-‐‑–—]+", "", normalize(text))


# В текстах задач встречаются числа из условия («8-й день», «1/16»), поэтому
# последнее число в ответе не годится: выбираем то, которое выглядит как итог.
def chat_models(models: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    """Оставляет модели, в которые можно отправить текстовый запрос.

    В GigaChat это модели с type == "chat": в общем списке есть Embeddings,
    которая на chat/completions отвечает 404. У Veai типа нет — смотрим модальности.
    """
    if provider == GIGACHAT_PROVIDER:
        return [m for m in models if str(m.get("type", "")) == "chat"]
    return [
        m
        for m in models
        if "text" in (m.get("input_modalities") or ["text"])
        and "text" in (m.get("output_modalities") or ["text"])
    ]


def reasoning_limitation(model: dict[str, Any], provider: str) -> str:
    """Почему на модели нельзя провести полноценный тест по уровню рассуждений.

    Пустая строка означает, что ограничений нет. Речь только о различии уровней:
    на GigaChat reasoning_effort принимается, но на ответ не влияет; у части
    моделей Veai параметра нет вовсе, и оба уровня дают одинаковый запрос.
    """
    if provider == GIGACHAT_PROVIDER:
        return "уровень рассуждений не влияет на ответ"
    supported = (model.get("reasoning") or {}).get("supported_efforts") or []
    needed = set(VEAI_REASONING_EFFORT.values())
    if not needed.issubset(set(supported)):
        return "не поддерживает уровни рассуждений: оба уровня пойдут одним запросом"
    return ""


ANSWER_HINT = re.compile(
    r"(?:ответ|итого|получается|получаем|равно|равен|равна|будет|составит)\D{0,15}?(-?\d+)",
    re.IGNORECASE,
)


def extract_answer(text: str, task: Task) -> str:
    """Достаёт из свободного ответа модели то, что сравнивается с эталоном."""
    if task.answer_type == "int":
        hints = ANSWER_HINT.findall(text)
        if hints:
            return hints[-1]
        numbers = [_strip_digits(item) for item in re.findall(r"-?\d[\d\s\u00a0]*", text)]
        numbers = [item for item in numbers if item]
        candidate = numbers[-1] if numbers else ""
        # Последнее число может оказаться частью дроби вроде 1/16.
        if candidate:
            position = text.rfind(candidate)
            if position > 0 and text[position - 1] in "/₁":
                return ""
        return candidate
    normalized = normalize(text)
    for variant in task.accepted:
        if squeeze(variant) in squeeze(normalized):
            return variant
    return normalized


def _strip_digits(raw: str) -> str:
    return re.sub(r"[\s\u00a0]", "", raw)


def is_correct(extracted: str, task: Task) -> bool:
    """Сверяет извлечённый ответ с эталоном.

    Для числовых задач требуется точное совпадение с эталоном; вариант из accepted
    принимается, только если в нём нет других чисел, иначе «24» из «1/16» считалось
    бы верным. Для текстовых задач достаточно любого варианта из accepted.
    """
    if not extracted:
        return False
    if task.answer_type == "int":
        try:
            if int(extracted) == int(task.expected):
                return True
        except ValueError:
            pass
        for variant in task.accepted:
            variant_numbers = re.findall(r"-?\d+", variant)
            if variant_numbers and [int(n) for n in variant_numbers] == [int(task.expected)]:
                return squeeze(extracted) == squeeze(variant)
        return False
    return squeeze(extracted) == squeeze(task.expected) or squeeze(extracted) in {
        squeeze(variant) for variant in task.accepted
    }


def build_prompt(task: Task, method: str) -> str:
    """Промпт для способов, которые не требуют отдельного запроса за промптом."""
    if method == "direct":
        return task.question
    if method == "steps":
        return f"{task.question}\n\nРешай пошагово, подробно объясняя каждый шаг рассуждения."
    if method == "experts":
        return (
            f"{task.question}\n\n"
            "Реши задачу в роли группы экспертов. Каждый эксперт высказывается по очереди, "
            "обращаясь к предыдущим мнениям:\n"
            "1. Аналитик — разбирает условие и выделяет ключевые данные.\n"
            "2. Инженер — предлагает метод решения и выполняет вычисления.\n"
            "3. Критик — проверяет расчёты инженера, ищет ошибки и ловушки условия.\n"
            "Затем все трое согласуют единый ответ и в самом конце дай итоговый ответ."
        )
    raise AppError(f"Неизвестный способ решения: {method}")


def build_self_prompt_request(task: Task) -> str:
    return (
        "Ты — методист, который готовит промпты для другой модели. "
        "Составь максимально эффективный промпт, с которым модель правильно решит "
        "следующую задачу. Не решай саму задачу, не давай ответ и не пересказывай "
        "условие — верни только текст промпта.\n\n"
        f"Задача:\n{task.question}"
    )


def combine_self_prompt(meta_prompt: str, task: Task) -> str:
    if not meta_prompt.strip():
        return task.question
    return (
        f"{meta_prompt.strip()}\n\n"
        f"Задача, которую нужно решить:\n{task.question}\n\n"
        "Заверши ответ явным итоговым ответом."
    )


def openai_error_hint(exc: Exception) -> str:
    text = str(exc)
    if "HTTP 429" in text:
        return " Исчерпаны минуты подписки Veai."
    if "HTTP 401" in text:
        return " Проверьте ключ."
    return ""


def chat(
    provider: str,
    model: str,
    prompt: str,
    reasoning: str,
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
) -> ChatReply:
    effort = (
        GIGACHAT_REASONING_EFFORT[reasoning]
        if provider == GIGACHAT_PROVIDER
        else VEAI_REASONING_EFFORT[reasoning]
    )
    try:
        if provider == GIGACHAT_PROVIDER:
            token = gigachat.get_access_token(use_cache=use_cache)
            response = gigachat.chat(
                token, model, prompt, reasoning_effort=effort
            )
        else:
            response = veai.chat(model, prompt, reasoning_effort=effort)
    except GigaChatError as exc:
        raise AppError(f"{provider}: {exc}") from exc
    except VeaiError as exc:
        raise AppError(f"{provider}: {exc}{openai_error_hint(exc)}") from exc

    usage = response.get("usage") or {}
    return ChatReply(
        content=extract_content(response, provider),
        effort=effort,
        prompt_tokens=_as_int(usage.get("prompt_tokens")),
        completion_tokens=_as_int(usage.get("completion_tokens")),
        precached_tokens=_as_int(usage.get("precached_prompt_tokens")),
        reasoning_chars=len(_reasoning_text(response)),
    )


def _reasoning_text(response: dict[str, Any]) -> str:
    """Цепочка рассуждений, если провайдер вернул её отдельным полем."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(message, dict):
        return ""
    return str(message.get("reasoning_content") or message.get("reasoning") or "")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_content(response: dict[str, Any], provider: str) -> str:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AppError(f"Ответ {provider} без choices/message: {response}") from exc
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return str(content or "").strip()


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError as exc:
        raise AppError("Ввод прерван: не получены ответы на вопросы.") from exc


def ask_choice(title: str, options: list[str]) -> str:
    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        raw = ask("Ваш выбор: ")
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        # Пользователь может ввести и чистое имя, без пометки в скобках.
        stripped = raw.split(" (")[0].strip()
        for option in options:
            if option.split(" (")[0].strip() == stripped:
                return option
        print(f"Введите номер от 1 до {len(options)} или точное имя.")


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


def run_task(
    task: Task,
    provider: str,
    model: str,
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
) -> list[RunResult]:
    results: list[RunResult] = []
    total = len(REASONING_LEVELS) * len(METHODS)
    done = 0
    for reasoning in REASONING_LEVELS:
        # Способ «промпт от модели» требует отдельного запроса, он общий для задачи.
        self_prompt = ""
        meta_tokens: tuple[int | None, int | None] = (None, None)
        try:
            meta = chat(
                provider,
                model,
                build_self_prompt_request(task),
                reasoning,
                gigachat,
                veai,
                use_cache,
            )
            self_prompt = meta.content
            # Токены мета-запроса тоже тратятся, поэтому входят в ячейку способа.
            meta_tokens = (meta.prompt_tokens, meta.completion_tokens)
        except AppError as exc:
            self_prompt = ""
            errors_note = str(exc)
        else:
            errors_note = ""

        for method in METHODS:
            done += 1
            print(
                f"  [{done}/{total}] {reasoning} / {method.title}…",
                end="",
                flush=True,
            )
            started = time.time()
            if method.key == "self_prompt" and errors_note:
                elapsed = time.time() - started
                print(f" ошибка: {errors_note}")
                results.append(
                    RunResult(task.id, reasoning, method.key, "", False, elapsed, errors_note)
                )
                continue
            prompt = (
                combine_self_prompt(self_prompt, task)
                if method.key == "self_prompt"
                else build_prompt(task, method.key)
            )
            try:
                reply = chat(
                    provider, model, prompt, reasoning, gigachat, veai, use_cache
                )
            except AppError as exc:
                elapsed = time.time() - started
                print(f" ошибка: {exc}")
                results.append(
                    RunResult(
                        task.id,
                        reasoning,
                        method.key,
                        "",
                        False,
                        elapsed,
                        str(exc),
                        GIGACHAT_REASONING_EFFORT[reasoning]
                        if provider == GIGACHAT_PROVIDER
                        else VEAI_REASONING_EFFORT[reasoning],
                    )
                )
                continue
            elapsed = time.time() - started
            extracted = extract_answer(reply.content, task)
            correct = is_correct(extracted, task)
            # В ячейку «промпт от модели» включаем и мета-запрос: он тоже тратит токены.
            prompt_tokens = _add_tokens(reply.prompt_tokens, meta_tokens[0])
            completion_tokens = _add_tokens(reply.completion_tokens, meta_tokens[1])
            print(
                f" {'верно' if correct else 'неверно'} "
                f"(ответ: {extracted or '—'}, ожидался {task.expected}, {elapsed:.1f} с, "
                f"effort={reply.effort}, pt={prompt_tokens or 0}, "
                f"ct={completion_tokens or 0}, precached={reply.precached_tokens or 0})"
            )
            results.append(
                RunResult(
                    task.id,
                    reasoning,
                    method.key,
                    extracted,
                    correct,
                    elapsed,
                    effort=reply.effort,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    precached_tokens=reply.precached_tokens,
                    reasoning_chars=reply.reasoning_chars,
                )
            )
    return results


def _add_tokens(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return first + second


def _status_cell(result: RunResult) -> str:
    """Один знак на ячейку таблицы. Ошибка запроса — «!», а не слово «ошибка»,
    иначе четыре подряд идущие ошибки слипаются в нечитаемую строку."""
    if result.error:
        return "!"
    return "+" if result.correct else "−"


def print_summary(
    tasks: list[Task],
    results: list[RunResult],
    provider: str,
    model: str,
    limitation: str = "",
) -> None:
    print("\n" + "=" * 78)
    print(f"ИТОГИ: {provider}, модель {model}, задач {len(tasks)}")
    print("=" * 78)
    if limitation:
        print(
            f"ВНИМАНИЕ: {limitation}.\n"
            "Сравнение уровней рассуждений на этой модели смысла не имеет: "
            "ячейки «низкий» и «максимальный» — один и тот же запрос к API.\n"
        )
    rows: list[tuple[str, ...]] = []
    max_correct = len(REASONING_LEVELS) * len(METHODS)
    for task in tasks:
        cells = [task.id]
        for reasoning in REASONING_LEVELS:
            cell = ""
            for method in METHODS:
                found = [
                    r
                    for r in results
                    if r.task_id == task.id
                    and r.reasoning == reasoning
                    and r.method == method.key
                ]
                cell += _status_cell(found[0]) if found else "?"
            cells.append(cell)
        task_results = [r for r in results if r.task_id == task.id]
        correct = sum(1 for r in task_results if r.correct)
        cells.append(f"{correct}/{max_correct}")
        rows.append(tuple(cells))

    header = ["Задача"] + [f"{reasoning[:5]}" for reasoning in REASONING_LEVELS] + ["Верно"]
    print(
        "\nЗнаки: + верно, − неверно, ! ошибка запроса, ? нет данных; "
        "порядок способов: напрямую, пошагово, промпт от модели, эксперты\n"
    )
    print_table(tuple(header), rows)

    print("\nПо способам решения:")
    method_rows = []
    for method in METHODS:
        cells = []
        for reasoning in REASONING_LEVELS:
            matched = [
                r for r in results if r.reasoning == reasoning and r.method == method.key
            ]
            cells.append(f"{sum(1 for r in matched if r.correct)}/{len(matched)}")
        total = [r for r in results if r.method == method.key]
        method_rows.append((method.title, *cells, f"{sum(1 for r in total if r.correct)}/{len(total)}"))
    print_table(("Способ", *REASONING_LEVELS, "Всего"), method_rows)

    print_method_topic_matrix(tasks, results)
    print_token_stats(tasks, results)
    print_diagnostics(results)

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nОшибок при запросах: {len(errors)}")
        seen: set[str] = set()
        for error in errors:
            if error.error in seen:
                continue
            seen.add(error.error)
            print(f"  {error.error}")


def _format_int(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}".replace(",", " ")


def print_method_topic_matrix(tasks: list[Task], results: list[RunResult]) -> None:
    """Матрица «способ × тема»: какой способ лучше на каком типе задач."""
    topics = sorted({task.topic for task in tasks if task.topic})
    if not topics:
        return
    topic_of = {task.id: task.topic for task in tasks}
    print("\nВерных ответов: способ × тема (оба уровня вместе):")
    rows = []
    for method in METHODS:
        cells = []
        for topic in topics:
            matched = [
                r
                for r in results
                if r.method == method.key and topic_of.get(r.task_id) == topic
            ]
            cells.append(f"{sum(1 for r in matched if r.correct)}/{len(matched)}")
        total = [r for r in results if r.method == method.key]
        rows.append(
            (method.title, *cells, f"{sum(1 for r in total if r.correct)}/{len(total)}")
        )
    print_table(("Способ", *topics, "Всего"), rows)

    print("\nЛучший способ в каждой теме:")
    best_rows = []
    for topic in topics:
        scored = []
        for method in METHODS:
            matched = [
                r
                for r in results
                if r.method == method.key and topic_of.get(r.task_id) == topic
            ]
            if matched:
                scored.append(
                    (sum(1 for r in matched if r.correct) / len(matched), method.title, matched)
                )
        if not scored:
            continue
        best_ratio = max(item[0] for item in scored)
        winners = [item for item in scored if item[0] == best_ratio]
        matched = winners[0][2]
        tokens = sum((r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in matched)
        best_rows.append(
            (
                topic,
                ", ".join(item[1] for item in winners),
                f"{best_ratio * 100:.0f}%",
                _format_int(tokens // len(matched) if matched else None),
            )
        )
    print_table(
        ("Тема", "Лучший способ", "Доля верных", "Токенов на запрос"),
        best_rows,
    )


def print_token_stats(tasks: list[Task], results: list[RunResult]) -> None:
    """Полная статистика расхода токенов и срезы по способам, уровням и темам."""
    measured = [
        r for r in results if r.prompt_tokens is not None or r.completion_tokens is not None
    ]
    if not measured:
        print("\nСтатистика токенов: API не вернул usage ни для одного запроса.")
        return

    prompt_total = sum(r.prompt_tokens or 0 for r in measured)
    completion_total = sum(r.completion_tokens or 0 for r in measured)
    cached_total = sum(r.precached_tokens or 0 for r in measured)
    correct = sum(1 for r in measured if r.correct)

    print("\nСтатистика токенов (включая мета-запросы за промптом):")
    print_table(
        ("Показатель", "Значение"),
        (
            ("Запросов с usage", str(len(measured))),
            ("Токенов на входе (prompt)", _format_int(prompt_total)),
            ("Токенов на выходе (completion)", _format_int(completion_total)),
            ("Всего токенов", _format_int(prompt_total + completion_total)),
            ("Из них переиспользовано из кэша", _format_int(cached_total)),
            (
                "Токенов на верный ответ",
                _format_int(
                    (prompt_total + completion_total) // correct if correct else None
                ),
            ),
            (
                "Токенов на один запрос в среднем",
                _format_int(
                    (prompt_total + completion_total) // len(measured) if measured else None
                ),
            ),
        ),
    )

    print("\nТокены и время по способам решения:")
    rows = []
    for method in METHODS:
        matched = [r for r in measured if r.method == method.key]
        if not matched:
            continue
        prompts = sum(r.prompt_tokens or 0 for r in matched)
        completions = sum(r.completion_tokens or 0 for r in matched)
        rows.append(
            (
                method.title,
                str(len(matched)),
                _format_int(prompts),
                _format_int(completions),
                _format_int(prompts + completions),
                str(sum(1 for r in matched if r.correct)),
                f"{sum(r.seconds for r in matched):.0f}",
            )
        )
    print_table(
        ("Способ", "Запросов", "prompt", "completion", "Всего", "Верно", "Секунд"),
        rows,
    )

    print("\nТокены по уровням рассуждений:")
    level_rows = []
    for reasoning in REASONING_LEVELS:
        matched = [r for r in measured if r.reasoning == reasoning]
        if not matched:
            continue
        prompts = sum(r.prompt_tokens or 0 for r in matched)
        completions = sum(r.completion_tokens or 0 for r in matched)
        level_rows.append(
            (
                reasoning,
                str(len(matched)),
                _format_int(prompts),
                _format_int(completions),
                _format_int(prompts + completions),
                str(sum(1 for r in matched if r.correct)),
            )
        )
    print_table(
        ("Уровень", "Запросов", "prompt", "completion", "Всего", "Верно"),
        level_rows,
    )

    print("\nСрезы по темам задач:")
    topics = sorted({task.topic for task in tasks if task.topic})
    topic_rows = []
    for topic in topics:
        task_ids = {task.id for task in tasks if task.topic == topic}
        matched = [r for r in measured if r.task_id in task_ids]
        if not matched:
            continue
        tokens = sum((r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in matched)
        ok = sum(1 for r in matched if r.correct)
        topic_rows.append(
            (
                topic,
                str(len(task_ids)),
                str(len(matched)),
                _format_int(tokens),
                f"{ok}/{len(matched)}",
                f"{100 * ok / len(matched):.0f}%",
                _format_int(tokens // ok if ok else None),
            )
        )
    print_table(
        ("Тема", "Задач", "Запросов", "Токенов", "Верно", "Доля", "Токенов на верный"),
        topic_rows,
    )

    print("\nСрезы по сложности задач:")
    difficulty_rows = []
    for difficulty in sorted({task.difficulty for task in tasks}):
        task_ids = {task.id for task in tasks if task.difficulty == difficulty}
        matched = [r for r in measured if r.task_id in task_ids]
        if not matched:
            continue
        ok = sum(1 for r in matched if r.correct)
        difficulty_rows.append(
            (
                str(difficulty),
                str(len(task_ids)),
                str(len(matched)),
                f"{ok}/{len(matched)}",
                f"{100 * ok / len(matched):.0f}%",
            )
        )
    print_table(
        ("Сложность", "Задач", "Запросов", "Верно", "Доля"), difficulty_rows
    )

    print("\nСамые дорогие запросы (по сумме токенов):")
    expensive = sorted(
        measured,
        key=lambda r: (r.prompt_tokens or 0) + (r.completion_tokens or 0),
        reverse=True,
    )[:5]
    print_table(
        ("Задача", "Способ", "Уровень", "prompt", "completion", "Всего", "Верно"),
        [
            (
                r.task_id,
                r.method,
                r.reasoning,
                _format_int(r.prompt_tokens),
                _format_int(r.completion_tokens),
                _format_int((r.prompt_tokens or 0) + (r.completion_tokens or 0)),
                _status_cell(r),
            )
            for r in expensive
        ],
    )


def print_diagnostics(results: list[RunResult]) -> None:
    """Показывает, что ушло в API и что вернулось: effort, токены, повторы ответов.

    precached > 0 означает, что GigaChat переиспользовал префикс промпта.
    Столбец «повтор» отмечает ответы, совпавшие с ответом другого уровня
    на ту же задачу тем же способом: так видно, что разница режимов не проявилась.
    """
    if not results:
        return
    print("\nДиагностика запросов:")
    rows = []
    for result in sorted(results, key=lambda r: (r.task_id, r.method, r.reasoning)):
        duplicate = any(
            other is not result
            and other.task_id == result.task_id
            and other.method == result.method
            and other.reasoning != result.reasoning
            and other.answer
            and other.answer == result.answer
            for other in results
        )
        rows.append(
            (
                result.task_id,
                result.method,
                result.reasoning,
                result.effort or "—",
                str(result.prompt_tokens) if result.prompt_tokens is not None else "—",
                str(result.completion_tokens)
                if result.completion_tokens is not None
                else "—",
                str(result.precached_tokens)
                if result.precached_tokens is not None
                else "—",
                f"{result.seconds:.1f}",
                str(result.reasoning_chars) if result.reasoning_chars else "—",
                "= тот же, что у другого уровня" if duplicate else "",
            )
        )
    print_table(
        (
            "Задача",
            "Способ",
            "Уровень",
            "effort",
            "prompt",
            "completion",
            "precached",
            "с",
            "размышл.",
            "повтор",
        ),
        rows,
    )

    precached_hits = sum(1 for r in results if r.precached_tokens)
    duplicates = sum(
        1
        for result in results
        if any(
            other is not result
            and other.task_id == result.task_id
            and other.method == result.method
            and other.reasoning != result.reasoning
            and other.answer
            and other.answer == result.answer
            for other in results
        )
    )
    print(
        f"\nЗапросов с переиспользованием промпта (precached > 0): "
        f"{precached_hits} из {len(results)}"
    )
    if duplicates:
        print(
            f"Ответов, совпавших с ответом другого уровня на ту же задачу тем же "
            f"способом: {duplicates}. Различие уровней в этих ячейках не проявилось."
        )


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и портит кириллицу.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        env_file, tasks_path, use_cache = parse_args(sys.argv[1:])
        load_env_file(env_file)
        # Набор задач спрашиваем до запросов к API, чтобы не тратить их зря.
        tasks_file = tasks_path or choose_tasks_file()
        tasks = load_tasks(tasks_file)

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
        print(f"Задачи: {tasks_file} — {len(tasks)} шт.")
        print(f"Запросов на задачу: {len(REASONING_LEVELS)} уровня × "
              f"{len(METHODS)} способа (+ {len(REASONING_LEVELS)} запросов за промптом)")

        print(f"\n{VEAI_PROVIDER}: запрос списка моделей...")
        try:
            veai_models = chat_models(veai.list_models(), VEAI_PROVIDER)
        except VeaiError as exc:
            veai_models = []
            print(f"  недоступен: {exc}")
        print(f"{GIGACHAT_PROVIDER}: авторизация и запрос списка моделей...")
        try:
            gigachat_models = chat_models(
                gigachat.list_models(gigachat.get_access_token(use_cache)),
                GIGACHAT_PROVIDER,
            )
        except GigaChatError as exc:
            gigachat_models = []
            print(f"  недоступен: {exc}")

        providers = []
        if gigachat_models:
            providers.append(GIGACHAT_PROVIDER)
        if veai_models:
            providers.append(VEAI_PROVIDER)
        if not providers:
            raise AppError("Ни один провайдер не ответил, сравнивать нечего.")
        provider = (
            providers[0]
            if len(providers) == 1
            else ask_choice("\nКакого провайдера проверять?", providers)
        )
        models = gigachat_models if provider == GIGACHAT_PROVIDER else veai_models
        labeled = {
            str(m.get("id", "")): reasoning_limitation(m, provider) for m in models
        }
        options = [
            f"{model_id} ({limitation})" if limitation else model_id
            for model_id, limitation in sorted(labeled.items())
        ]
        limited_count = sum(1 for limitation in labeled.values() if limitation)
        print(
            f"\nМоделей в списке: {len(options)}"
            + (
                f", из них с пометкой о неполном тесте: {limited_count} "
                "(выбрать можно, но различие уровней на них не проверяется)"
                if limited_count
                else ""
            )
        )
        choice = ask_choice(f"{provider}: какую модель проверять?", options)
        # Из пометки восстанавливаем чистое имя модели: оно уходит в API.
        model = choice.split(" (")[0].strip()
        limitation = labeled[model]

        print(f"\nМодель: {provider} / {model}")
        if limitation:
            print(
                f"ВНИМАНИЕ: {limitation}. Тест пройдёт, но результаты по уровням "
                "рассуждений на этой модели не показательны."
            )
        print("Уровни рассуждений: " + ", ".join(REASONING_LEVELS))
        started = time.time()
        results: list[RunResult] = []
        for index, task in enumerate(tasks, start=1):
            print(
                f"\nЗадача {index}/{len(tasks)}: {task.id} "
                f"(тема: {task.topic}, сложность {task.difficulty}, "
                f"тип ответа: {task.answer_type})"
            )
            print(f"  Ожидаемый ответ: {task.expected}")
            results.extend(
                run_task(task, provider, model, gigachat, veai, use_cache)
            )
        elapsed = time.time() - started
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 1

    print_summary(tasks, results, provider, model, limitation)
    correct = sum(1 for r in results if r.correct)
    print(f"\nВремя: {elapsed:.0f} с. Итог: {correct} верных из {len(results)} ответов "
          f"({100 * correct / len(results):.1f}%).")

    try:
        written = export_results(
            tasks, results, provider, model, elapsed, limitation, tasks_file
        )
    except AppError as exc:
        print(f"\nНе удалось сохранить результаты: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"Сохранено: {path}")
    return 0


def export_basename(provider: str, model: str, moment: datetime) -> str:
    """Имя файлов вида w1t3_модель_дата_время без символов, запрещённых в имени."""
    safe_model = re.sub(r"[^\w.-]+", "-", model.replace("/", "-")).strip("-.") or "model"
    return f"w1t3_{safe_model}_{moment:%Y-%m-%d_%H-%M-%S}"


def export_results(
    tasks: list[Task],
    results: list[RunResult],
    provider: str,
    model: str,
    elapsed: float,
    limitation: str = "",
    tasks_file: Path | None = None,
) -> list[Path]:
    """Сохраняет результаты в JSON и CSV в корне проектов (рядом с tasks.json)."""
    moment = datetime.now()
    base = export_basename(provider, model, moment)
    out_dir = PROJECT_DIR.parent
    json_path = out_dir / f"{base}.json"
    csv_path = out_dir / f"{base}.csv"

    task_by_id = {task.id: task for task in tasks}
    payload = {
        "meta": {
            "provider": provider,
            "model": model,
            "reasoning_levels_comparable": not limitation,
            "reasoning_limitation": limitation,
            "started_at": moment.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "tasks_file": (tasks_file or DEFAULT_TASKS_FILE).name,
            "reasoning_levels": list(REASONING_LEVELS),
            "reasoning_effort": {
                GIGACHAT_PROVIDER: GIGACHAT_REASONING_EFFORT,
                VEAI_PROVIDER: VEAI_REASONING_EFFORT,
            },
            "methods": [{"key": m.key, "title": m.title} for m in METHODS],
            "totals": {
                "requests": len(results),
                "correct": sum(1 for r in results if r.correct),
                "errors": sum(1 for r in results if r.error),
                "prompt_tokens": sum(r.prompt_tokens or 0 for r in results),
                "completion_tokens": sum(r.completion_tokens or 0 for r in results),
                "precached_tokens": sum(r.precached_tokens or 0 for r in results),
            },
        },
        "summary": {
            "by_method": _group_counts(results, lambda r: r.method),
            "by_reasoning": _group_counts(results, lambda r: r.reasoning),
            "by_topic": _group_counts(
                results, lambda r: task_by_id[r.task_id].topic if r.task_id in task_by_id else ""
            ),
            "by_difficulty": _group_counts(
                results,
                lambda r: str(task_by_id[r.task_id].difficulty)
                if r.task_id in task_by_id
                else "",
            ),
            "method_x_topic": {
                method.key: {
                    topic: _group_counts(
                        [
                            r
                            for r in results
                            if r.method == method.key
                            and r.task_id in task_by_id
                            and task_by_id[r.task_id].topic == topic
                        ],
                        lambda r: "итого",
                    ).get("итого", {"requests": 0, "correct": 0})
                    for topic in sorted({t.topic for t in tasks if t.topic})
                }
                for method in METHODS
            },
        },
        "results": [
            {
                "task_id": r.task_id,
                "topic": task_by_id[r.task_id].topic if r.task_id in task_by_id else "",
                "difficulty": task_by_id[r.task_id].difficulty
                if r.task_id in task_by_id
                else None,
                "expected": task_by_id[r.task_id].expected
                if r.task_id in task_by_id
                else "",
                "reasoning": r.reasoning,
                "effort": r.effort,
                "method": r.method,
                "answer": r.answer,
                "correct": r.correct,
                "seconds": round(r.seconds, 2),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "precached_tokens": r.precached_tokens,
                "reasoning_chars": r.reasoning_chars,
                "error": r.error,
            }
            for r in results
        ],
    }

    try:
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise AppError(f"не удалось записать {json_path}: {exc}") from exc

    fieldnames = [
        "task_id",
        "topic",
        "difficulty",
        "method",
        "reasoning",
        "effort",
        "expected",
        "answer",
        "correct",
        "seconds",
        "prompt_tokens",
        "completion_tokens",
        "precached_tokens",
        "reasoning_chars",
        "error",
    ]
    try:
        # utf-8-sig: иначе Excel на Windows не видит кириллицу в CSV.
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "task_id": r.task_id,
                        "topic": task_by_id[r.task_id].topic if r.task_id in task_by_id else "",
                        "difficulty": task_by_id[r.task_id].difficulty
                        if r.task_id in task_by_id
                        else "",
                        "method": r.method,
                        "reasoning": r.reasoning,
                        "effort": r.effort,
                        "expected": task_by_id[r.task_id].expected
                        if r.task_id in task_by_id
                        else "",
                        "answer": r.answer,
                        "correct": int(r.correct),
                        "seconds": f"{r.seconds:.2f}",
                        "prompt_tokens": r.prompt_tokens if r.prompt_tokens is not None else "",
                        "completion_tokens": r.completion_tokens
                        if r.completion_tokens is not None
                        else "",
                        "precached_tokens": r.precached_tokens
                        if r.precached_tokens is not None
                        else "",
                        "reasoning_chars": r.reasoning_chars,
                        "error": r.error,
                    }
                )
    except OSError as exc:
        raise AppError(f"не удалось записать {csv_path}: {exc}") from exc
    return [json_path, csv_path]


def _group_counts(
    results: list[RunResult], key: Any
) -> dict[str, dict[str, int]]:
    """Счётчики запросов, верных и токенов по произвольной группировке."""
    grouped: dict[str, dict[str, int]] = {}
    for result in results:
        name = key(result)
        bucket = grouped.setdefault(
            name, {"requests": 0, "correct": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        bucket["requests"] += 1
        bucket["correct"] += 1 if result.correct else 0
        bucket["prompt_tokens"] += result.prompt_tokens or 0
        bucket["completion_tokens"] += result.completion_tokens or 0
    return grouped


if __name__ == "__main__":
    sys.exit(main())
