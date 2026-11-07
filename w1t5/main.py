"""Сравнение трёх моделей разного класса: маленькая, средняя и флагманская.

Сценарий:
1. читает задачи из tasks.json (по умолчанию — в корне проектов, на уровень выше);
2. показывает модели, у которых в w1t5_prices.json заполнена группа (tier),
   и раскладывает их на три класса;
3. для каждого семейства с полным набором групп считает, во сколько раз
   вырастает цена от маленькой модели к флагманской;
4. отправляет каждую задачу всем трём моделям выбранного семейства при одном
   уровне рассуждений и сверяет ответы с эталоном;
5. печатает таблицу знаков как в w1t3 и денежную сводку: во что обошёлся прогон
   и сколько стоит один верный ответ у каждой группы.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
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
# Задачи и цены лежат в корне проектов, чтобы ими пользовались и следующие задачи.
DEFAULT_TASKS_FILE = PROJECT_DIR.parent / "tasks.json"
SHORT_TASKS_FILE = PROJECT_DIR.parent / "tasks_short.json"
DEFAULT_PRICES_FILE = PROJECT_DIR.parent / "w1t5_prices.json"
ENV_SEARCH_START = PROJECT_DIR

# Классы моделей. Порядок — от дешёвой к дорогой.
TIERS = ("small", "medium", "flagship")
TIER_TITLES = {
    "small": "маленькая и дешёвая",
    "medium": "средняя",
    "flagship": "флагманская",
}

# Один уровень рассуждений на все модели: иначе разница в качестве смешалась бы
# с разницей в глубине рассуждений, а цена сравнивалась бы при разных условиях.
GIGACHAT_REASONING_EFFORT = {"низкий": "low", "максимальный": "high"}
VEAI_REASONING_EFFORT = {"низкий": "low", "максимальный": "max"}
REASONING_LEVEL = "низкий"


class AppError(RuntimeError):
    """Ошибка сценария: неверный ввод, задачи, цены или отказ провайдера."""


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
class Price:
    """Цена модели. Валюта определяется тем, какое поле заполнено."""

    vendor: str
    tier: str
    usd_in: float | None = None
    usd_out: float | None = None
    rub_per_1m: float | None = None
    source: str = ""
    checked: str = ""

    @property
    def currency(self) -> str:
        return "RUB" if self.rub_per_1m is not None else "USD"

    def cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        """Стоимость запроса в валюте прайса. None — не хватает данных о токенах."""
        if prompt_tokens is None and completion_tokens is None:
            return None
        prompt_tokens = prompt_tokens or 0
        completion_tokens = completion_tokens or 0
        if self.rub_per_1m is not None:
            # В пакете GigaChat вход и выход считаются по одной ставке.
            return (prompt_tokens + completion_tokens) * self.rub_per_1m / 1_000_000
        return (
            prompt_tokens * (self.usd_in or 0.0)
            + completion_tokens * (self.usd_out or 0.0)
        ) / 1_000_000

    @property
    def input_price(self) -> float:
        if self.rub_per_1m is not None:
            return self.rub_per_1m
        return self.usd_in or 0.0

    @property
    def output_price(self) -> float:
        if self.rub_per_1m is not None:
            return self.rub_per_1m
        return self.usd_out or 0.0


@dataclass(frozen=True)
class PriceBook:
    usd_rub: float
    usd_rub_date: str
    usd_rub_source: str
    veai_minute_rub: dict[str, float]
    models: dict[str, Price]

    def to_rub(self, amount: float, currency: str) -> float:
        return amount if currency == "RUB" else amount * self.usd_rub


@dataclass(frozen=True)
class RunResult:
    task_id: str
    model: str
    tier: str
    answer: str
    correct: bool
    seconds: float
    error: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_chars: int = 0

    @property
    def status(self) -> str:
        if self.error:
            return "!"
        return "+" if self.correct else "−"

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@dataclass(frozen=True)
class ChatReply:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_chars: int = 0


@dataclass(frozen=True)
class Options:
    env_path: Path
    tasks_path: Path | None
    prices_path: Path
    use_cache: bool


def find_env_file(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise AppError(f"Файл .env не найден ни в {start}, ни в его родительских папках.")


def parse_args(argv: list[str]) -> Options:
    env_path: Path | None = None
    tasks_path: Path | None = None
    prices_path = DEFAULT_PRICES_FILE
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
        elif arg in ("--env", "--tasks", "--prices"):
            if not args:
                raise AppError(f"После {arg} требуется путь.")
            value = Path(args.pop(0)).expanduser()
            if arg == "--env":
                env_path = value
            elif arg == "--tasks":
                tasks_path = value
            else:
                prices_path = value
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env=") :]).expanduser()
        elif arg.startswith("--tasks="):
            tasks_path = Path(arg[len("--tasks=") :]).expanduser()
        elif arg.startswith("--prices="):
            prices_path = Path(arg[len("--prices=") :]).expanduser()
        else:
            raise AppError(
                f"Неизвестный аргумент: {arg}. Доступны: --env PATH, --tasks PATH, "
                "--prices PATH, --full, --short, --no-cache."
            )
    return Options(
        env_path or find_env_file(ENV_SEARCH_START), tasks_path, prices_path, use_cache
    )


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
        options.append(
            (
                f"сокращённый — {short_count} задач, быстрый ({SHORT_TASKS_FILE.name})",
                SHORT_TASKS_FILE,
            )
        )
    if full_count:
        options.append(
            (
                f"полный — {full_count} задач, долгий ({DEFAULT_TASKS_FILE.name})",
                DEFAULT_TASKS_FILE,
            )
        )
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


def load_prices(path: Path) -> PriceBook:
    """Читает цены. Модель без цены в файле не участвует в денежной сводке."""
    if not path.is_file():
        raise AppError(f"Файл цен не найден: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise AppError(f"{path} не является корректным JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AppError(f"{path} должен содержать объект.")

    try:
        usd_rub = float(data["usd_rub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AppError(f"В {path} нет числового поля usd_rub.") from exc
    if usd_rub <= 0:
        raise AppError(f"В {path} поле usd_rub должно быть больше нуля.")

    models: dict[str, Price] = {}
    for model_id, raw in (data.get("models") or {}).items():
        if not isinstance(raw, dict):
            raise AppError(f"Цена модели {model_id} описана не объектом.")
        tier = str(raw.get("tier", ""))
        if tier and tier not in TIERS:
            raise AppError(
                f"У модели {model_id} группа «{tier}», а допустимы: {', '.join(TIERS)}."
            )
        has_usd = raw.get("usd_in") is not None or raw.get("usd_out") is not None
        has_rub = raw.get("rub_per_1m") is not None
        if has_usd and has_rub:
            raise AppError(
                f"У модели {model_id} заполнены и usd_*, и rub_per_1m: нужна одна валюта."
            )
        if has_usd and (raw.get("usd_in") is None or raw.get("usd_out") is None):
            raise AppError(f"У модели {model_id} нужны оба поля usd_in и usd_out.")
        models[model_id] = Price(
            vendor=str(raw.get("vendor", "")),
            tier=tier,
            usd_in=_as_float(raw.get("usd_in")),
            usd_out=_as_float(raw.get("usd_out")),
            rub_per_1m=_as_float(raw.get("rub_per_1m")),
            source=str(raw.get("source", "")),
            checked=str(raw.get("checked", "")),
        )

    minutes = data.get("veai_minute_rub") or {}
    return PriceBook(
        usd_rub=usd_rub,
        usd_rub_date=str(data.get("usd_rub_date", "")),
        usd_rub_source=str(data.get("usd_rub_source", "")),
        veai_minute_rub={k: float(v) for k, v in minutes.items() if _is_number(v)},
        models=models,
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def group_by_vendor(prices: dict[str, Price]) -> dict[str, dict[str, str]]:
    """Группирует модели с заполненной группой по семействам.

    Возвращает {вендор: {класс: id модели}}. Классы, не заполненные в файле,
    в тройке отсутствуют: модель без цены не может участвовать в денежном сравнении.
    """
    families: dict[str, dict[str, str]] = {}
    for model_id, price in sorted(prices.items()):
        if not price.tier or not price.vendor:
            continue
        families.setdefault(price.vendor, {})[price.tier] = model_id
    return families


def complete_families(families: dict[str, dict[str, str]]) -> list[str]:
    """Семейства, в которых заполнены все три класса."""
    return [vendor for vendor, tiers in families.items() if len(tiers) == len(TIERS)]


def price_ratio(book: PriceBook, pair: dict[str, str]) -> str:
    """Во сколько раз цена выхода флагмана выше цены маленькой модели.

    Сравниваем выход: он дороже входа и сильнее влияет на стоимость прогона,
    где модели пишут длинные рассуждения.
    """
    small = book.models[pair["small"]]
    flagship = book.models[pair["flagship"]]
    if small.output_price <= 0:
        return "—"
    if small.currency != flagship.currency:
        return "разные валюты"
    ratio = flagship.output_price / small.output_price
    if ratio >= 10:
        return f"в {ratio:.0f} раз дороже"
    return f"в {ratio:.1f} раза дороже"


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


# Модели, которые каталог Veai рекламирует с уровнем max, но API отклоняет его с HTTP 400.
VEAI_EFFORT_REJECTED = {
    "Qwen/qwen3.6-flash": {"max"},
    "Qwen/qwen3.7-max": {"max"},
    "Qwen/qwen3.7-plus": {"max"},
}
VEAI_EFFORT_ORDER = ("max", "xhigh", "high", "medium", "low")


def veai_effort(model_id: str) -> str:
    """Значение reasoning_effort для общего уровня с учётом ограничений API модели."""
    effort = VEAI_REASONING_EFFORT[REASONING_LEVEL]
    rejected = VEAI_EFFORT_REJECTED.get(model_id, set())
    if effort not in rejected:
        return effort
    start = VEAI_EFFORT_ORDER.index(effort)
    return next(
        (item for item in VEAI_EFFORT_ORDER[start + 1 :] if item not in rejected),
        effort,
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
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
) -> ChatReply:
    try:
        if provider == GIGACHAT_PROVIDER:
            token = gigachat.get_access_token(use_cache=use_cache)
            response = gigachat.chat(
                token, model, prompt, reasoning_effort=GIGACHAT_REASONING_EFFORT[REASONING_LEVEL]
            )
        else:
            response = veai.chat(model, prompt, reasoning_effort=veai_effort(model))
    except GigaChatError as exc:
        raise AppError(f"{provider}: {exc}") from exc
    except VeaiError as exc:
        raise AppError(f"{provider}: {exc}{openai_error_hint(exc)}") from exc

    usage = response.get("usage") or {}
    return ChatReply(
        content=extract_content(response, provider),
        prompt_tokens=_as_int(usage.get("prompt_tokens")),
        completion_tokens=_as_int(usage.get("completion_tokens")),
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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return re.sub(r"[\s\-‐‑–—]+", " ", text).strip()


def squeeze(text: str) -> str:
    return re.sub(r"[\s\-‐‑–—]+", "", normalize(text))


# В текстах задач встречаются числа из условия («8-й день», «1/16»), поэтому
# последнее число в ответе не годится: выбираем то, которое выглядит как итог.
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
    принимается, только если в нём нет других чисел, иначе «24» из «1/16»
    считалось бы верным.
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
            if variant_numbers and [int(n) for n in variant_numbers] == [
                int(task.expected)
            ]:
                return squeeze(extracted) == squeeze(variant)
        return False
    return squeeze(extracted) == squeeze(task.expected) or squeeze(extracted) in {
        squeeze(variant) for variant in task.accepted
    }


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


def print_price_plan(book: PriceBook, families: dict[str, dict[str, str]]) -> None:
    """Какие тройки моделей есть и во сколько раз в каждой растёт цена."""
    complete = complete_families(families)
    print(f"\nЦены: {DEFAULT_PRICES_FILE.name}, курс {book.usd_rub:g} ₽ за $1"
          + (f" (на {book.usd_rub_date})" if book.usd_rub_date else ""))
    print("\nТройки моделей по классам (цена выхода за 1M токенов):")
    rows: list[tuple[str, ...]] = []
    for vendor in sorted(families):
        tiers = families[vendor]
        cells = []
        for tier in TIERS:
            model_id = tiers.get(tier)
            if not model_id:
                cells.append("—")
                continue
            price = book.models[model_id]
            cells.append(f"{price.output_price:g} {price.currency}")
        note = "" if vendor in complete else "не хватает классов"
        rows.append((vendor, *cells, price_ratio(book, tiers) if vendor in complete else note))
    print_table(("Семейство", *TIER_TITLES.values(), "Флагман к маленькой"), rows)


def run_model(
    tasks: list[Task],
    provider: str,
    model: str,
    tier: str,
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
    progress: tuple[int, int],
) -> list[RunResult]:
    results: list[RunResult] = []
    position, total = progress
    for task in tasks:
        position += 1
        print(f"  [{position}/{total}] {task.id}…", end="", flush=True)
        started = time.time()
        try:
            reply = chat(provider, model, task.question, gigachat, veai, use_cache)
        except AppError as exc:
            elapsed = time.time() - started
            print(f" ошибка: {exc}")
            results.append(
                RunResult(task.id, model, tier, "", False, elapsed, error=str(exc))
            )
            continue
        elapsed = time.time() - started
        extracted = extract_answer(reply.content, task)
        correct = is_correct(extracted, task)
        print(
            f" {'верно' if correct else 'неверно'} "
            f"(ответ: {extracted or '—'}, ожидался {task.expected}, {elapsed:.1f} с, "
            f"pt={reply.prompt_tokens or 0}, ct={reply.completion_tokens or 0})"
        )
        results.append(
            RunResult(
                task.id,
                model,
                tier,
                extracted,
                correct,
                elapsed,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                reasoning_chars=reply.reasoning_chars,
            )
        )
    return results


def print_summary(
    tasks: list[Task],
    trio: dict[str, str],
    results: list[RunResult],
) -> None:
    print("\n" + "=" * 78)
    print("ЗНАКИ: одна ячейка на задачу и класс модели")
    print("=" * 78)
    print("+ верно, − неверно, ! ошибка запроса; порядок классов: маленькая, средняя, флагманская\n")
    rows: list[tuple[str, ...]] = []
    for task in tasks:
        cells = [task.id]
        for tier in TIERS:
            model = trio[tier]
            found = [
                r
                for r in results
                if r.task_id == task.id and r.model == model
            ]
            cells.append(found[0].status if found else "?")
        correct = sum(
            1 for r in results if r.task_id == task.id and r.model in trio.values() and r.correct
        )
        cells.append(f"{correct}/{len(TIERS)}")
        rows.append(tuple(cells))
    print_table(("Задача", *(TIER_TITLES[tier][:9] for tier in TIERS), "Верно"), rows)


def print_money(book: PriceBook, trio: dict[str, str], results: list[RunResult]) -> None:
    """Сколько стоил прогон и сколько стоит один верный ответ у каждой группы."""
    print("\n" + "=" * 78)
    print("ЦЕНА: сколько стоил прогон и один верный ответ")
    print("=" * 78)
    print(
        "Стоимость считается по прайсу вендора модели. Veai тарифицирует время работы "
        "модели в минутах, а не токены, поэтому его собственный счёт по этим цифрам "
        "не восстановить — расход минут показан ниже отдельно.\n"
    )
    rows: list[tuple[str, ...]] = []
    unpriced: list[str] = []
    for tier in TIERS:
        model = trio[tier]
        price = book.models.get(model)
        matched = [r for r in results if r.model == model]
        answered = [r for r in matched if not r.error]
        correct = sum(1 for r in answered if r.correct)
        tokens = sum(r.total_tokens for r in answered)
        seconds = sum(r.seconds for r in matched)
        if price is None:
            unpriced.append(model)
            rows.append(
                (
                    TIER_TITLES[tier],
                    model,
                    f"{correct}/{len(matched)}",
                    str(tokens),
                    "нет цены",
                    "нет цены",
                    f"{seconds:.0f}",
                )
            )
            continue
        cost = sum(
            value
            for value in (
                price.cost(r.prompt_tokens, r.completion_tokens) for r in answered
            )
            if value is not None
        )
        per_correct = cost / correct if correct else None
        rows.append(
            (
                TIER_TITLES[tier],
                model,
                f"{correct}/{len(matched)}",
                _format_int(tokens),
                _money(cost, price.currency) + f" / {_money(book.to_rub(cost, price.currency), 'RUB')}",
                _money(per_correct, price.currency) if per_correct is not None else "—",
                f"{seconds:.0f}",
            )
        )
    print_table(
        ("Класс", "Модель", "Верно", "Токенов", "Прогон", "Цена верного", "Секунд"),
        rows,
    )
    if unpriced:
        print(
            f"\nБез цены в {DEFAULT_PRICES_FILE.name}: {', '.join(unpriced)} — "
            "модель выпала из денежного сравнения."
        )

    print("\nЦены за 1M токенов, по которым считали:")
    price_rows = []
    for tier in TIERS:
        model = trio[tier]
        price = book.models.get(model)
        if price is None:
            continue
        price_rows.append(
            (
                TIER_TITLES[tier],
                model,
                f"{price.input_price:g} {price.currency}",
                f"{price.output_price:g} {price.currency}",
                price.checked,
            )
        )
    if price_rows:
        print_table(("Класс", "Модель", "Вход", "Выход", "Сверено"), price_rows)

    print_veai_minutes(book, trio, results)


def print_veai_minutes(
    book: PriceBook, trio: dict[str, str], results: list[RunResult]
) -> None:
    """Во сколько минут Veai обошёлся прогон по времени ответа.

    Это оценка, а не счёт из API: Veai платит за чистое время работы модели,
    а время ответа включает сеть и ожидание в очереди.
    """
    if not book.veai_minute_rub:
        return
    answered = [r for r in results if not r.error]
    if not answered:
        return
    seconds = sum(r.seconds for r in answered)
    minutes = seconds / 60
    rates = book.veai_minute_rub
    print(
        f"\nМинуты Veai по времени ответа (оценка): {minutes:.1f} мин за прогон, "
        f"для сравнения решает одну задачу за {sum(r.seconds for r in answered) / len(answered):.1f} с."
    )
    rows = []
    for name, title in (
        ("personal", "Персональный"),
        ("professional", "Профессиональный"),
        ("ultra", "Ультра"),
    ):
        rate = rates.get(name)
        if rate is None:
            continue
        rows.append((title, f"{rate:g}", f"{minutes * rate:.2f}"))
    if rows:
        print_table(("Тариф", "₽/мин", "₽ за прогон"), rows)
    print(
        "Точность оценки ограничена: время ответа включает сеть, а оплачиваемое "
        "время работы модели сервис не публикует."
    )


def _money(value: float | None, currency: str) -> str:
    if value is None:
        return "—"
    if currency == "RUB":
        return f"{value:.2f} ₽"
    # Стоимость одного прогона — центы, поэтому показываем четыре знака.
    return f"{value:.4f} $"


def _format_int(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}".replace(",", " ")


def print_diagnostics(results: list[RunResult]) -> None:
    if not results:
        return
    print("\nДиагностика запросов:")
    print_table(
        ("Задача", "Класс", "Модель", "prompt", "completion", "с", "размышл.", "Ошибка"),
        [
            (
                r.task_id,
                TIER_TITLES[r.tier][:9],
                r.model,
                _format_int(r.prompt_tokens),
                _format_int(r.completion_tokens),
                f"{r.seconds:.1f}",
                str(r.reasoning_chars) if r.reasoning_chars else "—",
                r.error,
            )
            for r in sorted(results, key=lambda r: (r.task_id, r.tier))
        ],
    )


def _status_cell(result: RunResult) -> str:
    if result.error:
        return "!"
    return "+" if result.correct else "−"


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и портит кириллицу.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        options = parse_args(sys.argv[1:])
        load_env_file(options.env_path)
        # Набор задач спрашиваем до запросов к API, чтобы не тратить их зря.
        tasks_file = options.tasks_path or choose_tasks_file()
        tasks = load_tasks(tasks_file)
        book = load_prices(options.prices_path)
        families = group_by_vendor(book.models)
        if not families:
            raise AppError(
                f"В {options.prices_path} ни у одной модели не заполнена группа (tier)."
            )
        print_price_plan(book, families)
        complete = complete_families(families)
        if not complete:
            raise AppError(
                "Ни в одном семействе нет всех трёх классов (small, medium, flagship). "
                f"Допишите цены в {options.prices_path}."
            )

        gigachat_credentials = os.environ.get(
            GIGACHAT_CREDENTIALS_ENV
        ) or os.environ.get(GIGACHAT_FALLBACK_CREDENTIALS_ENV)
        if not gigachat_credentials:
            raise AppError(
                f"В {options.env_path} не найден ключ {GIGACHAT_CREDENTIALS_ENV} "
                f"(или {GIGACHAT_FALLBACK_CREDENTIALS_ENV})."
            )
        veai_key = os.environ.get(VEAI_CREDENTIALS_ENV, "").strip()
        if not veai_key:
            raise AppError(f"В {options.env_path} не найден ключ {VEAI_CREDENTIALS_ENV}.")

        scope = os.environ.get("GIGACHAT_SCOPE", "").strip() or DEFAULT_GIGACHAT_SCOPE
        gigachat = GigaChatClient(
            credentials=gigachat_credentials,
            scope=scope,
            ca_cert_files=default_ca_cert_files(),
            token_cache_file=TOKEN_CACHE_FILE,
        )
        veai = VeaiClient(api_key=veai_key)

        print(f"\nКлючи: {options.env_path}")
        print(f"Задачи: {tasks_file} — {len(tasks)} шт.")
        print(f"Уровень рассуждений: {REASONING_LEVEL} (одинаковый для всех трёх моделей)")

        print(f"\n{VEAI_PROVIDER}: запрос списка моделей...")
        try:
            veai_ids = {
                str(m.get("id", "")) for m in chat_models(veai.list_models(), VEAI_PROVIDER)
            }
        except VeaiError as exc:
            veai_ids = set()
            print(f"  недоступен: {exc}")
        print(f"{GIGACHAT_PROVIDER}: авторизация и запрос списка моделей...")
        try:
            gigachat_ids = {
                str(m.get("id", ""))
                for m in chat_models(
                    gigachat.list_models(gigachat.get_access_token(options.use_cache)),
                    GIGACHAT_PROVIDER,
                )
            }
        except GigaChatError as exc:
            gigachat_ids = set()
            print(f"  недоступен: {exc}")

        available = []
        for vendor in complete:
            tiers = families[vendor]
            ids = set(tiers.values())
            provider = GIGACHAT_PROVIDER if ids & gigachat_ids else VEAI_PROVIDER
            if not ids & (gigachat_ids if provider == GIGACHAT_PROVIDER else veai_ids):
                continue
            available.append(vendor)
        if not available:
            raise AppError(
                "Ни одна тройка моделей из файла цен не доступна в API. "
                "Проверьте ключи и список моделей."
            )
        print(f"\nТроек для сравнения: {len(available)} — {', '.join(available)}")
        labels = [
            f"{vendor}: " + " → ".join(families[vendor][tier] for tier in TIERS)
            for vendor in available
        ]
        choice = ask_choice("Какое семейство сравнивать?", labels)
        vendor = available[labels.index(choice)]
        trio = families[vendor]
        provider = (
            GIGACHAT_PROVIDER
            if set(trio.values()) & gigachat_ids
            else VEAI_PROVIDER
        )
        print(f"\nСемейство: {vendor} через {provider}")
        for tier in TIERS:
            model = trio[tier]
            price = book.models[model]
            print(
                f"  {TIER_TITLES[tier]}: {model} — "
                f"{price.input_price:g}/{price.output_price:g} {price.currency} за 1M"
                if price.input_price == price.output_price
                else f"  {TIER_TITLES[tier]}: {model} — "
                f"вход {price.input_price:g}, выход {price.output_price:g} "
                f"{price.currency} за 1M"
            )
        print(f"Всего запросов: {len(tasks) * len(TIERS)}")

        started = time.time()
        results: list[RunResult] = []
        total = len(tasks) * len(TIERS)
        position = 0
        for tier in TIERS:
            model = trio[tier]
            print(f"\n{TIER_TITLES[tier]}: {model}")
            model_results = run_model(
                tasks, provider, model, tier, gigachat, veai, options.use_cache,
                (position, total),
            )
            position += len(tasks)
            results.extend(model_results)
        elapsed = time.time() - started
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 1

    print_summary(tasks, trio, results)
    print_money(book, trio, results)
    print_diagnostics(results)

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nОшибок при запросах: {len(errors)}")
        for message in dict.fromkeys(r.error for r in errors):
            print(f"  {message}")
    correct = sum(1 for r in results if r.correct)
    print(
        f"\nВремя: {elapsed:.0f} с. Итог: {correct} верных из {len(results)} ответов "
        f"({100 * correct / len(results):.1f}%)."
    )

    try:
        written = export_results(
            tasks, results, trio, book, provider, elapsed, tasks_file, options.prices_path
        )
    except AppError as exc:
        print(f"\nНе удалось сохранить результаты: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"Сохранено: {path}")
    return 0


def export_basename(vendor: str, moment: datetime) -> str:
    """Имя файлов вида w1t5_семейство_дата_время без запрещённых в имени символов."""
    safe = re.sub(r"[^\w.-]+", "-", vendor.replace("/", "-")).strip("-.") or "vendor"
    return f"w1t5_{safe}_{moment:%Y-%m-%d_%H-%M-%S}"


def export_results(
    tasks: list[Task],
    results: list[RunResult],
    trio: dict[str, str],
    book: PriceBook,
    provider: str,
    elapsed: float,
    tasks_file: Path,
    prices_file: Path,
) -> list[Path]:
    """Сохраняет результаты в JSON и CSV в корне проектов (рядом с tasks.json)."""
    moment = datetime.now()
    base = export_basename(book.models[trio["small"]].vendor, moment)
    out_dir = PROJECT_DIR.parent
    json_path = out_dir / f"{base}.json"
    csv_path = out_dir / f"{base}.csv"

    task_by_id = {task.id: task for task in tasks}
    money = {}
    for tier in TIERS:
        model = trio[tier]
        price = book.models[model]
        matched = [r for r in results if r.model == model and not r.error]
        cost = sum(
            value
            for value in (
                price.cost(r.prompt_tokens, r.completion_tokens) for r in matched
            )
            if value is not None
        )
        correct = sum(1 for r in matched if r.correct)
        money[tier] = {
            "model": model,
            "currency": price.currency,
            "input_per_1m": price.input_price,
            "output_per_1m": price.output_price,
            "price_source": price.source,
            "price_checked": price.checked,
            "requests": len(matched),
            "correct": correct,
            "tokens": sum(r.total_tokens for r in matched),
            "seconds": round(sum(r.seconds for r in matched), 1),
            "cost": round(cost, 6),
            "cost_rub": round(book.to_rub(cost, price.currency), 4),
            "cost_per_correct": round(cost / correct, 6) if correct else None,
        }

    payload = {
        "meta": {
            "provider": provider,
            "vendor": book.models[trio["small"]].vendor,
            "models": trio,
            "reasoning_level": REASONING_LEVEL,
            "reasoning_effort": {
                GIGACHAT_PROVIDER: GIGACHAT_REASONING_EFFORT[REASONING_LEVEL],
                VEAI_PROVIDER: veai_effort(trio["small"])
                if provider == VEAI_PROVIDER
                else VEAI_REASONING_EFFORT[REASONING_LEVEL],
            },
            "started_at": moment.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "tasks_file": tasks_file.name,
            "prices_file": prices_file.name,
            "usd_rub": book.usd_rub,
            "usd_rub_date": book.usd_rub_date,
            "usd_rub_source": book.usd_rub_source,
            "veai_minute_rub": book.veai_minute_rub,
            "totals": {
                "requests": len(results),
                "correct": sum(1 for r in results if r.correct),
                "errors": sum(1 for r in results if r.error),
            },
        },
        "money": money,
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
                "tier": r.tier,
                "model": r.model,
                "answer": r.answer,
                "correct": r.correct,
                "seconds": round(r.seconds, 2),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "reasoning_chars": r.reasoning_chars,
                "cost": _round_cost(
                    book.models[r.model].cost(r.prompt_tokens, r.completion_tokens)
                    if r.model in book.models
                    else None
                ),
                "cost_currency": book.models[r.model].currency
                if r.model in book.models
                else "",
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
        "tier",
        "model",
        "expected",
        "answer",
        "correct",
        "seconds",
        "prompt_tokens",
        "completion_tokens",
        "cost",
        "cost_currency",
        "error",
    ]
    try:
        # utf-8-sig: иначе Excel на Windows не видит кириллицу в CSV.
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for r in results:
                price = book.models.get(r.model)
                cost = (
                    price.cost(r.prompt_tokens, r.completion_tokens) if price else None
                )
                writer.writerow(
                    {
                        "task_id": r.task_id,
                        "topic": task_by_id[r.task_id].topic
                        if r.task_id in task_by_id
                        else "",
                        "difficulty": task_by_id[r.task_id].difficulty
                        if r.task_id in task_by_id
                        else "",
                        "tier": r.tier,
                        "model": r.model,
                        "expected": task_by_id[r.task_id].expected
                        if r.task_id in task_by_id
                        else "",
                        "answer": r.answer,
                        "correct": int(r.correct),
                        "seconds": f"{r.seconds:.2f}",
                        "prompt_tokens": r.prompt_tokens
                        if r.prompt_tokens is not None
                        else "",
                        "completion_tokens": r.completion_tokens
                        if r.completion_tokens is not None
                        else "",
                        "cost": f"{cost:.8f}" if cost is not None else "",
                        "cost_currency": price.currency if price else "",
                        "error": r.error,
                    }
                )
    except OSError as exc:
        raise AppError(f"не удалось записать {csv_path}: {exc}") from exc
    return [json_path, csv_path]


def _round_cost(value: float | None) -> float | None:
    return None if value is None else round(value, 8)


if __name__ == "__main__":
    sys.exit(main())
