"""Исследование влияния температуры на ответы модели: одна модель, 3 × 5 прогонов.

Сценарий:
1. показывает модели GigaChat и Veai и просит выбрать провайдера и модель;
2. просит выбрать два запроса: с единственным правильным ответом и творческий;
3. отправляет каждый запрос 5 раз на температурах 0, 0.7 и 1.2;
4. печатает все ответы и считает, насколько они различаются внутри пятёрки:
   для точного ответа — долю верных и число уникальных ответов, для творческого —
   попарное лексическое сходство, разнообразие формулировок и разброс длины;
5. просит оценить творческие ответы вручную и сохраняет всё в JSON и CSV.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
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
from veai_client import VeaiClient, VeaiError

GIGACHAT_CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"
GIGACHAT_FALLBACK_CREDENTIALS_ENV = "GIGACHAT_API_KEY"
VEAI_CREDENTIALS_ENV = "VEAI_CREDENTIALS"
DEFAULT_GIGACHAT_SCOPE = "GIGACHAT_API_PERS"
GIGACHAT_PROVIDER = "GigaChat"
VEAI_PROVIDER = "Veai"
PROJECT_DIR = Path(__file__).resolve().parent
ENV_SEARCH_START = PROJECT_DIR

# Температуры и число повторов. Задаются флагами --temps и --repeat.
DEFAULT_TEMPERATURES = (0.0, 0.7, 1.2)
DEFAULT_REPEAT = 5
# Потолок на повторы и температуры: каждый повтор — отдельный платный запрос.
MAX_REPEAT = 20

EXACT_KIND = "exact"
CREATIVE_KIND = "creative"


class AppError(RuntimeError):
    """Ошибка сценария: неверный ввод, запросы или отказ провайдера."""


@dataclass(frozen=True)
class Request:
    """Запрос эксперимента.

    kind == "exact"     — есть единственный правильный ответ, поле accepted
                          перечисляет допустимые его записи;
    kind == "creative"  — правильного ответа нет, оценивается только разброс.
    """

    key: str
    title: str
    kind: str
    prompt: str
    answer_type: str = "word"
    answer: str = ""
    accepted: tuple[str, ...] = ()


# Готовые варианты запросов. Свои можно ввести при выборе — они уйдут в файл-экспорт.
EXACT_CHOICES: tuple[Request, ...] = (
    Request(
        key="exact-speed",
        title="средняя скорость (ловушка, ответ 48)",
        kind=EXACT_KIND,
        prompt=(
            "Половину пути по расстоянию автомобиль проехал со скоростью 60 км/ч, "
            "вторую половину — со скоростью 40 км/ч. Какова средняя скорость на всём "
            "пути в км/ч? Ответ дай числом без единиц измерения."
        ),
        answer_type="int",
        answer="48",
        accepted=("48",),
    ),
    Request(
        key="exact-zeros",
        title="нули в 25! (ответ 6)",
        kind=EXACT_KIND,
        prompt=(
            "Сколькими нулями заканчивается десятичная запись числа 25! "
            "(двадцать пять факториал)? Ответ дай одним числом."
        ),
        answer_type="int",
        answer="6",
        accepted=("6", "шесть"),
    ),
    Request(
        key="exact-clock",
        title="совпадения стрелок (ответ 22)",
        kind=EXACT_KIND,
        prompt=(
            "Сколько совпадений минутной и часовой стрелок происходит за сутки "
            "(моменты 00:00 и 12:00 считаются)? Ответ дай одним числом."
        ),
        answer_type="int",
        answer="22",
        accepted=("22",),
    ),
)

CREATIVE_CHOICES: tuple[Request, ...] = (
    Request(
        key="creative-name",
        title="название продукта",
        kind=CREATIVE_KIND,
        prompt=(
            "Придумай название для мобильного приложения, которое по фотографии "
            "подбирает, что приготовить из продуктов, найденных в холодильнике.\n\n"
            "Требования: одно название, 1–3 слова, до 20 символов, без пояснений и "
            "без кавычек. В ответе — только само название."
        ),
    ),
    Request(
        key="creative-text",
        title="короткое объявление",
        kind=CREATIVE_KIND,
        prompt=(
            "Напиши короткое объявление о запуске этого приложения: по фотографии "
            "холодильника оно подсказывает, что приготовить.\n\n"
            "Требования: 2–3 предложения, живой стиль, без списков, заголовков и "
            "пояснений. В ответе — только текст объявления."
        ),
    ),
)


@dataclass
class RunResult:
    request_key: str
    kind: str
    temperature: float
    attempt: int
    content: str
    latency: float
    extracted: str = ""
    correct: bool = False
    error: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_chars: int = 0

    @property
    def status(self) -> str:
        if self.error:
            return "!"
        if self.kind == EXACT_KIND:
            return "+" if self.correct else "−"
        return "ok"


@dataclass(frozen=True)
class ChatReply:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_chars: int = 0


@dataclass(frozen=True)
class Dispersion:
    """Насколько различаются ответы одной пятёрки."""

    kind: str
    count: int
    errors: int
    unique_answers: int
    correct: int | None
    # Лексическое сходство: 1.0 — тексты совпадают дословно.
    cosine_mean: float
    cosine_min: float
    jaccard_mean: float
    # Разнообразие формулировок: доля не встречавшихся раньше биграмм.
    distinct_2: float
    first_cells: dict[str, int] = field(default_factory=dict)
    lengths: tuple[int, ...] = ()
    # Строка-вид: по ней видно, какие именно повторы совпали.
    identical_groups: tuple[int, ...] = ()


def find_env_file(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise AppError(f"Файл .env не найден ни в {start}, ни в его родительских папках.")


@dataclass(frozen=True)
class Options:
    env_path: Path
    use_cache: bool
    temperatures: tuple[float, ...]
    repeat: int


def parse_args(argv: list[str]) -> Options:
    env_path: Path | None = None
    use_cache = True
    temperatures = DEFAULT_TEMPERATURES
    repeat = DEFAULT_REPEAT
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--no-cache":
            use_cache = False
        elif arg in ("--env", "--temps", "--repeat"):
            if not args:
                raise AppError(f"После {arg} требуется значение.")
            value = args.pop(0)
            if arg == "--env":
                env_path = Path(value).expanduser()
            elif arg == "--temps":
                temperatures = parse_temperatures(value)
            else:
                repeat = parse_repeat(value)
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env=") :]).expanduser()
        elif arg.startswith("--temps="):
            temperatures = parse_temperatures(arg[len("--temps=") :])
        elif arg.startswith("--repeat="):
            repeat = parse_repeat(arg[len("--repeat=") :])
        else:
            raise AppError(
                f"Неизвестный аргумент: {arg}. Доступны: --env PATH, --temps "
                "0,0.7,1.2, --repeat 5, --no-cache."
            )
    return Options(
        env_path or find_env_file(ENV_SEARCH_START),
        use_cache,
        temperatures,
        repeat,
    )


def parse_temperatures(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip().replace(",", ".")
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise AppError(f"Температура «{part}» — не число.") from exc
        if not 0.0 <= value <= 2.0:
            raise AppError(f"Температура {value} вне диапазона 0–2.")
        if value in values:
            raise AppError(f"Температура {value} указана дважды.")
        values.append(value)
    if not values:
        raise AppError("Список температур пуст.")
    return tuple(values)


def parse_repeat(raw: str) -> int:
    try:
        repeat = int(raw)
    except ValueError as exc:
        raise AppError(f"Число повторов «{raw}» — не целое.") from exc
    if not 1 <= repeat <= MAX_REPEAT:
        raise AppError(f"Число повторов должно быть от 1 до {MAX_REPEAT}.")
    return repeat


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


def temperature_support(model: dict[str, Any], provider: str) -> str:
    """Почему на модели нельзя менять температуру. Пустая строка — ограничений нет."""
    if provider == GIGACHAT_PROVIDER:
        return ""
    params = model.get("supported_parameters") or model.get("supported_params")
    if isinstance(params, list) and params and "temperature" not in params:
        return "каталог Veai не заявляет temperature: запросы уйдут без параметра"
    return ""


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
    temperature: float,
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
) -> ChatReply:
    try:
        if provider == GIGACHAT_PROVIDER:
            token = gigachat.get_access_token(use_cache=use_cache)
            response = gigachat.chat(token, model, prompt, temperature=temperature)
        else:
            response = veai.chat(model, prompt, temperature=temperature)
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


def extract_answer(text: str, request: Request) -> str:
    """Достаёт из свободного ответа модели то, что сравнивается с эталоном."""
    if request.answer_type == "int":
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
    for variant in request.accepted:
        if squeeze(variant) in squeeze(normalized):
            return variant
    return normalized


def _strip_digits(raw: str) -> str:
    return re.sub(r"[\s\u00a0]", "", raw)


def is_correct(extracted: str, request: Request) -> bool:
    """Сверяет извлечённый ответ с эталоном.

    Для числовых запросов требуется точное совпадение; вариант из accepted
    принимается, только если в нём нет других чисел, иначе «24» из «1/16»
    считалось бы верным.
    """
    if not extracted:
        return False
    if request.answer_type == "int":
        try:
            if int(extracted) == int(request.answer):
                return True
        except ValueError:
            pass
        for variant in request.accepted:
            variant_numbers = re.findall(r"-?\d+", variant)
            if variant_numbers and [int(n) for n in variant_numbers] == [
                int(request.answer)
            ]:
                return squeeze(extracted) == squeeze(variant)
        return False
    return squeeze(extracted) == squeeze(request.answer) or squeeze(extracted) in {
        squeeze(variant) for variant in request.accepted
    }


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError as exc:
        raise AppError("Ввод прерван: не получены ответы на вопросы.") from exc


def ask_choice(title: str, options: list[str], allow_empty: bool = False) -> str:
    """Возвращает выбранный пункт. allow_empty разрешает пустой ввод (первый пункт)."""
    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    while True:
        raw = ask("Ваш выбор: ")
        if not raw and allow_empty:
            return options[0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw in options:
            return raw
        stripped = raw.split(" (")[0].strip()
        for option in options:
            if option.split(" (")[0].strip() == stripped:
                return option
        print(f"Введите номер от 1 до {len(options)} или точное имя.")


def choose_request(
    title: str,
    choices: tuple[Request, ...],
    make_custom: Callable[[], Request],
) -> Request:
    """Выбор из готовых вариантов с возможностью ввести свой."""
    labels = [f"{item.title} — «{item.prompt}»" for item in choices]
    labels.append("другой вопрос — ввести текстом")
    picked = ask_choice(title, labels, allow_empty=True)
    index = labels.index(picked)
    return choices[index] if index < len(choices) else make_custom()


def make_custom_exact() -> Request:
    print("\nСвой запрос с единственным правильным ответом.")
    prompt = ask("Текст вопроса: ")
    if not prompt:
        raise AppError("Текст вопроса пуст.")
    answer = ask("Эталонный ответ (число или слово): ")
    if not answer:
        raise AppError("Эталонный ответ пуст.")
    answer_type = "int" if re.fullmatch(r"-?\d+", answer) else "word"
    return Request(
        key="exact-custom",
        title="свой вопрос",
        kind=EXACT_KIND,
        prompt=prompt,
        answer_type=answer_type,
        answer=answer,
        accepted=(answer,),
    )


def make_custom_creative() -> Request:
    print("\nСвой творческий запрос. Полезно завершить его требованием к формату:")
    print("  например «Ответь одним вариантом, без пояснений».")
    prompt = ask("Текст запроса: ")
    if not prompt:
        raise AppError("Текст запроса пуст.")
    return Request(
        key="creative-custom",
        title="свой творческий запрос",
        kind=CREATIVE_KIND,
        prompt=prompt,
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


def run_experiment(
    requests: list[Request],
    temperatures: tuple[float, ...],
    repeat: int,
    provider: str,
    model: str,
    gigachat: GigaChatClient,
    veai: VeaiClient,
    use_cache: bool,
) -> list[RunResult]:
    results: list[RunResult] = []
    total = len(temperatures) * len(requests) * repeat
    done = 0
    for temperature in temperatures:
        for request in requests:
            print(
                f"\nТемпература {temperature:g} / {request.title} "
                f"({request.kind}) — {repeat} повторов"
            )
            for attempt in range(1, repeat + 1):
                done += 1
                print(f"  [{done}/{total}] попытка {attempt}…", end="", flush=True)
                started = time.time()
                try:
                    reply = chat(
                        provider, model, request.prompt, temperature, gigachat, veai, use_cache
                    )
                except AppError as exc:
                    elapsed = time.time() - started
                    print(f" ошибка: {exc}")
                    results.append(
                        RunResult(
                            request.key,
                            request.kind,
                            temperature,
                            attempt,
                            "",
                            elapsed,
                            error=str(exc),
                        )
                    )
                    continue
                elapsed = time.time() - started
                extracted = (
                    extract_answer(reply.content, request)
                    if request.kind == EXACT_KIND
                    else ""
                )
                correct = is_correct(extracted, request) if request.kind == EXACT_KIND else False
                result = RunResult(
                    request.key,
                    request.kind,
                    temperature,
                    attempt,
                    reply.content,
                    elapsed,
                    extracted=extracted,
                    correct=correct,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    reasoning_chars=reply.reasoning_chars,
                )
                results.append(result)
                if request.kind == EXACT_KIND:
                    print(
                        f" {'верно' if correct else 'неверно'} (ответ: {extracted or '—'}, "
                        f"ожидался {request.answer}, {elapsed:.1f} с)"
                    )
                else:
                    print(f" {elapsed:.1f} с, {len(reply.content)} симв.: {reply.content}")
    return results


def group_responses(
    results: list[RunResult], request_key: str, temperature: float
) -> list[RunResult]:
    return sorted(
        [
            r
            for r in results
            if r.request_key == request_key and abs(r.temperature - temperature) < 1e-9
        ],
        key=lambda r: r.attempt,
    )


def _tokens(text: str) -> list[str]:
    return [token for token in normalize(text).split() if token]


def _word_set(text: str) -> set[str]:
    return set(_tokens(text))


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def cosine(first: str, second: str) -> float:
    """Косинусная близость на частотных векторах слов — TF-IDF без внешних библиотек.

    Похожесть считается по одной паре, поэтому веса — IDF по двум документам:
    слово, встречающееся в обоих, весит 1, встречающееся в одном — меньше.
    """
    left, right = _tokens(first), _tokens(second)
    if not left or not right:
        return 0.0
    vocabulary = sorted(set(left) | set(right))
    weights: dict[str, float] = {}
    for word in vocabulary:
        in_left = word in left
        in_right = word in right
        # idf от числа документов пары, в которых слово есть.
        weights[word] = math.log((2 + 1) / (1 + int(in_left) + int(in_right))) + 1
    a = [left.count(word) * weights[word] for word in vocabulary]
    b = [right.count(word) * weights[word] for word in vocabulary]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard(first: str, second: str) -> float:
    left, right = _word_set(first), _word_set(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Доля уникальных n-грамм от общего их числа: 1.0 — все формулировки разные."""
    grams: list[str] = []
    for text in texts:
        tokens = _tokens(text)
        grams.extend(_bigrams(tokens) if n == 2 else tokens)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def first_cell(text: str) -> str:
    """Первая содержательная строка ответа — для коротких креативных запросов
    это и есть сам результат, поэтому совпадение первой строки означает повтор."""
    for line in text.splitlines():
        cleaned = line.strip().strip("*_#-•. \"'«»")
        if cleaned:
            return normalize(cleaned)
    return ""


def identical_groups(results: list[RunResult]) -> tuple[int, ...]:
    """Размеры групп повторов с дословно совпадающим ответом, по убыванию."""
    counts: dict[str, int] = {}
    for result in results:
        if result.error or not result.content:
            continue
        key = squeeze(result.content)
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted((n for n in counts.values() if n > 1), reverse=True))


def measure(results: list[RunResult], kind: str) -> Dispersion:
    answered = [r for r in results if not r.error and r.content.strip()]
    texts = [r.content for r in answered]
    pairs = [
        (texts[i], texts[j])
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]
    cosine_values = [cosine(a, b) for a, b in pairs]
    jaccard_values = [jaccard(a, b) for a, b in pairs]
    cells: dict[str, int] = {}
    for text in texts:
        cell = first_cell(text)
        cells[cell] = cells.get(cell, 0) + 1
    unique_answers = (
        len({r.extracted for r in answered if r.extracted}) if kind == EXACT_KIND else len(set(texts))
    )
    return Dispersion(
        kind=kind,
        count=len(answered),
        errors=sum(1 for r in results if r.error),
        unique_answers=unique_answers,
        correct=(
            sum(1 for r in answered if r.correct) if kind == EXACT_KIND else None
        ),
        cosine_mean=_mean(cosine_values),
        cosine_min=min(cosine_values) if cosine_values else 0.0,
        jaccard_mean=_mean(jaccard_values),
        distinct_2=distinct_n(texts),
        first_cells=cells,
        lengths=tuple(len(text) for text in texts),
        identical_groups=identical_groups(results),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cell_span(cells: dict[str, int]) -> str:
    """Повторяемость первого результата: «ровно один» / «2 совпали, 3 разные»."""
    if not cells:
        return "—"
    repeats = sum(n for n in cells.values() if n > 1)
    variants = len(cells)
    if repeats == 0:
        return f"все {variants} разные"
    return f"повторов {repeats} из {sum(cells.values())}, вариантов {variants}"


def print_all_answers(
    requests: list[Request],
    results: list[RunResult],
    temperatures: tuple[float, ...],
) -> None:
    for request in requests:
        print("\n" + "=" * 78)
        print(f"ОТВЕТЫ: {request.title}")
        print("=" * 78)
        print(f"Запрос: {request.prompt}")
        if request.kind == EXACT_KIND:
            print(f"Ожидаемый ответ: {request.answer}")
        for temperature in temperatures:
            print(f"\n--- температура {temperature:g} ---")
            for result in group_responses(results, request.key, temperature):
                head = f"{result.attempt}. "
                if result.error:
                    print(f"{head}ошибка: {result.error}")
                    continue
                if request.kind == EXACT_KIND:
                    mark = "+" if result.correct else "−"
                    print(
                        f"{head}[{mark}] ответ: {result.extracted or '—'} "
                        f"({result.latency:.1f} с)"
                    )
                    print(f"     {_indent(result.content)}")
                else:
                    print(f"{head}({len(result.content)} симв., {result.latency:.1f} с)")
                    print(f"     {_indent(result.content)}")


def _indent(text: str) -> str:
    return text.replace("\n", "\n     ")


def print_dispersion(
    requests: list[Request],
    results: list[RunResult],
    temperatures: tuple[float, ...],
) -> dict[str, list[Dispersion]]:
    """Таблица разброса: строки — запрос и температура, столбцы — метрики."""
    print("\n" + "=" * 78)
    print("НАСКОЛЬКО РАЗЛИЧАЮТСЯ ОТВЕТЫ ВНУТРИ ПЯТЁРКИ")
    print("=" * 78)
    print(
        "Сходство: 1.00 — ответы совпадают дословно, 0 — общего нет. "
        "distinct-2: доля новых формулировок, 1.00 — все разные."
    )
    rows: list[tuple[str, ...]] = []
    measured: dict[str, list[Dispersion]] = {}
    for request in requests:
        for temperature in temperatures:
            group = group_responses(results, request.key, temperature)
            dispersion = measure(group, request.kind)
            measured.setdefault(request.key, []).append(dispersion)
            if request.kind == EXACT_KIND:
                rows.append(
                    (
                        request.title,
                        f"{temperature:g}",
                        str(dispersion.count),
                        f"{dispersion.correct or 0}/{dispersion.count}",
                        str(dispersion.unique_answers),
                        f"{dispersion.cosine_mean:.2f}",
                        f"{dispersion.cosine_min:.2f}",
                        str(dispersion.errors),
                    )
                )
            else:
                rows.append(
                    (
                        request.title,
                        f"{temperature:g}",
                        str(dispersion.count),
                        f"{dispersion.cosine_mean:.2f}",
                        f"{dispersion.cosine_min:.2f}",
                        f"{dispersion.jaccard_mean:.2f}",
                        f"{dispersion.distinct_2:.2f}",
                        _cell_span(dispersion.first_cells),
                    )
                )
    exact_rows = [
        row
        for request in requests
        if request.kind == EXACT_KIND
        for row in rows
        if row[0] == request.title
    ]
    if exact_rows:
        print("\nЗапрос с единственным правильным ответом:")
        print_table(
            (
                "Запрос",
                "Темп.",
                "Ответов",
                "Верных",
                "Уникальных",
                "Сходство ср.",
                "Сходство мин.",
                "Ошибок",
            ),
            exact_rows,
        )
    creative_rows = [
        row
        for request in requests
        if request.kind == CREATIVE_KIND
        for row in rows
        if row[0] == request.title
    ]
    if creative_rows:
        print("\nТворческий запрос:")
        print_table(
            (
                "Запрос",
                "Темп.",
                "Ответов",
                "Сходство ср.",
                "Сходство мин.",
                "Жаккар",
                "distinct-2",
                "Совпадение первого ответа",
            ),
            creative_rows,
        )
        print("\nДлины ответов и группы дословных повторов:")
        length_rows = []
        for request in requests:
            if request.kind != CREATIVE_KIND:
                continue
            for temperature, dispersion in zip(
                temperatures, measured.get(request.key, [])
            ):
                lengths = dispersion.lengths
                length_rows.append(
                    (
                        request.title,
                        f"{temperature:g}",
                        f"{_mean([float(v) for v in lengths]):.0f}"
                        if lengths
                        else "—",
                        f"{min(lengths)}–{max(lengths)}" if lengths else "—",
                        ", ".join(f"×{n}" for n in dispersion.identical_groups) or "нет",
                    )
                )
        print_table(
            ("Запрос", "Темп.", "Длина ср.", "Длина min–max", "Повторы дословно"),
            length_rows,
        )
    return measured


def print_token_stats(results: list[RunResult]) -> None:
    measured = [r for r in results if r.prompt_tokens is not None]
    if not measured:
        print("\nСтатистика токенов: API не вернул usage ни для одного запроса.")
        return
    print("\nТокены и время:")
    rows = []
    temperatures = sorted({r.temperature for r in measured})
    for temperature in temperatures:
        matched = [r for r in measured if abs(r.temperature - temperature) < 1e-9]
        prompts = sum(r.prompt_tokens or 0 for r in matched)
        completions = sum(r.completion_tokens or 0 for r in matched)
        rows.append(
            (
                f"{temperature:g}",
                str(len(matched)),
                _format_int(prompts),
                _format_int(completions),
                _format_int(prompts + completions),
                f"{_mean([r.latency for r in matched]):.1f}",
            )
        )
    total_prompt = sum(r.prompt_tokens or 0 for r in measured)
    total_completion = sum(r.completion_tokens or 0 for r in measured)
    rows.append(
        (
            "итого",
            str(len(measured)),
            _format_int(total_prompt),
            _format_int(total_completion),
            _format_int(total_prompt + total_completion),
            f"{_mean([r.latency for r in measured]):.1f}",
        )
    )
    print_table(("Темп.", "Запросов", "prompt", "completion", "Всего", "Секунд ср."), rows)


def _format_int(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}".replace(",", " ")


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и портит кириллицу.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        options = parse_args(sys.argv[1:])
        load_env_file(options.env_path)
        # Запросы спрашиваем до обращения к API, чтобы не тратить запросы зря.
        exact_request = choose_request(
            "Какой запрос с единственным правильным ответом использовать?",
            EXACT_CHOICES,
            make_custom_exact,
        )
        creative_request = choose_request(
            "Какой творческий запрос использовать?",
            CREATIVE_CHOICES,
            make_custom_creative,
        )
        requests = [exact_request, creative_request]

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
        print(f"Температуры: {', '.join(f'{t:g}' for t in options.temperatures)}")
        print(
            f"Повторов на температуру: {options.repeat}. Всего запросов: "
            f"{len(options.temperatures) * len(requests) * options.repeat}"
        )

        print(f"\n{VEAI_PROVIDER}: запрос списка моделей...")
        try:
            veai_models = chat_models(veai.list_models(), VEAI_PROVIDER)
        except VeaiError as exc:
            veai_models = []
            print(f"  недоступен: {exc}")
        print(f"{GIGACHAT_PROVIDER}: авторизация и запрос списка моделей...")
        try:
            gigachat_models = chat_models(
                gigachat.list_models(gigachat.get_access_token(options.use_cache)),
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
            raise AppError("Ни один провайдер не ответил, проверять нечего.")
        provider = (
            providers[0]
            if len(providers) == 1
            else ask_choice("\nКакого провайдера использовать?", providers)
        )
        models = gigachat_models if provider == GIGACHAT_PROVIDER else veai_models
        labeled = {
            str(m.get("id", "")): temperature_support(m, provider) for m in models
        }
        labels = [
            f"{model_id} ({limitation})" if limitation else model_id
            for model_id, limitation in sorted(labeled.items())
        ]
        choice = ask_choice(f"{provider}: какую модель проверять?", labels)
        # Из пометки восстанавливаем чистое имя модели: оно уходит в API.
        model = choice.split(" (")[0].strip()
        limitation = labeled[model]

        print(f"\nМодель: {provider} / {model}")
        if limitation:
            print(
                f"ВНИМАНИЕ: {limitation}. Сравнение температур на этой модели "
                "может показать нулевой разброс не из-за температуры."
            )
        print(f"Температура 0 не гарантирует детерминизм: провайдер вправе так же "
              f"выбирать из распределения. Пятёрка при 0 — проверка этого остатка.")
        started = time.time()
        results = run_experiment(
            requests,
            options.temperatures,
            options.repeat,
            provider,
            model,
            gigachat,
            veai,
            options.use_cache,
        )
        elapsed = time.time() - started
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 1

    print_all_answers(requests, results, options.temperatures)
    print_dispersion(requests, results, options.temperatures)
    print_token_stats(results)

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nОшибок при запросах: {len(errors)}")
        for message in dict.fromkeys(r.error for r in errors):
            print(f"  {message}")

    print(f"\nВремя: {elapsed:.0f} с. Запросов: {len(results)}, "
          f"из них с ответом: {len(results) - len(errors)}.")

    try:
        written = export_results(
            requests, results, provider, model, elapsed, options.temperatures, limitation
        )
    except AppError as exc:
        print(f"\nНе удалось сохранить результаты: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"Сохранено: {path}")
    return 0


def export_basename(provider: str, model: str, moment: datetime) -> str:
    """Имя файлов вида w1t4_модель_дата_время без символов, запрещённых в имени."""
    safe_model = re.sub(r"[^\w.-]+", "-", model.replace("/", "-")).strip("-.") or "model"
    return f"w1t4_{safe_model}_{moment:%Y-%m-%d_%H-%M-%S}"


def export_results(
    requests: list[Request],
    results: list[RunResult],
    provider: str,
    model: str,
    elapsed: float,
    temperatures: tuple[float, ...],
    limitation: str = "",
) -> list[Path]:
    """Сохраняет результаты в JSON и CSV в корне проектов (рядом с tasks.json)."""
    moment = datetime.now()
    base = export_basename(provider, model, moment)
    out_dir = PROJECT_DIR.parent
    json_path = out_dir / f"{base}.json"
    csv_path = out_dir / f"{base}.csv"
    request_by_key = {request.key: request for request in requests}

    dispersion = {
        request.key: {
            f"{temperature:g}": _dispersion_payload(
                measure(group_responses(results, request.key, temperature), request.kind)
            )
            for temperature in temperatures
        }
        for request in requests
    }
    payload = {
        "meta": {
            "provider": provider,
            "model": model,
            "temperature_supported": not limitation,
            "temperature_limitation": limitation,
            "started_at": moment.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "temperatures": list(temperatures),
            "repeat": len({r.attempt for r in results}) or DEFAULT_REPEAT,
            "requests": [
                {
                    "key": request.key,
                    "title": request.title,
                    "kind": request.kind,
                    "prompt": request.prompt,
                    "answer_type": request.answer_type,
                    "answer": request.answer,
                    "accepted": list(request.accepted),
                }
                for request in requests
            ],
            "totals": {
                "requests": len(results),
                "errors": sum(1 for r in results if r.error),
                "prompt_tokens": sum(r.prompt_tokens or 0 for r in results),
                "completion_tokens": sum(r.completion_tokens or 0 for r in results),
            },
            "note": (
                "temperature=0 не гарантирует детерминизм: разброс при 0 показывает "
                "остаточную неопределённость провайдера."
            ),
        },
        "dispersion": dispersion,
        "results": [
            {
                "request_key": r.request_key,
                "kind": r.kind,
                "temperature": r.temperature,
                "attempt": r.attempt,
                "content": r.content,
                "extracted": r.extracted,
                "correct": r.correct,
                "latency_seconds": round(r.latency, 2),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
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
        "request_key",
        "kind",
        "temperature",
        "attempt",
        "content",
        "extracted",
        "correct",
        "latency_seconds",
        "prompt_tokens",
        "completion_tokens",
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
                        "request_key": r.request_key,
                        "kind": r.kind,
                        "temperature": f"{r.temperature:g}",
                        "attempt": r.attempt,
                        # Переводы строк внутри CSV ломают таблицу, поэтому схлопываем.
                        "content": re.sub(r"\s+", " ", r.content),
                        "extracted": r.extracted,
                        "correct": int(r.correct) if r.kind == EXACT_KIND else "",
                        "latency_seconds": f"{r.latency:.2f}",
                        "prompt_tokens": r.prompt_tokens
                        if r.prompt_tokens is not None
                        else "",
                        "completion_tokens": r.completion_tokens
                        if r.completion_tokens is not None
                        else "",
                        "error": r.error,
                    }
                )
    except OSError as exc:
        raise AppError(f"не удалось записать {csv_path}: {exc}") from exc
    return [json_path, csv_path]


def _dispersion_payload(dispersion: Dispersion) -> dict[str, Any]:
    return {
        "count": dispersion.count,
        "errors": dispersion.errors,
        "unique_answers": dispersion.unique_answers,
        "correct": dispersion.correct,
        "cosine_mean": round(dispersion.cosine_mean, 4),
        "cosine_min": round(dispersion.cosine_min, 4),
        "jaccard_mean": round(dispersion.jaccard_mean, 4),
        "distinct_2": round(dispersion.distinct_2, 4),
        "first_cells": dispersion.first_cells,
        "identical_groups": list(dispersion.identical_groups),
        "lengths": list(dispersion.lengths),
    }


if __name__ == "__main__":
    sys.exit(main())
