"""Агент на GigaChat с явной моделью памяти из трёх слоёв.

Слои и их правила:

1. краткосрочная — история диалога из w2t2 (history.json и summary.json):
   в промпт уходят сводка и последние N обменов текущей задачи;
2. рабочая — данные текущей задачи (work.json): срок, бюджет, ограничение,
   встреча и прочее, что нужно только пока задача открыта; в промпт уходят
   последние 24 записи; меняется при явной смене задачи;
3. долговременная — профиль, решения и знания (long.json): имя, решения,
   формат, знания; в промпт уходят последние 12 записей; переживает задачи.

Что попадает в слой, решают три источника, и источник виден в /memory:
правила извлечения (те же шаблоны, что в w2t5), извлечение моделью по JSON
на каждый обмен и явные команды `/remember KEY=VALUE [long|work]`.

Задача сменяется только явно: рабочая память уходит в архив задач,
долговременная остаётся, краткосрочная история начинается заново.

Проверка по заданию: /memory показывает состав каждого слоя, /limits —
вклад каждого слоя в токены запроса, /cleanup по очереди удаляет слои
(в долговременной -> рабочей -> краткосрочной) в независимой копии и задаёт
те же вопросы из details.json, фиксируя, на какие вопросы модель перестаёт
отвечать. Исходные файлы и память основной сессии не меняются.

Токены считаются дважды: по `usage` из ответа API (это факт и деньги) и своей
оценкой того, сколько уйдёт следующим запросом (API не говорит заранее).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from gigachat_client import (
    TOKEN_CACHE_FILE,
    GigaChatClient,
    GigaChatError,
    HttpLogger,
    default_ca_cert_files,
)
from veai_client import VeaiClient, VeaiError

GIGACHAT_CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"
GIGACHAT_FALLBACK_CREDENTIALS_ENV = "GIGACHAT_API_KEY"
VEAI_CREDENTIALS_ENV = "VEAI_CREDENTIALS"
PROVIDERS = ("gigachat", "veai")
DEFAULT_PROVIDER = "gigachat"
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
# История разговора: без неё каждый запуск начинал бы беседу заново.
DEFAULT_HISTORY_FILE = PROJECT_DIR / "history.json"
# Сводка сжатой части разговора — отдельный файл: история не переписывается,
# сжатые обмены остаются на диске, а в контекст идёт только сводка.
DEFAULT_SUMMARY_FILE = PROJECT_DIR / "summary.json"
DEFAULT_BRANCHES_FILE = PROJECT_DIR / "branches.json"
# Сколько последних обменов уходит модели как есть. Всё, что старше,
# заменяется сводкой.
DEFAULT_KEEP_TURNS = 4
STRATEGY_SUMMARY = "summary"
STRATEGY_SLIDING = "sliding"
# Факты теперь живут в отдельных слоях памяти, а не в истории: стратегия
# оставлена для совместимости со старыми файлами истории из w2t5.
STRATEGY_FACTS = "facts"
STRATEGIES = (STRATEGY_SUMMARY, STRATEGY_SLIDING, STRATEGY_FACTS)
DEFAULT_STRATEGY = STRATEGY_SUMMARY
# Сколько обменов копить до автоматического сжатия.
DEFAULT_COMPRESS_EVERY = 5
# По сколько старых обменов отдавать модели за один шаг сжатия: сводка
# сама становится входом, и кусок больше этого рискует не влезть в контекст.
COMPRESS_CHUNK_TURNS = 10
# Сколько символов обмена оставлять в подсказке для сжатия: пересказывать
# всё дословно не нужно, а очень длинный ответ функции раздувает запрос.
COMPRESS_CHARS_PER_TURN = 1500
# Сколько символов прежней сводки передавать модели при досжатии.
COMPRESS_SUMMARY_CHARS = 4000
SUMMARY_DETAIL_MARK = "·"
# Инструмент readme_words берёт слова из README этой же задачи.
README_FILE = PROJECT_DIR / "README.md"
README_WORDS_PER_LINE = 16
# Файл с деталями для проверки памяти: список заранее известных фактов
# (срок, бюджет, ограничение, имя, решение) и вопрос к каждой детали.
DETAILS_FILE = PROJECT_DIR / "details.json"
BRANCHES_FILE = PROJECT_DIR / "branches.json"
# Рабочая и долговременная память: два отдельных файла, потому что это разные
# слои. Краткосрочная память — уже существующая история из w2t2
# (history.json + summary.json), её формат не меняется.
DEFAULT_WORK_FILE = PROJECT_DIR / "work.json"
DEFAULT_LONG_FILE = PROJECT_DIR / "long.json"
CLEANUP_FILE = PROJECT_DIR / "cleanup.json"
# Результаты проверки памяти — файл в папке задачи, а не в корне:
# это рабочий артефакт эксперимента, а не результат прогона модели.
CHECKS_FILE = PROJECT_DIR / "checks.json"
# Инструмент fetch_url: таймаут обращения и размер куска при чтении потока.
FETCH_TIMEOUT_SECONDS = 20.0
FETCH_CHUNK_BYTES = 64 * 1024
# Теги, содержимое которых к тексту страницы не относится.
FETCH_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "nav", "header", "footer"}
)
# Блочные теги: на их границах текст разбивается на строки, иначе абзацы слипнутся.
FETCH_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
        "li", "main", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }
)
# Адрес локального файла тоже принимается: так модели можно подсунуть
# собственную страницу или сохранённый текст.
FETCH_SCHEMES = ("http", "https", "file")
# Сколько символов адреса и картинки оставить в журнале вызовов: в адресе
# бывают токены и параметры, а в картинке — мегабайты base64.
LOG_ARGUMENT_CHARS = 300
# Лимит контекста в токенах — свойство модели, а не разговора. Модель его
# в ответе не заявляет: число узнано опытом (200000 слов → HTTP 422
# «maximum allowed context is 130048»), поэтому его можно переопределить.
DEFAULT_MODEL_LIMIT = 130048
# Служебные токены на одно сообщение (роль, разделители) — поправка оценки.
TOKENS_PER_MESSAGE = 4
# Сколько символов в среднем приходится на токен в кириллице. Оценка —
# пять (4–5 знаков на токен у токенизатора GigaChat на русской прозе),
# и в этом разделе меряется: на шагах без кэша префикса факт даёт
# около 1.15 оценки, то есть действительное значение ближе к шести
# и формула осторожна — оценивает вход с запасом, а не занижает его.
ESTIMATE_CHARS_PER_TOKEN = 5
# Режим /fill: десятки тысяч слов-заполнителей одним залпом, чтобы довести
# историю до лимита без тысячи введённых запросов. Слово — примерно токен.
FILL_FILLER = "память"
FILL_MIN_WORDS = 100
# Версия формата файла истории: по ней видно, когда формат менялся.
HISTORY_VERSION = 1
# Версия формата файла сводки.
SUMMARY_VERSION = 1
# Версия форматов рабочей и долговременной памяти.
MEMORY_VERSION = 1
# Слои памяти. Краткосрочный — история диалога из w2t2, рабочий — данные
# текущей задачи, долговременный — профиль, решения и знания.
SHORT_TERM = "short"
WORKING = "work"
LONG_TERM = "long"
LAYERS = (SHORT_TERM, WORKING, LONG_TERM)
LAYER_TITLES = {
    SHORT_TERM: "краткосрочная",
    WORKING: "рабочая",
    LONG_TERM: "долговременная",
}
# Откуда взялась запись памяти: правило в коде, модель или явная команда.
SOURCE_RULE = "правило"
SOURCE_MODEL = "модель"
SOURCE_USER = "явно"
# Порядок очистки в эксперименте: сначала то, что переживает задачи, потом
# данные задачи, потом сам диалог.
CLEANUP_ORDER = (LONG_TERM, WORKING, SHORT_TERM)
# Сколько последних обменов хранит краткосрочная память. История — хранилище
# слоя, а не лог разговора: старшее уже ушло в слои памяти или в сводку.
SHORT_TERM_TURNS = 4
# Правило отбора в промпт: долговременная память — до 12 последних записей,
# рабочая — до 24, краткосрочная — последние N обменов текущей задачи.
LONG_TERM_ITEMS = 12
WORKING_ITEMS = 24
# Сколько символов записи отдавать модели: важны ключ и значение, а длинное
# значение съедает контекст.
FACT_VALUE_CHARS = 300
# Что относится к данным задачи, а не к долговременной памяти.
WORKING_KEYS = frozenset({"срок", "дедлайн", "бюджет", "ограничение",
                         "встреча", "регион"})
# Отступ в JSON-записи HTTP-лога: один обмен читается глазами, вложенные
# сообщения и тела ответов стоят на своих строках.
HTTP_LOG_INDENT = 2

# Запросов на один вопрос. Агент с тремя инструментами укладывается в два шага;
# лимит нужен, чтобы модель, зациклившаяся на вызовах, не тратила токены.
MAX_STEPS = 6
# Команды диалога.
EXIT_COMMAND = "/exit"
CLEAR_COMMAND = "/clear"
CLEAR_LAYER_COMMAND = "/clear-layer"
LIMITS_COMMAND = "/limits"
MEMORY_COMMAND = "/memory"
REMEMBER_COMMAND = "/remember"
FORGET_COMMAND = "/forget"
TASK_COMMAND = "/task"
CLEANUP_COMMAND = "/cleanup"
COMPARE_COMMAND = "/compare"
FILL_COMMAND = "/fill"
HELP_COMMAND = "/help"
COMPRESS_COMMAND = "/compress"
CHECK_COMMAND = "/check"
# Справка выводится из таблиц, а обработка команд — из тех же констант, которые
# печатает таблица, поэтому справка не может разойтись с поведением.
COMMAND_HELP = (
    (EXIT_COMMAND, "выйти; история и память остаются на диске"),
    (CLEAR_COMMAND, "забыть разговор и начать новую задачу"),
    (CLEAR_LAYER_COMMAND, "стереть один слой: /clear-layer short|work|long, остальные не трогаются"),
    (MEMORY_COMMAND, "что лежит в каждом слое и сколько он занимает"),
    (REMEMBER_COMMAND, "записать факт: /remember KEY=VALUE [long|work]"),
    (FORGET_COMMAND, "удалить запись: /forget KEY [long|work]"),
    (TASK_COMMAND, "/task — текущая задача, /task NAME — новая, /task open N — вернуться"),
    (CLEANUP_COMMAND, "эксперимент: 4 состояния × вопросы из details.json"),
    (LIMITS_COMMAND, "расход токенов и вклад каждого слоя"),
    (COMPRESS_COMMAND, "сжать старые обмены в сводку: /compress или /compress N"),
    (CHECK_COMMAND, "прогнать вопросы и сохранить ответы для сравнения"),
    (COMPARE_COMMAND, "таблица результатов прогонов из cleanup.json"),
    (FILL_COMMAND, "наполнить историю словами: до лимита или /fill N слов"),
    (HELP_COMMAND, "эта справка"),
    ("/checkpoint NAME", "сохранить текущую точку для ветвления"),
    ("/branch NAME", "создать ветку от текущей точки"),
    ("/switch NAME", "переключиться на сохранённую ветку"),
)
OPTION_HELP = (
    ("--env PATH", "файл с ключом вместо поиска .env вверх по дереву"),
    ("--provider NAME", "провайдер: gigachat или veai"),
    ("--model NAME", f"модель выбранного провайдера (по умолчанию {DEFAULT_MODEL})"),
    ("--limit TOKENS", "лимит контекста модели для оценки истории"),
    ("--keep N", f"сколько обменов держать как есть (по умолчанию {DEFAULT_KEEP_TURNS})"),
    ("--every N", f"через сколько обменов сжимать (по умолчанию {DEFAULT_COMPRESS_EVERY})"),
    ("--strategy NAME", "summary, sliding или facts (Sticky Facts)"),
    ("--window N", f"размер окна последних сообщений (по умолчанию {DEFAULT_KEEP_TURNS})"),
    ("--no-compress", "устаревший алиас для sliding"),
    ("--history PATH", "другой файл истории разговора"),
    ("--work PATH", "другой файл рабочей памяти"),
    ("--long PATH", "другой файл долговременной памяти"),
    ("--summary PATH", "другой файл сводки"),
    ("--no-extract", "не извлекать факты моделью (только правила и явная запись)"),
    ("--keep-short", "не убирать из истории факты, ушедшие в слои памяти"),
    ("--branches PATH", "каталог файлов контрольных точек и веток"),
    ("--log PATH", "другой журнал вызовов инструментов"),
    ("--http-log PATH", "другой HTTP-лог диалога с моделью"),
    ("--no-cache", "переавторизоваться, минуя кэш токена"),
    ("--command-prefix", "раскрывать сокращения команд после Enter вместо Tab"),
)

# Инструмент, результат которого остаётся в истории как часть разговора: текст
# страницы кладётся в контекст и живёт там до /clear. Для исследования это
# сознательный выбор вопреки экономии: такая страница платится входными
# токенами на каждом следующем запросе.
PERSISTED_TOOLS = frozenset({"fetch_url"})

TIME_FORMAT = "%Y-%m-%d %H:%M:%S %Z"
SYSTEM_PROMPT = (
    "Ты агент с инструментами и явной памятью из трёх слоёв. Если для ответа "
    "нужен случайный бросок, текущее время или перевод числа в "
    "шестнадцатеричную систему, вызови подходящую функцию. Не выдумывай "
    "результаты броска и не считай системы счисления сам — вызывай функцию. "
    "Перед разговором идёт блок памяти: рабочая — данные текущей задачи, "
    "долговременная — профиль, решения и знания, дальше — прошлые обмены. "
    "Опирайся на все три слоя как на то, что уже известно. Отвечай кратко, "
    "по-русски, называя полученное число."
)
# Извлечение фактов: модель раскладывает новые сведения по слоям, а код решает,
# какое поле принять. Ответ строго JSON — иначе результат не разобрать.
EXTRACT_PROMPT = (
    "Ты ведёшь память ассистента. Разбери только сообщения пользователя и "
    "ассистента в конце инструкции. Верни JSON без пояснений строго вида: "
    "{\"facts\": [{\"layer\": \"long\" или \"work\" или \"short\", "
    "\"key\": \"короткий ключ по-русски\", "
    "\"value\": \"значение дословно из слов пользователя\"}]}. "
    "По умолчанию возвращай {\"facts\": []}: сохраняй сведения только при "
    "полной уверенности и только если пользователь сообщил их как факт. "
    "Никогда не сохраняй вопросы пользователя, просьбы, планы без параметров, "
    "рассуждения, общие советы, предположения, задачи ассистента, его ответы "
    "или факты, которые модель сама вывела из ответа. Не превращай тему "
    "разговора в факт. Одно предложение может дать несколько фактов только "
    "если каждый явно сказан пользователем.\n"
    "long — только сведения, которые должны пережить смену задач: имя, роль, "
    "постоянное предпочтение, язык, место проживания, устойчивое правило, "
    "принятое решение или явно сформулированное общее знание для будущих "
    "проектов. Если связь с будущими задачами не очевидна, не выбирай long.\n"
    "work — только конкретные параметры открытой задачи: срок, бюджет, "
    "ограничение, встреча, регион, объём или результат текущего проекта. "
    "Если это не относится к конкретной текущей задаче, не выбирай work.\n"
    "short — временная деталь текущего разговора, которая не является long или "
    "work. short не записывается в отдельный файл; такие обмены остаются в "
    "history.json. Для вопроса, просьбы или обычного обсуждения верни пустой "
    "список, а не short-факт.\n"
    "Пример 1: пользователь «Меня зовут Марина» -> "
    "{\"facts\":[{\"layer\":\"long\",\"key\":\"имя\",\"value\":\"Марина\"}]}. "
    "Пример 2: пользователь «Бюджет проекта 480000 рублей» -> "
    "{\"facts\":[{\"layer\":\"work\",\"key\":\"бюджет\",\"value\":\"480000 рублей\"}]}. "
    "Пример 3: пользователь «Какие риски проверить?» -> {\"facts\":[]}."
)
# В эксперименте проверяем способность читать память, а не способность модели
# догадаться. При отсутствии факта ответ должен быть ровно НЕТ ДАННЫХ.
MEMORY_CHECK_PROMPT = (
    "Ты отвечаешь на вопрос только по переданным блокам памяти. "
    "Не используй общие знания, догадки, арифметику или сведения из самого "
    "вопроса. Если точного ответа нет в памяти, ответь ровно «НЕТ ДАННЫХ». "
    "Если есть — приведи значение дословно и без добавления новых фактов."
)
# Просьба к модели при сжатии: пересказ уходит в контекст вместо десятков
# обменов, поэтому важен именно факт и цифра, а не литературность.
COMPRESS_PROMPT = (
    "Сожми часть диалога в сводку на русском. Сохрани все конкретные детали, "
    "которые могут понадобиться позже: имена, даты, сроки, суммы и бюджет, "
    "ограничения, принятые решения и договорённости, цифры и результаты "
    "вычислений. Ничего не выдумывай и не добавляй. Пиши тезисами, до 30 слов "
    "на строку, без вступлений вроде «в этом диалоге»."
)

# Три проверки агента: без инструментов, с одним и с двумя инструментами подряд.
# На них сохранены выводы прошлого прогона и по ним проверяется работа диалога.
EXAMPLE_QUERIES = (
    "Сколько будет 17 умножить на 23? Ответь одним числом.",
    "Брось игральную кость с 20 гранями и скажи, что выпало.",
    "Брось игральную кость с 8 гранями и переведи выпавшее число "
    "в шестнадцатеричную систему счисления.",
)

# Короткий диалог, длинный диалог и диалог за лимитом модели — три сценария
# по заданию. Запросы короткие, чтобы сценарий вводился одной вставкой.
SCENARIOS = {
    "short": ("Сколько будет 17 умножить на 23? Ответь одним числом.",
              "А если это число умножить на 2? Ответь одним числом."),
    "long": ("Запомни, меня зовут Анна.",
             "Сколько будет 12 умножить на 12? Ответь одним числом.",
             "Как меня зовут? Ответь одним словом.",
             "Назови три крупных города России через запятую.",
             "Сколько букв в слове «память»? Ответь одним числом.",
             "Какого цвета небо в ясный день? Ответь двумя словами.",
             "Сколько ног у паука? Ответь одним числом.",
             "Перечисли три времени года через запятую.",
             "Сколько дней в неделе? Ответь одним числом.",
             "Как меня зовут? Ответь одним словом."),
}
# Диалог за лимитом модели: история наполняется заполнителем до предела,
# и только потом задаётся вопрос.
OVERFLOW_SCENARIO = "overflow"


class AppError(RuntimeError):
    """Ошибка сценария: неверный ввод, ключ или отказ GigaChat."""


def estimate_tokens(text: str) -> int:
    """Оценка числа токенов по символам без вызова модели.

    Коэффициенты — единственное, что у нас есть до запроса: точного
    счётчика в стандартной библиотеке нет. Пять символов на токен взяты
    по русской прозе (README этой задачи, 4–5 знаков на токен у
    токенизатора GigaChat), латиница и цифры дешевле. Служебные токены
    на сообщение прибавляются в estimate_messages_tokens, а не здесь:
    описание инструментов — не сообщение.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    other_chars = len(text) - ascii_chars
    return round(ascii_chars / 4 + other_chars / ESTIMATE_CHARS_PER_TOKEN)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """То же для одного сообщения: текст, имя функции и сама просьба её вызвать."""
    total = TOKENS_PER_MESSAGE + estimate_tokens(str(message.get("content") or ""))
    total += estimate_tokens(str(message.get("name") or ""))
    if message.get("function_call"):
        total += estimate_tokens(
            json.dumps(message["function_call"], ensure_ascii=False)
        )
    return total


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Оценка входа для списка сообщений: текст, роли и имена функций."""
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_body_tokens(body: dict[str, Any]) -> int:
    """Оценка всего запроса: сообщения и описания инструментов."""
    return (
        estimate_messages_tokens(body["messages"])
        + estimate_tokens(json.dumps(body.get("functions") or [], ensure_ascii=False))
    )


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


def readme_words(lines: int) -> str:
    """Строки из случайных слов README: lines строк по README_WORDS_PER_LINE слов.

    Слова берутся из `w2t3/README.md` без повторов и в случайном порядке:
    так из одного файла получается связный по форме, но бессмысленный по смыслу
    текст, которым удобно наполнять разговор до нужного размера.
    """
    if lines < 1:
        raise ValueError(f"Нужна хотя бы одна строка, получено {lines}")
    if not README_FILE.is_file():
        raise FileNotFoundError(f"Не найден файл README: {README_FILE}")
    words = README_FILE.read_text(encoding="utf-8").split()
    # Одно и то же слово дважды в строке выглядит как сбой, а не как текст.
    unique = list(dict.fromkeys(words))
    needed = lines * README_WORDS_PER_LINE
    if len(unique) < needed:
        raise ValueError(
            f"В {README_FILE.name} только {len(unique)} разных слов, "
            f"а для {lines} строк нужно {needed}"
        )
    picked = random.sample(unique, needed)
    return "".join(
        f"И вот еще вопрос {number}: "
        + " ".join(picked[start:start + README_WORDS_PER_LINE])
        + "\n"
        for number, start in enumerate(
            range(0, needed, README_WORDS_PER_LINE), start=1
        )
    )


def fetch_url(url: str) -> str:
    """Текст страницы по адресу: заголовок, потом содержимое без разметки.

    Принимает http(s) и file: последнее — чтобы отдать модели локальный файл.
    Возвращается чистый текст, а не HTML: браузерного движка в стандартной
    библиотеке нет, поэтому страницы, которые без JavaScript пусты,
    вернут пустой текст, и это честный ответ, а не выдумка.
    """
    address = url.strip()
    if not address:
        raise ValueError("Пустой адрес")
    scheme = urllib.parse.urlsplit(address).scheme.lower()
    if scheme not in FETCH_SCHEMES:
        raise ValueError(
            f"Адрес {address} не поддерживается: нужен "
            + " или ".join(f"{name}://" for name in FETCH_SCHEMES)
        )
    body, charset = _read_url(address)
    title, text = _html_text(body, charset)
    heading = f"Заголовок: {title or address}"
    return f"{heading}\n\n{text}" if text else f"{heading}\n\nТекст не найден"


def _read_url(url: str) -> tuple[bytes, str]:
    """Скачивает адрес и возвращает тело целиком и имя кодировки."""
    if url.lower().startswith("file:"):
        path = _file_path(url)
        try:
            return path.read_bytes(), ""
        except OSError as exc:
            # Никакого fallback на «пустой ответ»: пользователь должен
            # видеть, что файла нет, а не читать выдуманный текст страницы.
            raise ValueError(f"Не удалось прочитать файл {path}: {exc}") from exc
    request = urllib.request.Request(url, headers={"User-Agent": "GigaChat-tool/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            # Размер не ограничиваем по заданию: получаем, что отдал сервер.
            return response.read(), response.headers.get_content_charset() or ""
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Страница ответила HTTP {exc.code}: {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"Не удалось скачать {url}: {exc}") from exc


def _file_path(url: str) -> Path:
    """Путь к локальному файлу из адреса вида file:///abs/path или file://rel/path.

    Руками чаще пишут не канонический `file:///`, а `file://a.txt` или
    `file:a.txt`. Привычный url2pathname съел бы имя файла как имя хоста,
    поэтому такой адрес разбирается вручную: путь без ведущего слеша —
    относительный, от текущего каталога.
    """
    rest = url[len("file:"):]
    if rest.startswith("//"):
        rest = rest[2:]
        # Путь с ведущим слешем — абсолютный; всё прочее — относительный,
        # иначе url2pathname превратил бы «a.txt» в «/a.txt».
        if not rest.startswith("/"):
            return Path.cwd() / urllib.request.url2pathname(rest)
    # file:///C:/path на Windows: снять слеш перед буквой диска.
    if re.fullmatch(r"/[A-Za-z]:[\\/].*", rest):
        rest = rest[1:]
    path = Path(urllib.request.url2pathname(rest))
    # Путь с буквой диска на Windows уже абсолютный, а на Linux такие
    # входные данные — обычное относительное имя.
    if path.is_absolute() or re.match(r"[A-Za-z]:[\\/]", rest):
        return path
    return Path.cwd() / path


def _html_text(body: bytes, charset: str) -> tuple[str, str]:
    """Заголовок и текст страницы: кодировка из ответа, затем UTF-8, затем cp1251.

    Latin-1 не ошибается никогда и стоит последним: лучше бредовая кириллица
    из чужой кодировки, чем отказ прочитать файл целиком.
    """
    for encoding in (charset, "utf-8", "cp1251", "latin-1"):
        if not encoding:
            continue
        try:
            markup = body.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        parser = _TextExtractor()
        parser.feed(markup)
        parser.close()
        return parser.title, parser.text()
    raise AssertionError("latin-1 не может не разобраться")


class _TextExtractor(HTMLParser):
    """Вытаскивает заголовок и текст, отбрасывая разметку и скрипты."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in FETCH_SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in FETCH_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in FETCH_SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in FETCH_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title + " " + data).strip()
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        """Схлопывает пробелы и пустые строки: разметка даёт их мешками."""
        joined = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.parts))
        lines = [line.strip() for line in joined.split("\n")]
        return "\n".join(line for line in lines if line)


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
    Tool(
        "readme_words",
        "Собирает текст из случайных слов файла README: строки по 16 слов, "
        "каждая начинается с «И вот еще вопрос N:».",
        {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "integer",
                    "description": "Сколько строк собрать, например 3 или 10.",
                }
            },
            "required": ["lines"],
        },
        readme_words,
    ),
    Tool(
        "fetch_url",
        "Скачивает страницу по адресу http://, https:// или file:// и возвращает "
        "её текст без разметки вместе с заголовком. Используй, когда просят "
        "проанализировать или пересказать содержимое страницы или файла.",
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Полный адрес, например https://example.com "
                    "или file:///home/user/page.html.",
                }
            },
            "required": ["url"],
        },
        fetch_url,
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
class StepTokenUsage:
    """Расход токенов на один шаг цикла — одно обращение к модели.

    Часть входа может не появиться в `usage` вовсе: GigaChat отдаёт
    по-разному — то весь вход, то только некэшированную часть
    (`prompt_tokens` против `precached_prompt_tokens`; в прогоне в `http.log`
    при 5775 символах история дала 330 токенов, а при 6128 — 2471).
    Поэтому оценка на шагах с подозрительно большим кэшем не годится для
    проверки точности: такие шаги в сравнение не идут.
    """

    step: int
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    estimated: int = 0
    function: str = ""

    @property
    def total(self) -> int:
        return self.prompt + self.completion


@dataclass(frozen=True)
class AgentRun:
    """Один цикл агента по запросу пользователя вместе с расходом токенов.

    prompt/completion — сумма по шагам цикла (это и есть «запрос» в отчёте).
    Вход оценивается на первых шагах цикла: там в сообщениях уже вся история,
    но ещё нет результатов функций — именно этот вход растёт от обмена
    к обмену.
    """

    query: str
    steps: list[Step]
    answer: str
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    usages: list[StepTokenUsage] = field(default_factory=list)
    finish_reason: str = ""

    @property
    def last_usage(self) -> StepTokenUsage | None:
        return self.usages[-1] if self.usages else None

    @property
    def first_usage(self) -> StepTokenUsage | None:
        return self.usages[0] if self.usages else None

    @property
    def tools(self) -> list[str]:
        return [step.tool for step in self.steps]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def answer_tokens(self) -> int:
        """Сколько токенов приходится на сам ответ: последний шаг дописывает
        именно его, а предыдущие — просьбы вызвать функцию."""
        return self.usages[-1].completion if self.usages else 0

    @property
    def function_tokens(self) -> int:
        if len(self.usages) < 2:
            return 0
        return sum(usage.completion for usage in self.usages[:-1])

    @property
    def estimated_context(self) -> int:
        """Оценка входа первого шага — это та часть, которую мы формируем сами."""
        return self.first_usage.estimated if self.usages else 0

    @property
    def estimated_prompt(self) -> int:
        """Оценка входа последнего шага — им и закрывается запрос, способный
        вырасти на результат функции. Сравнивать только с фактом того же шага."""
        return self.usages[-1].estimated if self.usages else 0


@dataclass(frozen=True)
class ChatTurn:
    """Один обмен разговора: что спросил пользователь и что ответил агент.

    Шаги и расход токенов сохраняются в файл истории, но в модель уходят
    не все шаги: только вызовы инструментов из PERSISTED_TOOLS — со своим
    результатом, как они были в конце того цикла.
    """

    query: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    at: str = ""
    usage: AgentRun | None = None

    @property
    def tools(self) -> list[str]:
        return [step.tool for step in self.steps]

    def context_steps(self) -> list[Step]:
        """Шаги, результат которых повторяется модели в следующих запросах."""
        return [step for step in self.steps if step.tool in PERSISTED_TOOLS]

    def context_tokens(self) -> int:
        """Сколько занимают в истории результаты сохранённых инструментов."""
        return sum(estimate_tokens(_function_results(step))
                   for step in self.context_steps())

    def tokens(self) -> dict[str, int]:
        if self.usage is None:
            return {"prompt": 0, "completion": 0, "total": 0,
                    "answer": 0, "functions": 0, "context": 0}
        return {
            "prompt": self.usage.prompt_tokens,
            "completion": self.usage.completion_tokens,
            "total": self.usage.total_tokens,
            "answer": self.usage.answer_tokens,
            "functions": self.usage.function_tokens,
            "context": self.usage.estimated_context,
        }


@dataclass(frozen=True)
class Summary:
    """Сводка сжатой части разговора — то, чем история заменяется в запросе.

    Живёт отдельным файлом, а не в `history.json`: историю можно читать
    и разбирать целиком, а в контекст идёт только этот текст.
    """

    text: str = ""
    compressed_turns: int = 0
    created: str = ""
    updated: str = ""
    # Обмены, которые уже вошли в сводку: по числу видно, что сжато,
    # а что ещё уходит модели как есть.
    sources: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "version": SUMMARY_VERSION,
            "text": self.text,
            "compressed_turns": self.compressed_turns,
            "created": self.created,
            "updated": self.updated,
            "sources": self.sources,
        }


@dataclass(frozen=True)
class Fact:
    """Запись памяти: ключ–значение и откуда она взялась.

    Источник пишется в файл и в `/memory`: это и есть «явный выбор, что куда
    сохраняется» — видно, кто решил положить запись в этот слой.
    """

    key: str
    value: str
    source: str = SOURCE_RULE
    task: int = 0
    updated: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "task": self.task,
            "updated": self.updated,
        }


@dataclass(frozen=True)
class Layer:
    """Слой памяти с записями в порядке добавления."""

    facts: dict[str, Fact] = field(default_factory=dict)

    def merged(self, new: list[Fact]) -> Layer:
        facts = dict(self.facts)
        for fact in new:
            facts[fact.key] = fact
        return Layer(facts)

    def important(self, fact: Fact) -> "Layer":
        """Ставит запись в конец: в промпт первыми уходят последние."""
        facts = {key: value for key, value in self.facts.items() if key != fact.key}
        facts[fact.key] = fact
        return Layer(facts)

    def without(self, key: str) -> "Layer":
        return Layer({k: v for k, v in self.facts.items() if k != key})

    def selected(self, limit: int) -> list[Fact]:
        """Правило отбора слой → промпт: последние `limit` записей, новые выше."""
        return list(self.facts.values())[-limit:][::-1] if limit else []

    def block(self, title: str, limit: int) -> list[dict[str, Any]]:
        """Записи слоя отдельной парой сообщений перед историей разговора."""
        if not self.facts:
            return []
        items = self.selected(limit)
        if not items:
            return []
        text = title + "\n" + "\n".join(
            f"- {fact.key}: {_single_line(fact.value)[:FACT_VALUE_CHARS]}"
            for fact in items
        )
        return [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "Память слоя принята и учитывается."},
        ]

    def payload(self) -> dict[str, Any]:
        return {"facts": [fact.payload() for fact in self.facts.values()]}


STAGES = ("planning", "execution", "validation", "done")


@dataclass(frozen=True)
class TaskState:
    stage: str = "planning"
    step: str = "Подготовить план"
    expected: str = "Утверждение плана пользователем"
    summary: str = ""
    paused: bool = False
    transcript: str = ""

    def block(self) -> list[dict[str, Any]]:
        text = (f"Состояние задачи: этап {self.stage}; шаг: {self.step}; "
                f"ожидается: {self.expected}; резюме: {self.summary or 'не задано'}. "
                "Работай только в текущем этапе. Не объявляй переход выполненным: "
                "этап меняется только командой пользователя после проверки программой.\n"
                f"Сохранённый контекст задачи:\n{self.transcript or 'пока нет'}")
        return [{"role": "user", "content": text},
                {"role": "assistant", "content": "Состояние задачи принято."}]


def parse_task_state(value: Any) -> TaskState:
    if value is None:
        return TaskState()
    if not isinstance(value, dict) or value.get("stage") not in STAGES:
        raise AppError("Некорректное состояние задачи в файле памяти.")
    if any(not isinstance(value.get(k), str) for k in ("step", "expected", "summary")) or not isinstance(value.get("paused"), bool):
        raise AppError("Некорректные поля состояния задачи в файле памяти.")
    transcript = value.get("transcript", "")
    if not isinstance(transcript, str):
        raise AppError("Некорректный контекст задачи в файле памяти.")
    return TaskState(**{key: value[key] for key in ("stage", "step", "expected", "summary", "paused")}, transcript=transcript)


def transition_state(state: TaskState, target: str, confirmed: bool) -> TaskState:
    if state.paused:
        raise AppError("Задача на паузе: сначала выполните /state resume.")
    index = STAGES.index(state.stage)
    if index + 1 >= len(STAGES) or target != STAGES[index + 1]:
        raise AppError(f"Переход {state.stage} → {target} запрещён: этапы нельзя пропускать.")
    if target in ("execution", "done") and not confirmed:
        raise AppError("Требуется явное подтверждение: добавьте confirm.")
    steps = {"execution": ("Выполнить утверждённый план", "Завершение выполнения"),
             "validation": ("Проверить результат", "Подтверждение проверки пользователем"),
             "done": ("Задача завершена", "Нет действий")}
    step, expected = steps[target]
    return dataclasses.replace(state, stage=target, step=step, expected=expected)


@dataclass(frozen=True)
class TaskRecord:
    """Одна задача: номер, тема и её рабочая память до архивации."""

    number: int
    title: str
    opened: str = ""
    closed: str = ""
    facts: dict[str, Fact] = field(default_factory=dict)
    state: TaskState = field(default_factory=TaskState)

    def payload(self) -> dict[str, Any]:
        return {
            "state": dataclasses.asdict(self.state),
            "number": self.number,
            "title": self.title,
            "opened": self.opened,
            "closed": self.closed,
            "facts": [fact.payload() for fact in self.facts.values()],
        }


@dataclass(frozen=True)
class Memory:
    """Рабочая и долговременная память вместе с текущей задачей.

    Краткосрочная память живёт в `History` (history.json и summary.json):
    это тот же механизм, что в w2t2, поэтому формат файла не менялся.
    Рабочая память чистится вместе с задачей, долговременная переживает её.
    """

    working: Layer = field(default_factory=Layer)
    long: Layer = field(default_factory=Layer)
    task: int = 1
    title: str = ""
    archive: list[TaskRecord] = field(default_factory=list)
    state: TaskState = field(default_factory=TaskState)

    def layer(self, name: str) -> Layer:
        if name == WORKING:
            return self.working
        if name == LONG_TERM:
            return self.long
        raise AppError(f"Слой {name!r} не хранится отдельно.")

    def with_layer(self, name: str, layer: Layer) -> Memory:
        field_name = "working" if name == WORKING else "long"
        return dataclasses.replace(self, **{field_name: layer})

    def remember(self, facts: list[Fact]) -> Memory:
        """Раскладывает факты по слоям по метке задачи.

        Запись без задачи (task=0) — долговременная; запись текущей задачи —
        только рабочая. Ответ модели разбирается один раз в `_extract`, а при
        смене задачи те же факты раскладываются заново — с тем же правилом.
        """
        working = self.working.merged([f for f in facts if f.task == self.task])
        long = self.long.merged([f for f in facts if f.task == 0])
        return dataclasses.replace(self, working=working, long=long)

    def open_task(self, title: str) -> Memory:
        """Новая задача: рабочая память уезжает в архив, долговременная остаётся.

        В архив уходит только то, что накоплено по текущей задаче: новая задача
        начинается с пустой рабочей памятью, но обрывки, найденные моделью
        в репликах предыдущей беседы, сохранены в архиве, а не теряются.
        """
        stamp = datetime.now().astimezone().strftime(TIME_FORMAT)
        current = TaskRecord(self.task, self.title, closed=stamp,
                             facts=dict(self.working.facts), state=self.state)
        number = max([self.task, *(record.number for record in self.archive)]) + 1
        return Memory(Layer(), self.long, number, title,
                      [*self.archive, current])

    def reopen(self, number: int) -> Memory:
        """Вернуться к архивной задаче: её рабочая память снова становится текущей.

        История краткосрочного слоя к архивной задаче не хранится: она
        относится к беседе, а бесед может быть несколько. Поэтому история
        начинается заново, а конкретика возвращается рабочим слоем.
        """
        found = next((item for item in self.archive if item.number == number), None)
        if found is None:
            raise AppError(f"Задачи №{number} нет в архиве.")
        others = [item for item in self.archive if item.number != number]
        current = TaskRecord(self.task, self.title,
                             closed=datetime.now().astimezone().strftime(TIME_FORMAT),
                             facts=dict(self.working.facts), state=self.state)
        return Memory(Layer(dict(found.facts)), self.long, found.number, found.title,
                      [*others, current], found.state)

    def payload(self) -> dict[str, Any]:
        return {
            "version": MEMORY_VERSION,
            "task": self.task,
            "title": self.title,
            "state": dataclasses.asdict(self.state),
            "working": self.working.payload(),
            "long": self.long.payload(),
            "archive": [record.payload() for record in self.archive],
        }

    def blocks(self) -> list[dict[str, Any]]:
        """Правила отбора в промпт для двух слоёв, хранённых здесь."""
        return [
            *self.long.block("Долговременная память (профиль, решения, знания):",
                             LONG_TERM_ITEMS),
            *self.working.block("Рабочая память текущей задачи:", WORKING_ITEMS),
        ]


@dataclass(frozen=True)
class History:
    """Краткосрочная память: обмены текущей задачи, сводка и состояние беседы.

    Сжатие историю не удаляет: `turns` — весь разговор, а в контекст уходят
    только последние `keep` обменов, всё старшее заменяется `summary`.
    Внешние слои памяти (рабочая и долговременная) сюда не входят: они
    приходят в `messages` параметром.
    """

    started: str
    turns: list[ChatTurn] = field(default_factory=list)
    summary: Summary = field(default_factory=Summary)
    # Последние обмены уходят как есть; None — весь разговор целиком.
    keep: int | None = None
    strategy: str = DEFAULT_STRATEGY
    # Обмены, уже разложенные по слоям извлечением: по курсору извлечение
    # не повторяет работу и не переизвлекает факты заново.
    extracted: int = 0

    def with_strategy(self, strategy: str, keep: int | None = None) -> History:
        return dataclasses.replace(self, strategy=strategy,
                                   keep=self.keep if keep is None else keep)

    def split(self) -> tuple[int, int]:
        """Граница сжатого и живого: индексы [0, cut) — в сводке, [cut, n) — как есть."""
        total = len(self.turns)
        if self.keep is None:
            return 0, total
        cut = max(0, total - self.keep)
        return cutoff(self.summary, cut), total

    def context_turns(self) -> list[ChatTurn]:
        """Обмены, которые уходят модели дословно."""
        cut, total = self.split()
        return self.turns[cut:total]

    def pending_turns(self) -> list[ChatTurn]:
        """Обмены, которые можно сжать: всё до начала живой части."""
        cut, _ = self.split()
        return self.turns[self.summary.compressed_turns:cut]

    def messages(self, layers: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Контекст для модели: блоки слоёв, сводка, затем последние обмены.

        Порядок задан как явное правило: сначала то, что помнится дольше
        (долговременная, рабочая), затем ход текущего диалога. Каждый блок —
        пара сообщений user/assistant, чтобы он не путался с вопросом.
        """
        messages: list[dict[str, Any]] = list(layers or [])
        if self.summary.text:
            messages.extend(summary_messages(self.summary))
        turns = self.context_turns()
        if self.strategy == STRATEGY_SLIDING:
            turns = self.turns[-(self.keep or DEFAULT_KEEP_TURNS):]
        for turn in turns:
            messages.append({"role": "user", "content": turn.query})
            for step in turn.context_steps():
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": step.tool,
                                     "arguments": step.arguments or {}},
                })
                messages.append({"role": "function", "name": step.tool,
                                 "content": _function_results(step)})
            if turn.answer:
                messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def extended(self, turn: ChatTurn) -> History:
        return dataclasses.replace(self, turns=[*self.turns, turn])

    def compressed(self, summary: Summary) -> History:
        return dataclasses.replace(self, summary=summary)

    def without_old(self, older: int) -> History:
        """Убирает первые `older` обменов из истории: они уже в слоях памяти.

        История — это хранилище краткосрочного слоя, а не лог всех разговоров:
        факт, разложенный в рабочий или долговременный слой, не должен ещё
        и лежать в живом окне. Так слои не дублируются и отключение любого
        слоя действительно убирает свои факты. Сводка не трогается: на неё
        не влияет, сколько обменов живёт в истории прямо сейчас.
        """
        older = max(0, min(older, len(self.turns)))
        return dataclasses.replace(
            self,
            turns=self.turns[older:],
            extracted=max(0, self.extracted - older),
        )

    def marked(self, extracted: int) -> History:
        return dataclasses.replace(self, extracted=extracted)


@dataclass(frozen=True)
class Options:
    env_path: Path
    provider: str
    model: str
    log_file: Path
    http_log_file: Path
    history_file: Path
    summary_file: Path
    work_file: Path
    long_file: Path
    use_cache: bool
    model_limit: int = DEFAULT_MODEL_LIMIT
    # Сколько последних обменов держать как есть; None — не сжимать вообще.
    keep_turns: int | None = DEFAULT_KEEP_TURNS
    # Через сколько несжатых обменов срабатывает автосжатие.
    compress_every: int = DEFAULT_COMPRESS_EVERY
    # Сколько слов-заполнителей вбросить в историю сразу после запуска
    # (режим /fill до первой строки): нужно для сценария «за лимитом».
    fill_words: int = 0
    strategy: str = DEFAULT_STRATEGY
    window: int = DEFAULT_KEEP_TURNS
    branches_file: Path = DEFAULT_BRANCHES_FILE
    # Извлечение фактов моделью: это лишний запрос на каждый обмен.
    extract: bool = True
    # Извлечённый факт убирается из истории: слои не дублируются.
    sync_short: bool = True
    command_prefix: bool = False

    @property
    def compression(self) -> bool:
        return self.keep_turns is not None

    def trigger(self) -> int:
        """Порог автосжатия: одновременно работающий и keep, и every.

        Сжать всё, кроме живых keep, можно только тогда, когда несжатого
        накопилось больше keep: иначе сжатие съело бы живой контекст.
        """
        keep = self.keep_turns or 0
        return max(self.compress_every, keep + 1)


def cutoff(summary: Summary, cut: int) -> int:
    """Граница не может зайти за уже сжатое и за живой хвост."""
    return max(summary.compressed_turns, cut)


def update_facts(memory: Memory, user_message: str) -> list[Fact]:
    """Явно заданные правила извлечения — быстрый путь и страховка модели.

    Шаблоны те же, что в w2t5 (факты Дня 10), но теперь с раскладкой по слоям:
    срок, бюджет, ограничение, регион и встреча — данные текущей задачи,
    имя, решение и формат — долговременная память. Нерегулярные фразы разбирает
    модель: см. `extract_facts`.
    """
    patterns = {
        "срок": (r"(?:срок|дедлайн).*?(\d{1,2}\s+\w+\s+\d{4})", WORKING),
        "бюджет": (r"бюджет.*?(\d[\d\s]*руб(?:лей|ля|\.)?)", WORKING),
        "ограничение": (r"ограничени\w*\s*[:—-]?\s*(только[^.!?\n]+)", WORKING),
        "имя": (r"зовут\s*[:—-]?\s*([А-ЯЁA-Z][\w-]*)", LONG_TERM),
        "решение": (r"решение\s*[:—-]?\s*([^.!?\n]+)", LONG_TERM),
        "формат": (r"(?:формат|выгрузк\w+).*?(CSV[^.!?\n]+)", LONG_TERM),
        "регион": (r"регион.*?[:—-]\s*([^.!?\n]+)", WORKING),
        "встреча": (r"встреч\w*.*?(во\s+вторник\s+в\s+\d{1,2}:\d{2})", WORKING),
    }
    stamp = datetime.now().astimezone().strftime(TIME_FORMAT)
    found: list[Fact] = []
    for key, (pattern, layer) in patterns.items():
        match = re.search(pattern, user_message, re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        task = 0 if layer == LONG_TERM else memory.task
        current = (memory.long if layer == LONG_TERM else memory.working).facts.get(key)
        if current is not None and current.value == value:
            continue
        found.append(Fact(key, value, SOURCE_RULE, task, stamp))
    return found


def extract_facts(query: str, answer: str, task: int, client: GigaChatClient,
                  access_token: str, options: Options,
                  context: tuple[int, str]) -> tuple[list[Fact], AgentRun]:
    """Модель раскладывает один обмен по слоям.

    Правила знают известные шаблоны, но не любой текст, поэтому основной
    способ — модель. Не разобрали ответ или ошибка API — пустой список:
    рассыпавшееся извлечение не должно ронять диалог.
    """
    prompt = (f"{EXTRACT_PROMPT}\n\nПользователь: {_cut(query, COMPRESS_CHARS_PER_TURN)}\n"
              f"Ассистент: {_cut(answer, COMPRESS_CHARS_PER_TURN)}")
    # run_agent подписывает запрос коротким именем шага (query_number и
    # текстом в HTTP-лог), поэтому подсказка передаётся отдельно: строка
    # «извлечение фактов» вместо подсказки отправляла модели бессмыслицу.
    run = run_agent("", client, access_token, options.model, options.log_file,
                    History(""), context=(context[0], "извлечение фактов"),
                    include_tools=False, prompt=prompt)
    facts: list[Fact] = []
    stamp = datetime.now().astimezone().strftime(TIME_FORMAT)
    for item in _json_facts(run.answer):
        layer = str(item.get("layer") or "").strip().lower()
        key = str(item.get("key") or "").strip().lower()[:40]
        value = str(item.get("value") or "").strip()
        if not key or not value:
            continue
        # Данные задачи лежат в рабочем слое только у текущей задачи; к старой
        # задаче они не относятся и ждут её в архиве (см. Memory.remember).
        fact = Fact(key, value, SOURCE_MODEL, task, stamp)
        if layer in ("long", "долг", "долговременная", "долговременный"):
            facts.append(Fact(key, value, SOURCE_MODEL, 0, stamp))
        elif layer in ("work", "рабочая", "рабочий", "задача"):
            facts.append(fact)
    return facts, run


def _json_facts(text: str) -> list[dict[str, Any]]:
    """Список фактов из ответа модели: JSON в тексте или в блоке ```json.

    Модель может ответить списком, объектом с полем facts или числом
    с объяснением рядом, поэтому разбор не падает ни на одном из этих ответов.
    """
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    for candidate in (raw, *_json_candidates(raw)):
        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = payload.get("facts") if isinstance(payload, dict) else payload
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _json_candidates(raw: str) -> list[str]:
    """Куски текста, которые могут оказаться JSON: от каждой { до каждой }."""
    candidates: list[str] = []
    opening = [index for index, char in enumerate(raw) if char == "{"]
    closing = [index for index, char in enumerate(raw) if char == "}"]
    for start in opening:
        for end in reversed(closing):
            if end > start:
                candidates.append(raw[start:end + 1])
                break
    return candidates


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def summary_messages(summary: Summary) -> list[dict[str, Any]]:
    """Сводка сжатой части разговора как пара сообщений истории."""
    return [
        {"role": "user", "content": "Сводка разговора:\n" + summary.text},
        {"role": "assistant", "content": "Сводка принята и учитывается."},
    ]


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


COMMAND_NAMES = tuple(item[0].split()[0] for item in COMMAND_HELP) + ("/profile", "/personal-test", "/state")
PROFILE_COMMANDS = ("show", "list", "select", "use", "set")
STATE_COMMANDS = ("show", "help", "pause", "resume", "step", "expected", "summary", "go")


def resolve_command(query: str, prefix: bool) -> str:
    if not query.startswith("/"):
        return query
    parts = query.split(None, 1)
    token = parts[0].lower()
    matches = [name for name in COMMAND_NAMES if name.startswith(token)]
    if token in COMMAND_NAMES:
        name = token
    elif prefix and len(matches) == 1:
        name = matches[0]
    elif prefix and matches:
        raise AppError("Неоднозначная команда: " + ", ".join(matches))
    else:
        raise AppError(f"Неизвестная команда: {parts[0]}. Введите /help для списка команд.")
    remainder = query[len(parts[0]):]
    if prefix and name in ("/profile", "/state") and remainder.strip():
        commands = PROFILE_COMMANDS if name == "/profile" else STATE_COMMANDS
        match = re.match(r"(\s+)(\S+)(.*)", remainder, re.DOTALL)
        if match:
            sub = match.group(2).lower()
            candidates = [value for value in commands if value.startswith(sub)]
            if sub in commands:
                chosen = sub
            elif len(candidates) == 1:
                chosen = candidates[0]
            elif candidates:
                raise AppError(f"Неоднозначная подкоманда {name}: " + ", ".join(candidates))
            else:
                raise AppError(f"Неизвестная подкоманда {name}: {match.group(2)}.")
            remainder = match.group(1) + chosen + match.group(3)
    return name + remainder


def ask_command(prompt: str, prefix: bool) -> str:
    if prefix:
        return ask(prompt)
    try:
        import readline
    except ImportError:
        return ask(prompt)
    previous = readline.get_completer()
    delimiters = readline.get_completer_delims()

    def complete(text: str, state: int) -> str | None:
        line = readline.get_line_buffer()[:readline.get_endidx()]
        words = line.split()
        if len(words) == 1 and not line[-1:].isspace():
            candidates = COMMAND_NAMES if line.startswith("/") else ()
        elif words and words[0].lower() in ("/profile", "/state") and (len(words) == 1 or
                len(words) == 2 and not line[-1:].isspace()):
            candidates = PROFILE_COMMANDS if words[0].lower() == "/profile" else STATE_COMMANDS
        elif len(words) >= 2 and words[0].lower() == "/profile" and words[1].lower() == "use" and (
                len(words) == 2 or len(words) == 3 and not line[-1:].isspace()):
            try:
                candidates = profile_names()
            except AppError:
                candidates = []
        else:
            candidates = ()
        matches = [candidate for candidate in candidates if candidate.lower().startswith(text.lower())]
        return matches[state] if state < len(matches) else None

    try:
        readline.set_completer_delims(" \t\n")
        readline.set_completer(complete)
        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        return ask(prompt)
    finally:
        readline.set_completer(previous)
        readline.set_completer_delims(delimiters)


def parse_args(argv: list[str]) -> Options:
    env_path: Path | None = None
    provider = DEFAULT_PROVIDER
    model = DEFAULT_MODEL
    log_file = DEFAULT_LOG_FILE
    http_log_file = DEFAULT_HTTP_LOG_FILE
    history_file = DEFAULT_HISTORY_FILE
    summary_file = DEFAULT_SUMMARY_FILE
    work_file = DEFAULT_WORK_FILE
    long_file = DEFAULT_LONG_FILE
    branches_file = DEFAULT_BRANCHES_FILE
    model_limit = DEFAULT_MODEL_LIMIT
    fill_words = 0
    keep_turns: int | None = DEFAULT_KEEP_TURNS
    compress_every = DEFAULT_COMPRESS_EVERY
    strategy = DEFAULT_STRATEGY
    window = DEFAULT_KEEP_TURNS
    use_cache = True
    extract = True
    sync_short = True
    command_prefix = False
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg.split("=", 1)[0] in ("--history", "--summary", "--work", "--long", "--branches"):
            raise AppError(f"{arg.split('=', 1)[0]} несовместим с профилями: файлы памяти хранятся отдельно для каждого профиля.")
        if arg == "--command-prefix":
            command_prefix = True
        elif arg == "--no-cache":
            use_cache = False
        elif arg == "--no-extract":
            extract = False
        elif arg == "--keep-short":
            # Сравнение со слоями: факт остаётся и в истории, и в слое.
            sync_short = False
        elif arg == "--no-compress":
            keep_turns = None
            strategy = STRATEGY_SLIDING
        elif arg in ("--strategy", "--window"):
            if not args:
                raise AppError(f"После {arg} требуется значение.")
            value = args.pop(0)
            if arg == "--strategy":
                if value not in STRATEGIES:
                    raise AppError(f"Стратегия должна быть одной из: {', '.join(STRATEGIES)}")
                strategy = value
                if value == STRATEGY_SLIDING:
                    keep_turns = None
            else:
                window = _positive_int(arg, value)
                keep_turns = window
        elif arg in ("--env", "--provider", "--model", "--log", "--http-log", "--history",
                     "--summary", "--work", "--long", "--branches", "--limit",
                     "--fill", "--keep", "--every", "--strategy", "--window"):
            if not args:
                raise AppError(f"После {arg} требуется значение.")
            value = args.pop(0)
            if arg == "--env":
                env_path = Path(value).expanduser()
            elif arg == "--provider":
                provider = value.lower()
                if provider not in PROVIDERS:
                    raise AppError(f"Провайдер должен быть одним из: {', '.join(PROVIDERS)}")
            elif arg == "--model":
                model = value
            elif arg == "--log":
                log_file = Path(value).expanduser()
            elif arg == "--history":
                history_file = Path(value).expanduser()
            elif arg == "--summary":
                summary_file = Path(value).expanduser()
            elif arg == "--work":
                work_file = Path(value).expanduser()
            elif arg == "--long":
                long_file = Path(value).expanduser()
            elif arg == "--branches":
                branches_file = Path(value).expanduser()
            elif arg == "--limit":
                model_limit = _positive_int(arg, value)
            elif arg == "--fill":
                fill_words = _positive_int(arg, value)
            elif arg == "--keep":
                keep_turns = _positive_int(arg, value)
            elif arg == "--every":
                compress_every = _positive_int(arg, value)
            elif arg == "--strategy":
                if value not in STRATEGIES:
                    raise AppError(f"Стратегия должна быть одной из: {', '.join(STRATEGIES)}")
                strategy = value
            elif arg == "--window":
                window = _positive_int(arg, value)
                keep_turns = window
            else:
                http_log_file = Path(value).expanduser()
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env="):]).expanduser()
        elif arg.startswith("--provider="):
            provider = arg[len("--provider="):].lower()
            if provider not in PROVIDERS:
                raise AppError(f"Провайдер должен быть одним из: {', '.join(PROVIDERS)}")
        elif arg.startswith("--model="):
            model = arg[len("--model="):]
        elif arg.startswith("--log="):
            log_file = Path(arg[len("--log="):]).expanduser()
        elif arg.startswith("--history="):
            history_file = Path(arg[len("--history="):]).expanduser()
        elif arg.startswith("--summary="):
            summary_file = Path(arg[len("--summary="):]).expanduser()
        elif arg.startswith("--work="):
            work_file = Path(arg[len("--work="):]).expanduser()
        elif arg.startswith("--long="):
            long_file = Path(arg[len("--long="):]).expanduser()
        elif arg.startswith("--branches="):
            branches_file = Path(arg[len("--branches="):]).expanduser()
        elif arg.startswith("--limit="):
            model_limit = _positive_int("--limit", arg[len("--limit="):])
        elif arg.startswith("--fill="):
            fill_words = _positive_int("--fill", arg[len("--fill="):])
        elif arg.startswith("--keep="):
            keep_turns = _positive_int("--keep", arg[len("--keep="):])
        elif arg.startswith("--every="):
            compress_every = _positive_int("--every", arg[len("--every="):])
        elif arg.startswith("--strategy="):
            strategy = arg[len("--strategy="):]
            if strategy not in STRATEGIES:
                raise AppError(f"Стратегия должна быть одной из: {', '.join(STRATEGIES)}")
        elif arg.startswith("--window="):
            window = _positive_int("--window", arg[len("--window="):])
            keep_turns = window
        elif arg.startswith("--http-log="):
            http_log_file = Path(arg[len("--http-log="):]).expanduser()
        else:
            raise AppError(
                f"Неизвестный аргумент: {arg}. Доступны: "
                + ", ".join(flag.split()[0] if " " in flag else flag
                            for flag, _ in OPTION_HELP)
                + "."
            )
    return Options(
        env_path or find_env_file(ENV_SEARCH_START), provider, model, log_file,
        http_log_file, history_file, summary_file, work_file, long_file,
        use_cache, model_limit, keep_turns, compress_every, fill_words, strategy,
        window, branches_file, extract, sync_short, command_prefix,
    )


def _positive_int(flag: str, value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise AppError(f"{flag}: ожидается целое число, получено {value!r}.") from exc
    if number < 1:
        raise AppError(f"{flag}: ожидается число больше нуля, получено {number}.")
    return number


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
    # В журнал — только начало и размер: текст страницы бывает в сотни тысяч
    # символов, а журнал нужен для того, что и чем вызывали.
    recorded = (f"{text[:LOG_ARGUMENT_CHARS]}… ({len(text)} символов)"
                if len(text) > LOG_ARGUMENT_CHARS else text)
    log_call(CallLogEntry(at, query, step, tool.name, _format_arguments(arguments),
                          recorded),
             log_file)
    return Step(tool=tool.name, arguments=arguments, result=text)


def _format_arguments(arguments: dict[str, Any]) -> str:
    """Аргументы для журнала и консоли: длинные значения обрезаются.

    Адрес страницы с токеном в параметрах и картинка в base64 не должны
    раздувать журнал вызовов и вывод диалога на мегабайты.
    """
    if not arguments:
        return "—"
    short = {
        key: (value[:LOG_ARGUMENT_CHARS] + "…"
              if isinstance(value, str) and len(value) > LOG_ARGUMENT_CHARS
              else value)
        for key, value in arguments.items()
    }
    return json.dumps(short, ensure_ascii=False)


def _function_results(step: Step) -> str:
    """Результат функции для модели.

    API разбирает content сообщения роли function как JSON и отвечает 422
    на голую строку вроде `0xff`, поэтому результат заворачивается в объект.
    """
    if step.error:
        return json.dumps({"error": step.error}, ensure_ascii=False)
    return json.dumps({"result": step.result}, ensure_ascii=False)


def _tokens(response: dict[str, Any]) -> tuple[int, int, int]:
    """prompt, completion и закэшированные токены из `usage` ответа."""
    usage = response.get("usage") or {}
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(usage.get("precached_prompt_tokens") or 0),
    )


def _veai_prompt(messages: list[dict[str, Any]]) -> str:
    """Преобразует сообщения GigaChat в один OpenAI-compatible prompt Veai."""
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        if content:
            parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _chat_response(client: Any, model: str, messages: list[dict[str, Any]],
                   body: dict[str, Any], query_number: int, step: int) -> dict[str, Any]:
    """Единый вызов двух провайдеров: GigaChat получает messages, Veai — prompt."""
    if isinstance(client, VeaiClient):
        return client.chat(model=model, prompt=_veai_prompt(messages), max_tokens=4096)
    return client.chat_messages(None, body)


def run_agent(
    query: str,
    client: Any,
    access_token: str,
    model: str,
    log_file: Path,
    history: History,
    on_step: Callable[[Step, int], None] | None = None,
    context: tuple[int, str] = (0, ""),
    include_tools: bool = True,
    layers: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
) -> AgentRun:
    """Цикл агента: запрос → вызов инструментов → ответ, не больше MAX_STEPS шагов.

    history — прошлые обмены разговора: они уходят модели перед новым запросом,
    поэтому она продолжает тему, а не начинает беседу заново.
    on_step вызывается на каждый выполненный инструмент: по нему вызывающий
    код показывает шаги по ходу работы, а не только после ответа.
    context — номер запроса в сеансе и текст запроса: они уходят в HTTP-лог,
    чтобы в нём было видно, какой шаг какого диалога описывает строка.
    include_tools=False — обычный ответ модели без инструментов: так делается
    сжатие истории, где функции только мешают пересказу.
    layers — блоки слоёв памяти перед историей; порядок и состав они задают сами.
    prompt — готовая подсказка вместо текста запроса: так служебные вызовы
    (извлечение, детектор темы) пишут в лог своё короткое имя, а модели
    уходит настоящий текст.

    Оценка входа считается на каждом шаге по тем же сообщениям, что идут в API,
    и сохраняется рядом с фактом того же шага (`StepTokenUsage.estimated`).
    Сравнивать их надо именно на одном шаге: после вызова инструмента
    в сообщениях появляется его результат, и «оценка до цикла» против
    «факта после цикла» — уже разные запросы.
    """
    query_number, user_query = context
    started = time.time()
    content = user_query if prompt is None else prompt
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history.messages(layers),
        {"role": "user", "content": content},
    ]
    steps: list[Step] = []
    usages: list[StepTokenUsage] = []
    prompt_tokens = completion_tokens = cached_tokens = 0

    for step_number in range(1, MAX_STEPS + 1):
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if include_tools and not isinstance(client, VeaiClient):
            body["functions"] = [tool.schema() for tool in TOOLS]
            body["function_call"] = "auto"
        if isinstance(client, VeaiClient):
            include_tools = False
        if hasattr(client, "log_context"):
            client.log_context = {
            "query_number": query_number,
            "query": user_query,
            "step": step_number,
        }
        try:
            # Токен не передаём: клиент сам обновит его, если GigaChat
            # ответит 401 после долгого простоя.
            response = _chat_response(client, model, messages, body,
                                      query_number, step_number)
        except (GigaChatError, VeaiError) as exc:
            raise AppError(f"{type(client).__name__} не ответил: {exc}") from exc
        sent, received, cached = _tokens(response)
        prompt_tokens += sent
        completion_tokens += received
        cached_tokens += cached

        choices = response.get("choices") or []
        if not choices:
            raise AppError(f"Ответ GigaChat без choices: {response}")
        message = choices[0].get("message") or {}
        function_call = message.get("function_call")
        name = str(function_call.get("name", "")) if function_call else ""
        usages.append(StepTokenUsage(
            step=step_number, prompt=sent, completion=received, cached=cached,
            estimated=estimate_body_tokens(body), function=name,
        ))
        if not function_call:
            return AgentRun(
                query, steps, (message.get("content") or "").strip(),
                time.time() - started, prompt_tokens, completion_tokens,
                cached_tokens, usages, str(choices[0].get("finish_reason") or ""),
            )

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
        query, steps, "", time.time() - started, prompt_tokens, completion_tokens,
        cached_tokens, usages,
    )


def print_event(step: Step, number: int) -> None:
    """Строка о вызове инструмента по ходу цикла."""
    print(
        f"  шаг {number}: {step.tool}({_format_arguments(step.arguments)})"
        f" → {'ошибка: ' if step.error else ''}{step.error or step.result}",
        flush=True,
    )


def print_answer(run: AgentRun, history_left: int, history_limit: int) -> None:
    """Ответ и расход токенов: цикл, сеанс, история и сам ответ."""
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
    print(f"Время: {run.seconds:.1f} с")
    print("Токены:")
    for line in token_lines(run):
        print(f"  {line}")
    # Контекст здесь — история до этого обмена: после ответа она выросла
    # на пару «запрос–ответ», и эти строки печатаются в main.
    print_share_line(history_left, history_limit)
    if run.answer:
        # Следующий запрос уйдёт в модель вместе с историей: длина пары
        # оценивается той же формулой, что и вся оценка.
        print("  следующий вопрос уйдёт с историей: " + _context_line(
            history_left
            + estimate_message_tokens({"role": "user", "content": run.query})
            + estimate_message_tokens({"role": "assistant", "content": run.answer}),
            history_limit))


def token_lines(run: AgentRun) -> list[str]:
    """Строки отчёта по токенам одного цикла — их же показывает /limits.

    Оценка сравнивается с фактом того же шага и по шагам отдельно: после
    вызова инструмента в сообщениях появляется его результат, поэтому
    расхождение первого и последнего шага — разные числа и разная причина.
    """
    steps = len(run.usages)
    return [
        f"запрос: {run.prompt_tokens} вход + {run.completion_tokens} выход "
        f"= {run.total_tokens} за {steps} обр. (вход — только некэшированная часть)",
        f"ответ модели: {run.answer_tokens} выход"
        + (f" (ещё {run.function_tokens} — на просьбы вызвать функцию)"
           if run.function_tokens else ""),
        *_estimate_lines(run),
    ]


def _estimate_lines(run: AgentRun) -> list[str]:
    """Размер входа по шагам: оценка против факта и рост от шага к шагу."""
    if not run.usages:
        return ["оценка входа: —"]
    lines = _step_lines(run.usages, indent="")
    if len(run.usages) > 1:
        first, last = run.usages[0], run.usages[-1]
        lines.append(
            f"оценка входа по шагам: первый {_percent(first.prompt, first.estimated)}, "
            f"последний {_percent(last.prompt, last.estimated)}"
        )
    return lines


def _step_lines(usages: list[StepTokenUsage], indent: str = "  ",
                limit: int = 0) -> list[str]:
    """Строки по шагам: сколько контекста ушло в модель, рост и запас до лимита.

    Показываются оба числа: оценка — это полный вход, именно она растёт
    от шага к шагу; факт `usage` — только некэшированная часть, и сравнивать
    его надо с оценкой того же шага. Рост считается по оценке, потому что
    факт в соседних шагах может быть посчитан по-разному.
    """
    lines = []
    previous = 0
    for usage in usages:
        growth = "" if not previous else f", +{usage.estimated - previous} к шагу выше"
        step = (f"{indent}шаг {usage.step}: контекст ~{usage.estimated}, "
                f"факт {usage.prompt} ({_percent(usage.prompt, usage.estimated)})"
                f"{growth}")
        if limit:
            step += f", до лимита {limit - usage.estimated}"
        if usage.function:
            step += f" → {usage.function}"
        lines.append(step)
        previous = usage.estimated
    return lines


def print_share_line(history_left: int, history_limit: int) -> None:
    """Насколько история близка к лимиту контекста модели."""
    print("  история разговора: " + _context_line(history_left, history_limit))


def context_left(tokens: int, history_limit: int) -> int:
    """Сколько токенов останется от лимита контекста."""
    return history_limit - tokens


def _context_line(tokens: int, history_limit: int) -> str:
    """Размер контекста в токенах, его доля от лимита и остаток.

    tokens — сколько займёт весь вход (история и запрос). Строка одна,
    потому что её печатают в нескольких местах: по ней видно и текущий
    размер контекста, и сколько осталось до отказа модели.
    """
    share = 100.0 * tokens / history_limit if history_limit else 0.0
    left = context_left(tokens, history_limit)
    if left <= 0:
        return (f"~{tokens} токенов ({share:.1f}% лимита {history_limit}), "
                f"лимит превышен на {-left}")
    return (f"~{tokens} токенов ({share:.1f}% лимита {history_limit}), "
            f"осталось {left}")


def _percent(actual: int, estimate: int) -> str:
    """Расхождение факта с оценкой.

    API называет в `prompt_tokens` то весь вход, то только некэшированную честь,
    поэтому «факта меньше оценки» — обычное дело, а не поломка формулы.
    """
    if not estimate:
        return "—"
    return f"{(actual - estimate) / estimate * 100:+.0f}%"


def compress_history(history: History, options: Options, client: GigaChatClient,
                     runs: list[AgentRun], access_token: str = "",
                     limit: int | None = None,
                     context: tuple[int, str] = (0, "")) -> History:
    """Пересказывает старые обмены моделью и дописывает пересказ в сводку.

    limit — сколько обменов сжать за раз; по умолчанию всё несжатое
    до живой части. Модель не сама решает, что помнить: ей дают готовый
    кусок диалога и просят сохранить конкретные детали.
    """
    pending = history.pending_turns()
    if not options.compression or not pending:
        return history
    chunk = pending if limit is None else pending[:limit]
    if not chunk:
        return history
    print(f"\nСжатие: пересказываю {len(chunk)} обменов "
          f"(сжато было {history.summary.compressed_turns})")
    prompt = _compress_prompt(history.summary, chunk)
    run = run_agent(prompt, client, access_token, options.model,
                    options.log_file, _summaryless(history),
                    context=(context[0], "сжатие истории"),
                    include_tools=False)
    runs.append(run)
    text = run.answer.strip()
    if not text:
        print("  модель не дала сводку — история осталась как была", file=sys.stderr)
        return history
    stamp = datetime.now().astimezone().strftime(TIME_FORMAT)
    summary = Summary(
        text=text,
        compressed_turns=history.summary.compressed_turns + len(chunk),
        created=history.summary.created or stamp,
        updated=stamp,
        sources=len(chunk),
    )
    print(f"  сводка: {run.prompt_tokens} вход + {run.completion_tokens} выход, "
          f"{len(text)} символов; сжато обменов: {summary.compressed_turns}")
    return history.compressed(summary)


def _summaryless(history: History) -> History:
    """Контекст без сводки: при сжатии она передаётся в самой подсказке."""
    return dataclasses.replace(history, summary=Summary())


def _compress_prompt(summary: Summary, chunk: list[ChatTurn]) -> str:
    """Подсказка для сжатия: прежняя сводка и кусок разговора."""
    parts = [COMPRESS_PROMPT, ""]
    if summary.text:
        parts.append("Уже сжато раньше (не повторяй дословно, только дополняй):")
        parts.append(summary.text[:COMPRESS_SUMMARY_CHARS])
        parts.append("")
    parts.append("Новые обмены:")
    for number, turn in enumerate(chunk, start=summary.compressed_turns + 1):
        parts.append(f"#{number} Пользователь: {_cut(turn.query, COMPRESS_CHARS_PER_TURN)}")
        steps = [f"{step.tool} → {_cut(step.result, 200)}"
                 for step in turn.context_steps()]
        if steps:
            parts.append("Инструменты: " + "; ".join(steps))
        if turn.answer:
            parts.append(f"Агент: {_cut(turn.answer, COMPRESS_CHARS_PER_TURN)}")
    return "\n".join(parts)


def _cut(text: str, limit: int) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    head = text[:limit].split("\n")[0]
    return head + "…"


def load_details(path: Path) -> list[dict[str, str]]:
    """Детали для проверки памяти: id, вид, сама деталь и вопрос к ней."""
    if not path.is_file():
        raise AppError(f"Не найден файл деталей: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError(f"Не удалось прочитать {path}: {exc}") from exc
    items = payload.get("details") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise AppError(f"В {path} нет списка details.")
    return [item for item in items if isinstance(item, dict)]


# Состояния эксперимента: полная память и по одной исключённой. Порядок —
# от самого долгого к самому быстрому, как в CLEANUP_ORDER.
CLEANUP_ALL = "all"
CLEANUP_STATES = {
    CLEANUP_ALL: (),
    "-long": (LONG_TERM,),
    "-work": (LONG_TERM, WORKING),
    "-short": (LONG_TERM, WORKING, SHORT_TERM),
}


def clear_layer(query: str, memory: Memory, history: History,
                options: Options) -> tuple[Memory, History]:
    """/clear-layer short|work|long — физически стереть один слой, остальные целы.

    Рабочий слой вместе с архивом задач: архив тоже хранится в work.json и
    относится к рабочей памяти. Долговременный очищается целиком.
    """
    rest = query[len(CLEAR_LAYER_COMMAND):].strip()
    if not rest:
        print(f"Использование: {CLEAR_LAYER_COMMAND} short|work|long")
        return memory, history
    layer = _layer_by_name(rest)
    if layer == SHORT_TERM:
        clear_history(options.history_file)
        clear_summary(options.summary_file)
        history = History(datetime.now().astimezone().strftime(TIME_FORMAT),
                          keep=options.keep_turns, strategy=options.strategy)
        print(f"Очищена «{LAYER_TITLES[layer]}»: история и сводка удалены.")
        return memory, history
    if layer == WORKING:
        memory = dataclasses.replace(memory, working=Layer(), archive=[])
    else:
        memory = memory.with_layer(LONG_TERM, Layer())
    save_memory(memory, options.work_file, options.long_file)
    print(f"Очищена «{LAYER_TITLES[layer]}».")
    return memory, history


def cleanup_memory(memory: Memory, state: str) -> Memory:
    """Копия памяти с фактически удалёнными слоями.

    Исходная память сеанса не меняется: удаление происходит только в копии
    эксперимента, но очищенный слой получает пустое хранилище.
    """
    removed = CLEANUP_STATES[state]
    return dataclasses.replace(
        memory,
        working=Layer() if WORKING in removed else memory.working,
        long=Layer() if LONG_TERM in removed else memory.long,
    )


def check_details(details: list[dict[str, str]], client: GigaChatClient,
                  runs: list[AgentRun], history: History, memory: Memory,
                  options: Options, received: int,
                  layering: tuple[Memory, History] | None = None
                  ) -> list[dict[str, Any]]:
    """Задаёт вопрос по каждой детали и запоминает ответ и вклад слоёв.

    layering — набор слоёв, который уходит в контекст этих вопросов; без него
    берутся текущие память и история. Обмены в историю не дописываются:
    иначе каждый вопрос менял бы состояние, которое мы сравниваем.
    """
    use_memory, use_history = layering or (memory, history)
    results: list[dict[str, Any]] = []
    for item in details:
        question = str(item.get("question") or "") or str(item.get("expected") or "")
        if not question:
            continue
        # include_tools=False и пустой log_file: проверочный вопрос не должен
        # ни вызывать инструменты (агент или к current_time), ни попадать
        # в журнал вызовов основной сессии.
        run = run_agent(_strict_check_prompt(question), client, "", options.model,
                        Path(os.devnull), use_history,
                        context=(received, question), layers=use_memory.blocks(),
                        include_tools=False)
        runs.append(run)
        fact = str(item.get("expected") or "")
        answer = run.answer.strip()
        hit = _mentions(answer, fact)
        origin = _answer_origin(answer, fact, use_memory, use_history)
        results.append({
            "id": item.get("id", ""),
            "kind": item.get("kind", ""),
            "fact": fact,
            "question": question,
            "answer": answer,
            "hit": hit,
            "origin": origin,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
        })
    return results


def _mentions(answer: str, fact: str) -> str:
    """Есть ли деталь в ответе: «прямо» — дословно, «частично» — по словам.

    Судию-модель не привлекаем: сравниваем буквальное вхождение, а при
    его отсутствии — все значимые слова детали и числа.
    """
    low = answer.lower()
    needle = fact.lower().strip()
    if needle and needle in low:
        return "прямо"
    words = [word for word in re.findall(r"\w+", needle) if len(word) > 2]
    numbers = re.findall(r"\d[\d\s.:]*", needle)
    matches_words = bool(words) and all(word in low for word in words)
    matches_numbers = bool(numbers) and any(
        number.strip().rstrip(".,:") in low for number in numbers
    )
    if matches_words or matches_numbers:
        return "частично"
    return ""


def _answer_origin(answer: str, fact: str, memory: Memory,
                   history: History) -> str:
    """Отличает воспроизведение из памяти, отказ и догадку.

    Модель-судья здесь не нужна: сравниваем ответ с ожидаемым фактом и
    отдельно ищем тот же факт в переданных блоках. Совпадение без факта
    в блоках — `guess`, а не успешное чтение памяти.
    """
    normalized = answer.strip().lower()
    if any(marker in normalized for marker in
           ("нет данных", "не знаю", "нет информации", "отсутствует")):
        return "no_data"
    if not _mentions(answer, fact):
        return "other"
    needle = _normalize_memory_text(fact)
    if needle and any(needle in _normalize_memory_text(text)
                      for text in _memory_texts(memory, history)):
        return "memory"
    return "guess"


def _normalize_memory_text(text: str) -> str:
    return " ".join(str(text).lower().replace("ё", "е").split())


def _memory_texts(memory: Memory, history: History) -> list[str]:
    """Точный набор текстов, который передавался модели в проверке."""
    return [str(message.get("content") or "")
            for message in (*memory.blocks(), *history.messages())]


def _strict_check_prompt(question: str) -> str:
    return f"{MEMORY_CHECK_PROMPT}\n\nВопрос: {question}"
def print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    """Таблица с выравниванием по ширине колонки — без внешних библиотек."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    print("  " + "  ".join(header.ljust(widths[index])
                          for index, header in enumerate(headers)).rstrip())
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(str(cell).ljust(widths[index])
                              for index, cell in enumerate(row)).rstrip())


def run_cleanup(details: list[dict[str, str]], client: GigaChatClient,
                runs: list[AgentRun], history: History, memory: Memory,
                options: Options, received: int) -> list[dict[str, Any]]:
    """Фактически удаляет слои в независимых копиях и задаёт вопросы."""
    if not history.turns and not (memory.working.facts or memory.long.facts):
        print("Память пуста: сначала наполните хотя бы один слой.")
        return []
    results: list[dict[str, Any]] = []
    for state in CLEANUP_STATES:
        # При очистке краткосрочного слоя используется пустая история, без
        # заглушки: заглушка сама стала бы данными краткосрочной памяти.
        short = (History("", keep=options.keep_turns, strategy=options.strategy)
                 if SHORT_TERM in CLEANUP_STATES[state] else history)
        layers = cleanup_memory(memory, state)
        print(f"\nСостояние {state}: краткосрочная "
              f"{'удалена' if SHORT_TERM in CLEANUP_STATES[state] else 'сохранена'}, "
              f"рабочая {'удалена' if WORKING in CLEANUP_STATES[state] else 'сохранена'}, "
              f"долговременная {'удалена' if LONG_TERM in CLEANUP_STATES[state] else 'сохранена'}")
        checked = check_details(details, client, runs, short, layers, options,
                                received, layering=(layers, short))
        received += len(checked)
        tokens = layer_tokens(layers, short)
        for item in checked:
            item["state"] = state
            item["context_tokens"] = tokens["total"]
        results.extend(checked)
        found = sum(1 for item in checked if item["origin"] == "memory")
        print(f"  из памяти {found} из {len(checked)}, контекст ~{tokens['total']} токенов")
    print()
    print_table(
        ("#", "Деталь", "Ожидалось"),
        [(item.get("id", ""), item.get("kind", ""), item.get("expected", ""))
         for item in details],
    )
    for state in CLEANUP_STATES:
        rows = [(item["id"], item["answer"][:60], item.get("origin", "other"))
                for item in results if item["state"] == state]
        print(f"\n  {state}")
        print_table(("Деталь", "Ответ модели", "Происхождение"), rows)
    print("\nСводка по состояниям:")
    print_table(("Состояние", "Найдено", "Из", "Вход на вопросы", "Контекст"),
                [(state, sum(1 for item in results
                             if item["state"] == state and item.get("origin") == "memory"),
                  len(details),
                  sum(item["prompt_tokens"] for item in results
                      if item["state"] == state),
                  results[-1]["context_tokens"] if results else 0)
                 for state in CLEANUP_STATES])
    return results


def save_cleanup(path: Path, results: list[dict[str, Any]],
                 memory: Memory, options: Options) -> None:
    """Результаты опыта копятся по состояниям в одном файле."""
    old: dict[str, Any] = {}
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
    states = old.get("states") or {}
    stamp = datetime.now().astimezone().strftime(TIME_FORMAT)
    for state in CLEANUP_STATES:
        rows = [item for item in results if item["state"] == state]
        if not rows:
            continue
        states[state] = {
            "model": options.model,
            "checked": stamp,
            "found": sum(1 for item in rows if item.get("origin") == "memory"),
            "guessed": sum(1 for item in rows if item.get("origin") == "guess"),
            "no_data": sum(1 for item in rows if item.get("origin") == "no_data"),
            "details": len(rows),
            "prompt_tokens": sum(item["prompt_tokens"] for item in rows),
            "context_tokens": rows[0]["context_tokens"],
            "results": rows,
        }
    payload = {"version": 1, "checked": stamp,
               "memory": memory.payload(), "states": states}
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    except OSError as exc:
        raise AppError(f"Не удалось сохранить {path}: {exc}") from exc


def compare_cleanup(path: Path) -> None:
    """Сводка опыта: что теряется при исключении каждого слоя."""
    if not path.is_file():
        print(f"Нет файла результатов: {path}")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Не удалось прочитать результаты: {exc}")
        return
    states = payload.get("states") or {}
    if not states:
        print("В файле ещё нет прогонов: выполните /cleanup.")
        return
    print_table(("Состояние", "Из памяти", "Догадка", "НЕТ ДАННЫХ", "Из", "Вход", "Контекст"),
                [(state, item.get("found", 0), item.get("guessed", 0),
                  item.get("no_data", 0), item.get("details", 0),
                  item.get("prompt_tokens", 0), item.get("context_tokens", 0))
                 for state, item in states.items()])
    ids = [item["id"] for item in (next(iter(states.values())).get("results") or [])]
    rows = []
    for detail_id in ids:
        row: list[Any] = [detail_id]
        for state in CLEANUP_STATES:
            item = next((entry for entry in states.get(state, {}).get("results") or []
                         if entry["id"] == detail_id), None)
            row.append(item.get("origin", "other") if item else "—")
        rows.append(tuple(row))
    print()
    print_table(("Деталь", *CLEANUP_STATES), rows)


def print_help(history: History) -> None:
    print(f"Введите запрос и нажмите Enter. {HELP_COMMAND} — все команды "
          f"и параметры запуска, {EXIT_COMMAND} — выход.")
    print("Например:")
    for query in EXAMPLE_QUERIES:
        print(f"  {query}")
    if history.turns:
        print(f"Продолжается разговор из {len(history.turns)} обменов "
              f"(начат {history.started}).")


def _layer_by_name(value: str) -> str:
    """Имя слоя в командах: короткое (long/work/short) или по-русски."""
    low = value.strip().lower()
    for layer in LAYERS:
        if low in (layer, LAYER_TITLES[layer]):
            return layer
    raise AppError(f"Слой должен быть одним из: {', '.join(LAYERS)}.")


def _split_layer_argument(rest: str, memory: Memory) -> tuple[str, str | None]:
    """Разбирает «ключ=значение [слой]» или «ключ [слой]»."""
    parts = rest.split()
    layer = None
    if parts and parts[-1].lower() in {*LAYERS, *LAYER_TITLES.values()}:
        layer = _layer_by_name(parts[-1])
        parts = parts[:-1]
    return " ".join(parts), layer


def _remember(query: str, memory: Memory, options: Options) -> Memory:
    """/remember KEY=VALUE [long|work] — явная запись в выбранный слой."""
    rest = query[len(REMEMBER_COMMAND):].strip()
    if not rest:
        print(f"Использование: {REMEMBER_COMMAND} ключ=значение [long|work]")
        return memory
    pair, layer = _split_layer_argument(rest, memory)
    if "=" not in pair:
        print(f"Нужно «ключ=значение», например: {REMEMBER_COMMAND} имя=Марина long")
        return memory
    key, value = (part.strip() for part in pair.split("=", 1))
    if not key or not value:
        print("Ключ и значение не могут быть пустыми.")
        return memory
    # Слой по умолчанию решает правило по ключу: данные задачи — в рабочую,
    # профиль и решения — в долговременную.
    if layer is None:
        layer = WORKING if key.lower() in WORKING_KEYS else LONG_TERM
    fact = Fact(key.lower(), value, SOURCE_USER,
                0 if layer == LONG_TERM else memory.task,
                datetime.now().astimezone().strftime(TIME_FORMAT))
    memory = memory.with_layer(layer, memory.layer(layer).important(fact))
    save_memory(memory, options.work_file, options.long_file)
    print(f"Записано в «{LAYER_TITLES[layer]}»: {fact.key} = {fact.value}")
    return memory


def _forget(query: str, memory: Memory, options: Options) -> Memory:
    """/forget KEY [long|work] — удалить запись из выбранного слоя."""
    rest = query[len(FORGET_COMMAND):].strip()
    if not rest:
        print(f"Использование: {FORGET_COMMAND} ключ [long|work]")
        return memory
    pair, layer = _split_layer_argument(rest, memory)
    key = pair.strip().lower()
    if layer is None:
        layer = LONG_TERM if (key in memory.long.facts
                              and key not in memory.working.facts) else WORKING
    if key not in memory.layer(layer).facts:
        print(f"В слое «{LAYER_TITLES[layer]}» нет записи «{key}».")
        return memory
    memory = memory.with_layer(layer, memory.layer(layer).without(key))
    save_memory(memory, options.work_file, options.long_file)
    print(f"Удалено из «{LAYER_TITLES[layer]}»: {key}")
    return memory


def _topic(query: str) -> str:
    """Тема задачи из первого сообщения: для архива и /task."""
    text = _single_line(query)
    return text[:60] + ("…" if len(text) > 60 else "")


def _task_command(query: str, memory: Memory, history: History,
                  options: Options) -> tuple[Memory, History]:
    """/task NAME — новая задача, /task open N — вернуться к архивной."""
    parts = query.split(maxsplit=2)
    if len(parts) == 1:
        print(f"Текущая задача №{memory.task}"
              + (f" «{memory.title}»" if memory.title else "")
              + f": {len(memory.working.facts)} записей в рабочей памяти")
        if memory.archive:
            print_table(("№", "Тема", "Закрыта", "Записей"),
                        [(item.number, item.title or "—", item.closed,
                          len(item.facts)) for item in memory.archive])
        return memory, history
    if parts[1].lower() == "open" and len(parts) > 2 and parts[2].strip().isdigit():
        memory = memory.reopen(int(parts[2].strip()))
        save_memory(memory, options.work_file, options.long_file)
        print(f"Открыта задача №{memory.task} «{memory.title}»: "
              f"{len(memory.working.facts)} записей рабочей памяти.")
        return memory, history
    title = " ".join(parts[1:]).strip()
    memory = memory.open_task(title)
    history = History(datetime.now().astimezone().strftime(TIME_FORMAT),
                      keep=options.keep_turns, strategy=options.strategy)
    try:
        save_history(history, options.history_file)
        save_memory(memory, options.work_file, options.long_file)
    except AppError as exc:
        print(f"  предупреждение: {exc}", file=sys.stderr)
    print(f"Новая задача №{memory.task} «{title}»: рабочая память начата заново, "
          f"долговременная сохранена.")
    return memory, history


def _extract(history: History, memory: Memory, options: Options,
             client: GigaChatClient, access_token: str, runs: list[AgentRun],
             received: int) -> tuple[Memory, History]:
    """Раскладывает новый обмен по слоям: правила, затем модель.

    Курсор `history.extracted` не даёт извлекать обмен дважды. Обмен, чьи факты
    легли в рабочий или долговременный слой, убирается из истории (sync short):
    иначе он был бы виден модели и в живом окне, и из слоя, а слои не были бы
    различимы в опыте с очисткой. Ошибка извлечения не роняет диалог.
    """
    pending = history.turns[history.extracted:]
    if not pending:
        return memory, history
    facts: list[Fact] = []
    rule_facts: list[Fact] = []
    # Обмен, который пополнил внешний слой, убирается из истории: факт,
    # разложенный по слоям, не должен ещё и лежать в живом окне. Обмен без
    # фактов — наоборот, и есть краткосрочный слой: история его сохраняет.
    removable = 0
    for turn in pending:
        found = update_facts(memory, turn.query)
        rule_facts.extend(found)
        if found:
            removable += 1
    if options.extract:
        turn = pending[-1]
        try:
            extracted, run = extract_facts(turn.query, turn.answer, memory.task,
                                          client, access_token, options,
                                          (received, turn.query))
            runs.append(run)
            # Тот же ключ уже нашли правила — обмен засчитываем один раз.
            if extracted and not {fact.key for fact in rule_facts} & {
                    fact.key for fact in extracted}:
                removable += 1
            facts.extend(extracted)
        except AppError as exc:
            print(f"  извлечение фактов не удалось: {exc}", file=sys.stderr)
    facts.extend(rule_facts)
    if facts:
        memory = memory.remember(facts)
        print("  память пополнена: " + ", ".join(
            f"{fact.key} → {LAYER_TITLES[LONG_TERM if fact.task == 0 else WORKING]}"
            for fact in facts))
    if options.sync_short and removable:
        return memory, history.without_old(removable)
    return memory, history.marked(len(history.turns))


def brief_history(history: History, limit: int) -> History:
    """Оставляет краткосрочному слою `limit` последних обменов (по умолчанию 4).

    История — хранилище краткосрочного слоя, а не лог разговора: держать
    в ней всю беседу значит дублировать то, что уже легло в слои памяти
    и в сводку. Память (рабочая и долговременная) и сводка при этом целы.
    """
    if limit <= 0 or len(history.turns) <= limit:
        return history
    rest = history.without_old(len(history.turns) - limit)
    return rest.marked(min(rest.extracted, len(rest.turns)))


def print_help(history: History) -> None:
    print(f"Введите запрос и нажмите Enter. {HELP_COMMAND} — все команды "
          f"и параметры запуска, {EXIT_COMMAND} — выход.")
    print("Например:")
    for query in EXAMPLE_QUERIES:
        print(f"  {query}")
    if history.turns:
        print(f"Продолжается разговор из {len(history.turns)} обменов "
              f"(начат {history.started}).")


def _layer_by_name(value: str) -> str:
    """Имя слоя в командах: короткое (long/work/short) или по-русски."""
    low = value.strip().lower()
    for layer in LAYERS:
        if low in (layer, LAYER_TITLES[layer]):
            return layer
    raise AppError(f"Слой должен быть одним из: {', '.join(LAYERS)}.")


def _split_layer_argument(rest: str, memory: Memory) -> tuple[str, str | None]:
    """Разбирает «ключ=значение [слой]» или «ключ [слой]»."""
    parts = rest.split()
    layer = None
    if parts and parts[-1].lower() in {*LAYERS, *LAYER_TITLES.values()}:
        layer = _layer_by_name(parts[-1])
        parts = parts[:-1]
    return " ".join(parts), layer


def _remember(query: str, memory: Memory, options: Options) -> Memory:
    """/remember KEY=VALUE [long|work] — явная запись в выбранный слой."""
    rest = query[len(REMEMBER_COMMAND):].strip()
    if not rest:
        print(f"Использование: {REMEMBER_COMMAND} ключ=значение [long|work]")
        return memory
    pair, layer = _split_layer_argument(rest, memory)
    if "=" not in pair:
        print(f"Нужно «ключ=значение», например: {REMEMBER_COMMAND} имя=Марина long")
        return memory
    key, value = (part.strip() for part in pair.split("=", 1))
    if not key or not value:
        print("Ключ и значение не могут быть пустыми.")
        return memory
    # Слой по умолчанию решает правило по ключу: данные задачи — в рабочую,
    # профиль и решения — в долговременную.
    if layer is None:
        layer = WORKING if key.lower() in WORKING_KEYS else LONG_TERM
    fact = Fact(key.lower(), value, SOURCE_USER,
                0 if layer == LONG_TERM else memory.task,
                datetime.now().astimezone().strftime(TIME_FORMAT))
    memory = memory.with_layer(layer, memory.layer(layer).important(fact))
    save_memory(memory, options.work_file, options.long_file)
    print(f"Записано в «{LAYER_TITLES[layer]}»: {fact.key} = {fact.value}")
    return memory


def _forget(query: str, memory: Memory, options: Options) -> Memory:
    """/forget KEY [long|work] — удалить запись из выбранного слоя."""
    rest = query[len(FORGET_COMMAND):].strip()
    if not rest:
        print(f"Использование: {FORGET_COMMAND} ключ [long|work]")
        return memory
    pair, layer = _split_layer_argument(rest, memory)
    key = pair.strip().lower()
    if layer is None:
        layer = LONG_TERM if (key in memory.long.facts
                              and key not in memory.working.facts) else WORKING
    if key not in memory.layer(layer).facts:
        print(f"В слое «{LAYER_TITLES[layer]}» нет записи «{key}».")
        return memory
    memory = memory.with_layer(layer, memory.layer(layer).without(key))
    save_memory(memory, options.work_file, options.long_file)
    print(f"Удалено из «{LAYER_TITLES[layer]}»: {key}")
    return memory


def _topic(query: str) -> str:
    """Тема задачи из первого сообщения: для архива и /task."""
    text = _single_line(query)
    return text[:60] + ("…" if len(text) > 60 else "")


def _task_command(query: str, memory: Memory, history: History,
                  options: Options) -> tuple[Memory, History]:
    """/task NAME — новая задача, /task open N — вернуться к архивной."""
    parts = query.split(maxsplit=2)
    if len(parts) == 1:
        print(f"Текущая задача №{memory.task}"
              + (f" «{memory.title}»" if memory.title else "")
              + f": {len(memory.working.facts)} записей в рабочей памяти")
        if memory.archive:
            print_table(("№", "Тема", "Закрыта", "Записей"),
                        [(item.number, item.title or "—", item.closed,
                          len(item.facts)) for item in memory.archive])
        return memory, history
    if parts[1].lower() == "open" and len(parts) > 2 and parts[2].strip().isdigit():
        memory = memory.reopen(int(parts[2].strip()))
        save_memory(memory, options.work_file, options.long_file)
        print(f"Открыта задача №{memory.task} «{memory.title}»: "
              f"{len(memory.working.facts)} записей рабочей памяти.")
        return memory, history
    title = " ".join(parts[1:]).strip()
    memory = memory.open_task(title)
    history = History(datetime.now().astimezone().strftime(TIME_FORMAT),
                      keep=options.keep_turns, strategy=options.strategy)
    try:
        save_history(history, options.history_file)
        save_memory(memory, options.work_file, options.long_file)
    except AppError as exc:
        print(f"  предупреждение: {exc}", file=sys.stderr)
    print(f"Новая задача №{memory.task} «{title}»: рабочая память начата заново, "
          f"долговременная сохранена.")
    return memory, history


def _extract(history: History, memory: Memory, options: Options,
             client: GigaChatClient, access_token: str, runs: list[AgentRun],
             received: int) -> tuple[Memory, History]:
    """Раскладывает новый обмен по слоям: правила, затем модель.

    Курсор `history.extracted` не даёт извлекать обмен дважды. Обмен, чьи факты
    легли в рабочий или долговременный слой, убирается из истории (sync short):
    иначе он был бы виден модели и в живом окне, и из слоя, а слои не были бы
    различимы в опыте с очисткой. Ошибка извлечения не роняет диалог.
    """
    pending = history.turns[history.extracted:]
    if not pending:
        return memory, history
    facts: list[Fact] = []
    rule_facts: list[Fact] = []
    # Обмен, который пополнил внешний слой, убирается из истории: факт,
    # разложенный по слоям, не должен ещё и лежать в живом окне. Обмен без
    # фактов — наоборот, и есть краткосрочный слой: история его сохраняет.
    removable = 0
    for turn in pending:
        found = update_facts(memory, turn.query)
        rule_facts.extend(found)
        if found:
            removable += 1
    if options.extract:
        turn = pending[-1]
        try:
            extracted, run = extract_facts(turn.query, turn.answer, memory.task,
                                          client, access_token, options,
                                          (received, turn.query))
            runs.append(run)
            # Тот же ключ уже нашли правила — обмен засчитываем один раз.
            if extracted and not {fact.key for fact in rule_facts} & {
                    fact.key for fact in extracted}:
                removable += 1
            facts.extend(extracted)
        except AppError as exc:
            print(f"  извлечение фактов не удалось: {exc}", file=sys.stderr)
    facts.extend(rule_facts)
    if facts:
        memory = memory.remember(facts)
        print("  память пополнена: " + ", ".join(
            f"{fact.key} → {LAYER_TITLES[LONG_TERM if fact.task == 0 else WORKING]}"
            for fact in facts))
    if options.sync_short and removable:
        return memory, history.without_old(removable)
    return memory, history.marked(len(history.turns))


def print_help(history: History, memory: Memory | None = None) -> None:
    memory = memory or Memory()
    print(f"Введите запрос и нажмите Enter. {HELP_COMMAND} — все команды "
          f"и параметры запуска, {MEMORY_COMMAND} — слои памяти, "
          f"{EXIT_COMMAND} — выход.")
    print("Например:")
    for query in EXAMPLE_QUERIES:
        print(f"  {query}")
    if history.turns:
        print(f"Продолжается разговор из {len(history.turns)} обменов "
              f"(начат {history.started}).")
    print(f"Память: задача №{memory.task}"
          + (f" «{memory.title}»" if memory.title else "")
          + f", рабочая {len(memory.working.facts)} записей, "
            f"долговременная {len(memory.long.facts)}.")


def print_commands() -> None:
    """Полная справка: команды диалога и параметры запуска."""
    print("Команды диалога:")
    print("  /state           этап, шаг и ожидаемое действие; /state help — команды")
    print("  /profile         просмотр и изменение предпочтений, переключение пользователя")
    print("  /profile list    список профилей; /profile select — выбор по номеру")
    print("  /personal-test   опыт с профилями и устойчивостью (ручная оценка)")
    for command, about in COMMAND_HELP:
        print(f"  {command:<16} {about}")
    print()
    print("Параметры запуска (для --flag=value и --flag value):")
    for flag, about in OPTION_HELP:
        if flag.split()[0] not in ("--history", "--summary", "--work", "--long", "--branches"):
            print(f"  {flag:<20} {about}")
    print()
    print("Файлы по умолчанию:")
    print(f"  память и опыт       {PROJECT_DIR / 'profiles'} / <имя профиля>")
    print(f"  контрольные детали  {DETAILS_FILE}")
    print(f"  журнал инструментов {DEFAULT_LOG_FILE}")
    print(f"  HTTP-лог диалога    {DEFAULT_HTTP_LOG_FILE}")
    print(f"  модель по умолчанию {DEFAULT_MODEL}, лимит {DEFAULT_MODEL_LIMIT} токенов")
    print("Ключ GIGACHAT_CREDENTIALS берётся из .env в поиске вверх от папки задачи.")


def save_memory(memory: Memory, work_file: Path, long_file: Path) -> None:
    """Пишет обе памяти отдельными файлами: слои не смешиваются на диске.

    Рабочая память пишется целиком вместе с архивом задач: так задача
    восстанавливается после возврата к ней. Долговременная — только своим слоем.
    """
    stamp = datetime.now().astimezone().strftime(TIME_FORMAT)
    working = {
        "version": MEMORY_VERSION,
        "saved": stamp,
        "task": memory.task,
        "title": memory.title,
        "state": dataclasses.asdict(memory.state),
        "archive": [record.payload() for record in memory.archive],
        "layer": memory.working.payload(),
    }
    long_term = {"version": MEMORY_VERSION, "saved": stamp,
                 "layer": memory.long.payload()}
    for path, payload in ((long_file, long_term), (work_file, working)):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            raise AppError(f"Не удалось сохранить память {path}: {exc}") from exc


def _layer_payload(payload: Any) -> Layer:
    """Слой из файла: битая запись пропускается, а не роняет запуск."""
    if not isinstance(payload, dict):
        return Layer()
    facts: dict[str, Fact] = {}
    for item in payload.get("facts") or []:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        fact = Fact(
            key=str(item.get("key")),
            value=str(item.get("value", "")),
            source=str(item.get("source") or SOURCE_RULE),
            task=int(item.get("task") or 0),
            updated=str(item.get("updated") or ""),
        )
        facts[fact.key] = fact
    return Layer(facts)


def load_memory(work_file: Path, long_file: Path) -> Memory:
    """Читает оба файла памяти; файла нет — соответствующий слой пуст."""
    memory = Memory()
    if work_file.is_file():
        payload = _read_memory(work_file)
        archive = []
        for item in payload.get("archive") or []:
            if not isinstance(item, dict):
                continue
            archive.append(TaskRecord(
                number=int(item.get("number") or 0),
                title=str(item.get("title") or ""),
                opened=str(item.get("opened") or ""),
                closed=str(item.get("closed") or ""),
                facts=_layer_payload({"facts": item.get("facts")}).facts,
                state=parse_task_state(item.get("state")),
            ))
        memory = Memory(
            working=_layer_payload(payload.get("layer")),
            long=memory.long,
            task=int(payload.get("task") or 1),
            title=str(payload.get("title") or ""),
            archive=archive,
            state=parse_task_state(payload.get("state")),
        )
    if long_file.is_file():
        memory = dataclasses.replace(
            memory, long=_layer_payload(_read_memory(long_file).get("layer")))
    return memory


def _read_memory(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError(f"Не удалось прочитать память {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AppError(f"Файл памяти {path} не похож на память.")
    return payload


def save_checkpoint(history: History, name: str, branches_file: Path) -> None:
    """Сохраняет снимок истории в отдельный файл ветки."""
    safe = re.sub(r"[^\w.-]+", "_", name).strip("._")
    if not safe:
        raise AppError("Имя checkpoint не может быть пустым.")
    path = branches_file.parent / f"{branches_file.stem}_{safe}.json"
    save_history(history, path)
    print(f"Checkpoint «{name}» сохранён: {path.name}")


def load_checkpoint(name: str, branches_file: Path, strategy: str,
                    keep: int | None) -> History:
    safe = re.sub(r"[^\w.-]+", "_", name).strip("._")
    path = branches_file.parent / f"{branches_file.stem}_{safe}.json"
    if not path.is_file():
        raise AppError(f"Не найден checkpoint «{name}»: {path}")
    return load_history(path, None, keep, strategy)


def save_history(history: History, history_file: Path) -> None:
    """Пишет историю разговора целиком: файл — состояние, а не журнал."""
    payload = {
        "version": HISTORY_VERSION,
        "started": history.started,
        "saved": datetime.now().astimezone().strftime(TIME_FORMAT),
        "strategy": history.strategy,
        "extracted": history.extracted,
        "summary": history.summary.payload(),
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
                "tokens": turn.tokens(),
                "usage": [_usage_payload(usage) for usage in
                          (turn.usage.usages if turn.usage else [])],
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


def load_history(history_file: Path, summary_file: Path | None = None,
                 keep: int | None = None, strategy: str = DEFAULT_STRATEGY) -> History:
    """Читает сохранённый разговор и его сводку; битый файл — ошибка, а не потеря."""
    if not history_file.is_file():
        return History(datetime.now().astimezone().strftime(TIME_FORMAT),
                       summary=load_summary(summary_file) if summary_file else Summary(),
                       keep=keep, strategy=strategy)
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
            usage=AgentRun(
                query=item["query"], steps=steps, answer=str(item.get("answer", "")),
                seconds=0.0,
                prompt_tokens=int((item.get("tokens") or {}).get("prompt") or 0),
                completion_tokens=int((item.get("tokens") or {}).get("completion") or 0),
                usages=[StepTokenUsage(
                    step=int(usage.get("step") or 0),
                    prompt=int(usage.get("prompt") or 0),
                    completion=int(usage.get("completion") or 0),
                    cached=int(usage.get("cached") or 0),
                    estimated=int(usage.get("estimated") or 0),
                    function=str(usage.get("function") or ""),
                ) for usage in item.get("usage") or [] if isinstance(usage, dict)],
            ) if item.get("tokens") else None,
        ))
    saved_summary = Summary(**{key: payload.get("summary", {}).get(key, default)
                               for key, default in (("text", ""), ("compressed_turns", 0),
                                                    ("created", ""), ("updated", ""),
                                                    ("sources", 0))})
    return History(str(payload.get("started", "")), turns,
                   load_summary(summary_file) if summary_file else saved_summary, keep,
                   strategy or str(payload.get("strategy") or DEFAULT_STRATEGY),
                   int(payload.get("extracted") or 0))


def clear_history(history_file: Path) -> None:
    """Забыть разговор: файла нет — уже забыт."""
    try:
        history_file.unlink(missing_ok=True)
    except OSError as exc:
        raise AppError(f"Не удалось удалить историю {history_file}: {exc}") from exc


def _usage_payload(usage: StepTokenUsage) -> dict[str, Any]:
    """Расход одного обращения к модели для файла истории."""
    return {
        "step": usage.step,
        "prompt": usage.prompt,
        "completion": usage.completion,
        "cached": usage.cached,
        "estimated": usage.estimated,
        "function": usage.function,
    }


def replace(options: Options, **changes: int) -> Options:
    """Копия настроек с другим значением поля (dataclass неизменяемый)."""
    return dataclasses.replace(options, **changes)


def save_summary(summary: Summary, summary_file: Path) -> None:
    """Пишет сводку отдельным файлом: история не переписывается."""
    try:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(
            json.dumps(summary.payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AppError(f"Не удалось сохранить сводку {summary_file}: {exc}") from exc


def load_summary(summary_file: Path) -> Summary:
    """Читает сводку; файла нет — сводки нет, разговор без сжатия."""
    if not summary_file.is_file():
        return Summary()
    try:
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError(f"Не удалось прочитать сводку {summary_file}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AppError(f"Файл сводки {summary_file} не похож на сводку.")
    return Summary(
        text=str(payload.get("text") or ""),
        compressed_turns=int(payload.get("compressed_turns") or 0),
        created=str(payload.get("created") or ""),
        updated=str(payload.get("updated") or ""),
        sources=int(payload.get("sources") or 0),
    )


def clear_summary(summary_file: Path) -> None:
    try:
        summary_file.unlink(missing_ok=True)
    except OSError as exc:
        raise AppError(f"Не удалось удалить сводку {summary_file}: {exc}") from exc


def layer_tokens(memory: Memory, history: History) -> dict[str, int]:
    """Сколько токенов добавляет каждый слой к запросу.

    Считается по тем же сообщениям, которые слой отдаёт в промпт, с накладными
    токенами сообщения — так числа сопоставимы. Системная подсказка и описания
    инструментов входят в `total`: это общая цена запроса.
    """
    long_block = memory.long.block("Долговременная память (профиль, решения, знания):",
                                   LONG_TERM_ITEMS)
    work_block = memory.working.block("Рабочая память текущей задачи:", WORKING_ITEMS)
    short_messages = history.messages()
    summary = (estimate_messages_tokens(summary_messages(history.summary))
               if history.summary.text else 0)
    short = estimate_messages_tokens(short_messages) - summary
    system = estimate_messages_tokens([{"role": "system", "content": SYSTEM_PROMPT}])
    tools = estimate_tokens(
        json.dumps([tool.schema() for tool in TOOLS], ensure_ascii=False)
    )
    tokens = {
        LONG_TERM: estimate_messages_tokens(long_block),
        WORKING: estimate_messages_tokens(work_block),
        SHORT_TERM: max(0, short),
        "summary": summary,
        "system": system,
        "tools": tools,
    }
    tokens["total"] = (tokens[LONG_TERM] + tokens[WORKING] + tokens[SHORT_TERM]
                       + summary + system + tokens["tools"])
    return tokens


def history_tokens(history: History, memory: Memory | None = None) -> int:
    """Оценка того, сколько токенов займёт запрос с этой памятью и историей."""
    return layer_tokens(memory or Memory(), history)["total"]


def full_history_tokens(history: History, memory: Memory | None = None) -> int:
    """То же без сжатия: столько было бы, если бы ушла вся история целиком."""
    everything = dataclasses.replace(history, summary=Summary(), keep=None)
    return history_tokens(everything, memory)


def layer_lines(memory: Memory, history: History) -> list[str]:
    """Таблица вклада слоёв — её показывают /memory и /limits."""
    tokens = layer_tokens(memory, history)
    rows = []
    for layer in LAYERS:
        content = memory.layer(layer) if layer != SHORT_TERM else None
        rows.append((LAYER_TITLES[layer], _layer_size(memory, history, layer),
                     tokens[layer], "в промпте" if tokens[layer] else "пусто"))
    return [f"  {title:<16}{size:>6} в промпте  ~{tokens:>6} токенов  {state}"
            for title, size, tokens, state in rows]


def _layer_size(memory: Memory, history: History, layer: str) -> str:
    if layer == SHORT_TERM:
        return f"{len(history.context_turns())}/{len(history.turns)}"
    content = memory.layer(layer)
    limit = LONG_TERM_ITEMS if layer == LONG_TERM else WORKING_ITEMS
    return f"{min(limit, len(content.facts))}/{len(content.facts)}"


def compression_lines(history: History, options: Options,
                      memory: Memory | None = None) -> list[str]:
    """Сколько сейчас стоит контекст и сколько сэкономило сжатие.

    Первая строка — без отступа (её печатают после слова «Сжатие:»),
    остальные — с отступом в готовом виде.
    """
    now = history_tokens(history, memory)
    full = full_history_tokens(history, memory)
    saved = max(0, full - now)
    summary_tokens = (estimate_messages_tokens(summary_messages(history.summary))
                      if history.summary.text else 0)
    live = len(history.context_turns())
    saved_share = f"{100.0 * saved / full:.0f}%" if full else "0%"
    lines = [
        f" ~{now} токенов, без сжатия было бы ~{full} токенов",
        f"  сжато: {history.summary.compressed_turns} обменов -> сводка "
        f"~{summary_tokens} токенов, экономия ~{saved} токенов ({saved_share})",
        f"  как есть: {live} обменов из {len(history.turns)}"
        + (f", до автосжатия: {max(0, options.compress_every - len(history.pending_turns()))} обменов"
           if options.compression else ", сжатие выключено"),
    ]
    if history.summary.text:
        lines.append("  сводка: " + history.summary.text.replace("\n", " / "))
    return lines


def print_gigachat_balance(client: GigaChatClient, access_token: str,
                          model: str) -> None:
    """Показывает остаток токенов выбранной модели при запуске."""
    try:
        balance = client.get_balance(access_token)
    except GigaChatError as exc:
        print(f"Остаток токенов: недоступен ({exc})")
        return
    matches = [item for item in balance
               if str(item.get("usage") or "") == model]
    if not matches:
        print(f"Остаток токенов для {model}: не найден в ответе API")
        return
    value = matches[0].get("value")
    try:
        rendered = f"{int(value or 0):,}".replace(",", " ")
    except (TypeError, ValueError):
        rendered = str(value)
    print(f"Остаток токенов для {model}: {rendered}")


def print_memory(memory: Memory, history: History, options: Options) -> None:
    """/memory: что лежит в каждом слое, чем он отдаётся в промпт и кто решал."""
    print("\nСлои памяти")
    tokens = layer_tokens(memory, history)
    print(f"  рабочая память: задача №{memory.task}"
          + (f" «{memory.title}»" if memory.title else ""))
    for layer in LAYERS:
        content = memory.layer(layer) if layer != SHORT_TERM else None
        print(f"\n  {LAYER_TITLES[layer]} ({layer})")
        if layer == SHORT_TERM:
            print(f"    файл: {options.history_file.name}"
                  + (f" + {options.summary_file.name}" if history.summary.text else ""))
            print(f"    правило: последние {options.keep_turns or 'все'} обменов "
                  f"текущей задачи" + (", старшие — сводкой"
                                       if history.summary.text else ""))
            print(f"    в промпте: {len(history.context_turns())} обменов "
                  f"из {len(history.turns)}, ~{tokens[SHORT_TERM]} токенов"
                  + (f" + сводка ~{tokens['summary']}" if history.summary.text else ""))
        else:
            limit = LONG_TERM_ITEMS if layer == LONG_TERM else WORKING_ITEMS
            path = options.long_file if layer == LONG_TERM else options.work_file
            print(f"    файл: {path.name}")
            print(f"    правило: последние {limit} записей, новые выше")
            items = content.selected(limit)
            print(f"    в промпте: {len(items)} из {len(content.facts)} записей, "
                  f"~{tokens[layer]} токенов")
            for fact in items[:8]:
                print(f"      - {fact.key}: {_single_line(fact.value)[:70]} "
                      f"({fact.source})")
    if memory.archive:
        print(f"\n  архив задач: "
              + ", ".join(f"№{item.number}{' «' + item.title + '»' if item.title else ''}"
                          f" ({len(item.facts)} записей)" for item in memory.archive))
    print(f"\n  итого запрос: ~{tokens['total']} токенов "
          f"(из них система ~{tokens['system']} и инструменты ~{tokens['tools']})")


def print_session_limits(runs: list[AgentRun], history: History,
                        options: Options, memory: Memory | None = None) -> None:
    """/limits: вклад слоёв, сжатие, история, сеанс и каждый запрос сеанса."""
    memory = memory or Memory()
    print("\nТокены")
    print(f"  модель: {options.model}, лимит контекста {options.model_limit}")
    print_share_line(history_tokens(history, memory), options.model_limit)
    print("Слои памяти:")
    for line in layer_lines(memory, history):
        print(line)
    print("Сжатие:" + compression_lines(history, options, memory)[0])
    for line in compression_lines(history, options, memory)[1:]:
        print(line)
    if runs:
        prompt = sum(run.prompt_tokens for run in runs)
        completion = sum(run.completion_tokens for run in runs)
        answer = sum(run.answer_tokens for run in runs)
        print(f"  сеанс: {prompt} вход + {completion} выход = {prompt + completion} "
              f"за {len(runs)} запросов (это трата и деньги)")
        print(f"  из выхода {answer} — текст ответов, "
              f"{completion - answer} — просьбы вызвать инструменты")
    else:
        print("  сеанс: запросов ещё не было")
    for history_left, turn in enumerate(history.turns, start=1):
        tokens = turn.tokens()
        usage = turn.usage
        estimate = ""
        if usage is not None and usage.first_usage is not None:
            first = usage.first_usage
            estimate = (f", оценка входа {first.estimated} "
                        f"({_percent(first.prompt, first.estimated)})")
        # Помечаем, что из контекста уже заменено сводкой, а что ещё лежит как есть.
        state = (" (в сводке)" if history_left <= history.summary.compressed_turns
                 else "")
        print(
            f"  история #{history_left}{state}: вход {tokens['prompt']} + выход "
            f"{tokens['completion']} = {tokens['total']}{estimate}"
        )
        kept = turn.context_steps()
        if kept:
            # Сколько из истории — это уже прочитанные страницы: они уходят
            # модели в каждом следующем запросе, в отличие от шага цикла.
            tokens_kept = sum(len(str(step.result)) for step in kept)
            print(f"      сохранено в контексте: {len(kept)} вызов(ов) "
                  f"{', '.join(step.tool for step in kept)}, "
                  f"{tokens_kept} символов, ~{turn.context_tokens()} токенов")
        if usage is not None and usage.usages:
            for line in _step_lines(usage.usages, indent="      ",
                                    limit=options.model_limit):
                print(line)
    print(f"  всего в истории: {len(history.turns)} обменов")
    kept = sum(turn.context_tokens() for turn in history.context_turns())
    if kept:
        print(f"  из них сохранённые результаты инструментов: ~{kept} токенов, "
              f"платятся в каждом запросе до {CLEAR_COMMAND}")


def _parse_fill_words(query: str, options: Options) -> int:
    """Сколько слов-заполнителей наполнить.\n
    Без аргумента — до лимита; с числом — ровно столько (для отладки);
    с \"limit\" — до лимита, даже если он больше FILL_LIMIT_WORDS.
    """
    parts = query.split(maxsplit=1)
    if len(parts) == 1:
        return _words_for_limit(options)
    value = parts[1].strip()
    if value.isdigit():
        words = int(value)
        if words < FILL_MIN_WORDS:
            raise AppError(f"Слишком мало заполнителя: минимум {FILL_MIN_WORDS} слов.")
        return words
    raise AppError("После /fill можно указать число слов, например /fill 5000.")


def _words_for_limit(options: Options) -> int:
    """Сколько слов добавить, чтобы оценка истории подобралась к лимиту.

    Доля токенов на слово считается той же формулой, что и остальная оценка,
    по настоящей строке-заполнителю: отдельного коэффициента для заполнителя
    нет — прежний (два токена на слово) был догадкой, а не измерением.
    """
    probe = 1000
    per_word = estimate_message_tokens(
        {"role": "user", "content": _fill_query(probe)}
    ) / probe
    return max(FILL_MIN_WORDS, round(options.model_limit / per_word))


def _fill_query(words: int) -> str:
    return (f"Для проверки истории: {' '.join([FILL_FILLER] * words)}. "
            "Ответь одним словом.")


def fill_history(history: History, options: Options, client: GigaChatClient,
                 access_token: str, runs: list[AgentRun],
                 received: int = 0, memory: Memory | None = None,
                 profile_blocks: list[dict[str, Any]] | None = None) -> History:
    """Наполняет историю заполнителем — для сценария «за лимитом модели».

    Заполнитель вставляется как прошлый обмен (запрос-заполнитель и короткий
    ответ агента), поэтому дальше это обычный длинный разговор. Обращение
    к модели одно и сделано только чтобы получить факт по `usage` и сравнить
    с оценкой — так видно, насколько точна формула.
    """
    words = options.fill_words
    query = _fill_query(words)
    answer = "наполнено"
    print(f"\nНаполняю историю: {words} слов-заполнителей")
    print(f"  оценка запроса: ~{history_tokens(history, memory)} токенов "
          f"без заполнителя + ~{estimate_tokens(query)}")
    run = run_agent(query, client, access_token, options.model,
                    options.log_file, history,
                    context=(received or len(runs) + 1, query),
                    layers=[*(profile_blocks or []), *(memory or Memory()).blocks()])
    runs.append(run)
    # Сравнивать надо факт и оценку одного и того же шага: у заполнителя цикл
    # тоже может пройти в несколько шагов, и тогда prompt_tokens — сумма.
    print(f"  модель ответила за {len(run.usages)} обр.")
    for line in _estimate_lines(run):
        print(f"  {line}")
    turn = ChatTurn(query=query, answer=run.answer or answer,
                    steps=run.steps, at=datetime.now().astimezone().strftime(TIME_FORMAT),
                    usage=run)
    history = history.extended(turn)
    try:
        save_history(history, options.history_file)
    except AppError as exc:
        print(f"  предупреждение: {exc}", file=sys.stderr)
    print_share_line(history_tokens(history, memory), options.model_limit)
    return history


def export(runs: list[AgentRun], options: Options, base: Path) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_path = OUTPUT_DIR / f"{base}_{stamp}.json"
    csv_path = OUTPUT_DIR / f"{base}_{stamp}.csv"
    payload = {
        "meta": {
            "provider": options.provider,
            "model": options.model,
            "strategy": options.strategy,
            "window": options.window,
            "max_steps": MAX_STEPS,
            "tools": [tool.schema() for tool in TOOLS],
            "log_file": str(options.log_file),
        "layers": {"short": str(options.history_file), "work": str(options.work_file),
                   "long": str(options.long_file)},
        "extract": options.extract,
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
                "total_tokens": run.total_tokens,
                "answer_tokens": run.answer_tokens,
                "function_tokens": run.function_tokens,
                "estimated_context": run.estimated_context,
                "estimated_prompt": run.estimated_prompt,
                "tokens_by_step": [_usage_payload(usage) for usage in run.usages],
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
                 "Вход", "Выход", "Всего", "В ответе", "На функции",
                 "Контекст до запроса", "Оценка входа", "Ошибка")
            )
            for run in runs:
                errors = "; ".join(step.error for step in run.steps if step.error)
                writer.writerow(
                    (run.query, ", ".join(run.tools), len(run.steps), run.answer,
                     f"{run.seconds:.1f}", run.prompt_tokens,
                     run.completion_tokens, run.total_tokens, run.answer_tokens,
                     run.function_tokens, run.estimated_context,
                     run.estimated_prompt, errors)
                )
    except OSError as exc:
        raise AppError(f"Не удалось сохранить результаты: {exc}") from exc
    return json_path, csv_path


PROFILE_DIR = PROJECT_DIR / "profiles"
PROFILE_FIELDS = ("style", "format", "constraints")
PROFILE_LABELS = {"style": "стиль", "format": "формат", "constraints": "ограничения"}


@dataclass(frozen=True)
class UserProfile:
    name: str
    style: str = ""
    format: str = ""
    constraints: str = ""

    def block(self) -> list[dict[str, Any]]:
        values = [f"{PROFILE_LABELS[key]}: {getattr(self, key)}"
                  for key in PROFILE_FIELDS if getattr(self, key)]
        if not values:
            return []
        return [{"role": "user", "content": "Предпочтения пользователя "
                 "(соблюдай, если они не противоречат более важным инструкциям):\n"
                 + "\n".join(values)},
                {"role": "assistant", "content": "Предпочтения приняты и учитываются."}]


def profile_options(options: Options, name: str) -> Options:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", name):
        raise AppError("Имя профиля: до 40 латинских букв, цифр, _ или -.")
    directory = PROFILE_DIR / name
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AppError(f"Не удалось создать профиль: {exc}") from exc
    return dataclasses.replace(options, history_file=directory / "history.json",
                               summary_file=directory / "summary.json",
                               work_file=directory / "work.json",
                               long_file=directory / "long.json",
                               branches_file=directory / "branches.json")


def load_profile(name: str) -> UserProfile:
    path = PROFILE_DIR / name / "profile.json"
    if not path.exists():
        return UserProfile(name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or any(not isinstance(data.get(key, ""), str)
                                             for key in PROFILE_FIELDS):
            raise ValueError("неверный формат полей")
        return UserProfile(name, **{key: data.get(key, "") for key in PROFILE_FIELDS})
    except (OSError, ValueError) as exc:
        raise AppError(f"Не удалось прочитать профиль {name}: {exc}") from exc


def save_profile(profile: UserProfile) -> None:
    path = PROFILE_DIR / profile.name / "profile.json"
    try:
        path.write_text(json.dumps(dataclasses.asdict(profile), ensure_ascii=False,
                                   indent=2), encoding="utf-8")
    except OSError as exc:
        raise AppError(f"Не удалось сохранить профиль: {exc}") from exc


def preference_rules(query: str, profile: UserProfile) -> dict[str, str]:
    if not re.search(r"\b(?:всегда|обычно|предпочитаю|впредь|в дальнейшем|никогда)\b", query, re.IGNORECASE):
        return {}
    result: dict[str, str] = {}
    style = profile.style
    for pattern, group in ((r"\b(кратко|коротко|лаконично|подробно)\b",
                            r"\b(?:кратко|коротко|лаконично|подробно)\b"),
                           (r"\b(неформально|формально|официально)\b",
                            r"\b(?:неформально|формально|официально)\b")):
        found = list(re.finditer(pattern, query, re.IGNORECASE))
        matches = [match for match in found
                   if not re.search(r"\bне\b[^.!?,;]*$", query[:match.start()], re.IGNORECASE)]
        for match in found:
            if match not in matches and re.search(rf"\b{re.escape(match.group(1))}\b", style, re.IGNORECASE):
                style = re.sub(rf"\b{re.escape(match.group(1))}\b", "", style, flags=re.IGNORECASE).strip(" и,")
                result["style"] = style
        if matches:
            value = matches[-1].group(1).lower()
            if re.search(group, style, re.IGNORECASE):
                style = re.sub(group, value, style, flags=re.IGNORECASE)
            else:
                style = f"{style} и {value}" if style else value
            result["style"] = style
    formats = [(match.start(), value) for pattern, value in (
        (r"(?:без\s+(?:примеров\s+)?кода|не\s+(?:приводи|пиши|показывай|используй)\s+(?:примеров\s+)?код(?:а)?)", "без кода"),
        (r"(?:с|приводи|показывай)\s+(?:примерами?\s+)?код(?:а|ом)?", "с примерами кода"))
        for match in re.finditer(pattern, query, re.IGNORECASE)]
    if formats:
        result["format"] = max(formats)[1]
    constraint = re.search(r"никогда\s+не\s+([^.!?\n]+)", query, re.IGNORECASE)
    if constraint and "код" not in constraint.group(0).lower():
        result["constraints"] = constraint.group(0).strip()
    return result


def extract_preferences(query: str, profile: UserProfile, options: Options,
                        client: Any, access_token: str, received: int,
                        runs: list[AgentRun]) -> UserProfile:
    updates: dict[str, str] = {}
    if re.search(r"\b(?:только сейчас|на этот раз|в этом ответе|для этого ответа)\b",
                 query, re.IGNORECASE):
        return profile
    if options.extract:
        prompt = ("Извлеки только явно выраженные пользователем постоянные предпочтения "
                  "ответов, а не просьбы только для текущего вопроса. Верни JSON "
                  '{"facts":[{"key":"style|format|constraints","value":"текст"}]}. '
                  "Если предпочтений нет, верни пустой список. Не выдумывай.\n"
                  "Поле style сохраняет и длину, и тон: при изменении одного "
                  "сохрани второе. Не записывай одноразовые просьбы.\n"
                  f"Текущий стиль: {profile.style}\nСообщение пользователя: {query}")
        try:
            run = run_agent("", client, access_token, options.model, options.log_file,
                            History(""), context=(received, "извлечение предпочтений"),
                            include_tools=False, prompt=prompt)
            runs.append(run)
            for fact in _json_facts(run.answer):
                key = str(fact.get("key", "")).strip().lower()
                value = str(fact.get("value", "")).strip()
                if key in PROFILE_FIELDS and value:
                    updates[key] = value
        except AppError as exc:
            print(f"  извлечение предпочтений не удалось: {exc}", file=sys.stderr)
    rules = preference_rules(query, profile)
    updates.update(rules)
    if not updates:
        return profile
    result = dataclasses.replace(profile, **updates)
    save_profile(result)
    print("  профиль обновлён: " + ", ".join(PROFILE_LABELS[key] for key in updates))
    return result


def profile_names() -> list[str]:
    try:
        return sorted(path.name for path in PROFILE_DIR.iterdir()
                      if path.is_dir() and re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", path.name))
    except OSError as exc:
        raise AppError(f"Не удалось прочитать список профилей: {exc}") from exc


def print_profiles(names: list[str], active: str) -> None:
    print("Профили:")
    for number, name in enumerate(names, 1):
        print(f"  {number}. {name}" + (" (активный)" if name == active else ""))


def profile_command(query: str, profile: UserProfile) -> UserProfile:
    rest = query[len("/profile"):].strip()
    if not rest or rest == "show":
        print(f"Профиль: {profile.name}")
        for key in PROFILE_FIELDS:
            print(f"  {key} ({PROFILE_LABELS[key]}): {getattr(profile, key) or 'не задано'}")
        print("/profile set style|format|constraints=значение; пустое значение удаляет поле")
        print("/profile use ИМЯ — переключить профиль; /profile select — выбрать из списка")
        return profile
    if rest == "list":
        print_profiles(profile_names(), profile.name)
        return profile
    if rest.startswith("set ") and "=" in rest:
        key, value = rest[4:].split("=", 1)
        key = key.strip().lower()
        if key not in PROFILE_FIELDS:
            raise AppError("Поля профиля: style, format, constraints.")
        result = dataclasses.replace(profile, **{key: value.strip()})
        save_profile(result)
        return result
    raise AppError("Использование: /profile [show|list|select|use ИМЯ|set поле=значение].")


def task_command(query: str, memory: Memory, history: History,
                 options: Options) -> tuple[Memory, History]:
    parts = query.split(maxsplit=2)
    if len(parts) == 1:
        print(f"Текущая задача №{memory.task} «{memory.title}»: этап {memory.state.stage}")
        if memory.archive:
            print_table(("№", "Тема", "Этап", "Пауза"),
                        [(item.number, item.title, item.state.stage,
                          "да" if item.state.paused else "нет") for item in memory.archive])
        return memory, history
    if parts[1].lower() == "open":
        if len(parts) != 3 or not parts[2].strip().isdigit():
            raise AppError("Использование: /task open НОМЕР.")
        updated = memory.reopen(int(parts[2]))
    else:
        updated = memory.open_task(" ".join(parts[1:]).strip())
    new_history = History(datetime.now().astimezone().strftime(TIME_FORMAT),
                          keep=options.keep_turns, strategy=options.strategy)
    save_memory(updated, options.work_file, options.long_file)
    save_history(new_history, options.history_file)
    clear_summary(options.summary_file)
    print(f"Открыта задача №{updated.task} «{updated.title}», этап {updated.state.stage}.")
    return updated, new_history


def state_command(query: str, memory: Memory) -> Memory:
    parts = query.split(maxsplit=2)
    action = parts[1].lower() if len(parts) > 1 else "show"
    value = parts[2].strip() if len(parts) > 2 else ""
    state = memory.state
    if action == "show":
        print(f"Задача №{memory.task}: этап {state.stage}, "
              f"{'пауза' if state.paused else 'активна'}")
        print(f"  шаг: {state.step}\n  ожидается: {state.expected}\n  резюме: {state.summary or 'не задано'}")
        return memory
    if action == "help":
        print("/state [show|pause|resume|step ТЕКСТ|expected ТЕКСТ|summary ТЕКСТ|go ЭТАП [confirm]]")
        print("Переходы: planning → execution confirm → validation → done confirm")
        return memory
    if action in ("pause", "resume") and not value:
        return dataclasses.replace(memory, state=dataclasses.replace(state, paused=action == "pause"))
    if action in ("step", "expected", "summary") and value:
        return dataclasses.replace(memory, state=dataclasses.replace(state, **{action: value}))
    if action == "go":
        args = value.lower().split()
        if len(args) not in (1, 2) or (len(args) == 2 and args[1] != "confirm"):
            raise AppError("Использование: /state go ЭТАП [confirm].")
        return dataclasses.replace(memory, state=transition_state(state, args[0], len(args) == 2))
    raise AppError("Использование: /state help.")


def run_personalization_test(client: Any, access_token: str, options: Options) -> None:
    print("Опыт: 2 одинаковых вопроса и 10 вопросов подряд (минимум 12 запросов "
          "к модели; возможны дополнительные шаги). Оценка вручную.")
    if ask("Начать платный опыт? [да/нет]: ").strip().lower() not in ("да", "д", "yes", "y"):
        return
    profiles = [UserProfile("brief", "кратко и неформально", "без кода", ""),
                UserProfile("detailed", "подробно и формально", "с примером кода", "")]
    questions = ["Что такое функция в Python?", "Чем аргумент отличается от параметра?",
                 "Что возвращает функция без return?", "Как передать значение по умолчанию?",
                 "Что такое именованный аргумент?", "Зачем нужна документация функции?",
                 "Как вернуть два значения?", "Что такое локальная переменная?",
                 "Как передать функцию другой функции?", "Когда полезна рекурсия?"]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = OUTPUT_DIR / f"w3t3_personalization_{stamp}"
    json_path, csv_path = base.with_suffix(".json"), base.with_suffix(".csv")
    results: list[dict[str, Any]] = []

    def summary() -> dict[str, Any]:
        stable = [row for row in results if row["experiment"] == "stability"]
        violation = next((row["number"] for row in stable
                          if "нарушено" in row["assessment"].values()), None)
        uncertain = any(row["error"] or len(row["assessment"]) != len(row["criteria"]) or
                        "не определено" in row["assessment"].values()
                        for row in stable if violation is None or row["number"] < violation)
        counts: dict[str, Any] = {}
        for experiment in ("comparison", "stability"):
            rows = [row for row in results if row["experiment"] == experiment]
            counts[experiment] = {
                "total": len(rows),
                "all_satisfied": sum(not row["error"] and bool(row["assessment"]) and
                                     len(row["assessment"]) == len(row["criteria"]) and
                                     all(value == "соблюдено" for value in row["assessment"].values())
                                     for row in rows),
                "per_criterion": {key: sum(not row["error"] and row["assessment"].get(key) == "соблюдено"
                                           for row in rows)
                                  for key in ("length", "tone", "format")},
                "errors": sum(bool(row["error"]) for row in rows),
                "evaluated": sum(not row["error"] and len(row["assessment"]) == len(row["criteria"])
                                 and "не определено" not in row["assessment"].values() for row in rows),
                "undetermined": sum(not row["error"] and (len(row["assessment"]) != len(row["criteria"])
                                    or "не определено" in row["assessment"].values()) for row in rows),
            }
        return {"first_violation": violation, "first_violation_certain": not uncertain,
                "stability_complete": len(stable) == 10, "counts": counts}

    def save() -> None:
        try:
            json_path.write_text(json.dumps({"results": results, "summary": summary()},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("experiment", "profile", "number", "question",
                                        "answer", "criteria", "assessment", "error"), delimiter=";")
                writer.writeheader()
                for row in results:
                    writer.writerow({**row, "criteria": json.dumps(row["criteria"], ensure_ascii=False),
                                     "assessment": json.dumps(row["assessment"], ensure_ascii=False)})
        except OSError as exc:
            raise AppError(f"Не удалось сохранить результаты опыта: {exc}") from exc

    sequences = [("comparison", p, [questions[0]]) for p in profiles]
    sequences.append(("stability", profiles[0], questions))
    for experiment, profile, sequence in sequences:
        history = History("", keep=None, strategy=STRATEGY_SUMMARY)
        for number, question in enumerate(sequence, 1):
            criteria = {"length": "кратко" if profile.name == "brief" else "подробно",
                        "tone": "неформально" if profile.name == "brief" else "формально",
                        "format": profile.format}
            row: dict[str, Any] = {"experiment": experiment, "profile": profile.name,
                                   "number": number, "question": question, "answer": "",
                                   "criteria": criteria, "assessment": {}, "error": ""}
            evaluating = False
            try:
                run = run_agent(question, client, access_token, options.model, options.log_file,
                                history, context=(number, question), include_tools=False,
                                layers=profile.block())
                row["answer"] = run.answer
                if run.finish_reason == "length":
                    raise AppError("Ответ обрезан из-за ограничения длины (finish_reason=length).")
                history = history.extended(ChatTurn(question, run.answer, run.steps, usage=run))
                print(f"\n{experiment}, {profile.name}, вопрос {number}: {question}\n{run.answer}")
                evaluating = True
                for key, criterion in criteria.items():
                    while True:
                        value = ask(f"{key} ({criterion}): соблюдено [д], нарушено [н], не определено [?]: ").strip().lower()
                        if value in ("д", "н", "?"):
                            row["assessment"][key] = {"д": "соблюдено", "н": "нарушено", "?": "не определено"}[value]
                            break
            except AppError as exc:
                row["error"] = str(exc)
                print(f"  ошибка: {exc}", file=sys.stderr)
            results.append(row)
            save()
            if evaluating and row["error"]:
                raise AppError("Оценка прервана; частичные результаты сохранены.")
    for experiment in ("comparison", "stability"):
        rows = [row for row in results if row["experiment"] == experiment]
        good = sum(bool(row["assessment"]) and all(v == "соблюдено" for v in row["assessment"].values())
                   and len(row["assessment"]) == len(row["criteria"]) and not row["error"] for row in rows)
        counts = summary()["counts"][experiment]
        print(f"{experiment}: все предпочтения соблюдены в {good} из {counts['evaluated']} оценённых ответов; "
              f"неопределённых ответов: {counts['undetermined']}, ошибок запросов: {counts['errors']}")
        for key, count in counts["per_criterion"].items():
            evaluated = sum(not row["error"] and row["assessment"].get(key) in ("соблюдено", "нарушено")
                            for row in rows)
            print(f"  {key}: соблюдено в {count} из {evaluated} оценённых ответов")
    outcome = summary()
    violation = outcome["first_violation"]
    if not outcome["first_violation_certain"]:
        print(f"Первое нарушение: точный номер неизвестен" + (f"; обнаружено на ответе {violation}" if violation else "."))
    elif violation is None:
        print("За 10 ответов нарушений не обнаружено.")
    else:
        print(f"Первое нарушение: ответ №{violation}.")
    print(f"Результаты: {json_path}, {csv_path}")


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и портит кириллицу.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        options = parse_args(sys.argv[1:])
        options = profile_options(options, "default")
        profile = load_profile("default")
        load_env_file(options.env_path)
        if options.provider == "veai":
            credentials = os.environ.get(VEAI_CREDENTIALS_ENV)
            if not credentials:
                raise AppError(f"В {options.env_path} не найден ключ {VEAI_CREDENTIALS_ENV}.")
            client = VeaiClient(api_key=credentials)
            access_token = ""
        else:
            credentials = (os.environ.get(GIGACHAT_CREDENTIALS_ENV)
                           or os.environ.get(GIGACHAT_FALLBACK_CREDENTIALS_ENV))
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
            access_token = client.get_access_token(options.use_cache)

        print(f"Профиль: {profile.name} (/profile — управление, /personal-test — опыт)")
        print(f"Провайдер: {options.provider}")
        print(f"Ключи: {options.env_path}")
        print(f"Модель: {options.model}")
        if options.provider == "gigachat":
            print_gigachat_balance(client, access_token, options.model)
        print("Инструменты: " + ", ".join(tool.name for tool in TOOLS))
        print(f"HTTP-лог: {options.http_log_file}")
        print(f"Журнал вызовов: {options.log_file}")
        print(f"История: {options.history_file}")
        print(f"Сводка: {options.summary_file}")
        print(f"Рабочая память: {options.work_file.name}, "
              f"долговременная: {options.long_file.name}")
        print(f"Ветки/checkpoints: {options.branches_file}")
        print(f"Извлечение фактов моделью: "
              f"{'включено' if options.extract else 'выключено'}")
        print(f"Лимит контекста: {options.model_limit} токенов")
        history = load_history(options.history_file, options.summary_file,
                              options.keep_turns, options.strategy)
        memory = load_memory(options.work_file, options.long_file)
        details = load_details(DETAILS_FILE) if DETAILS_FILE.is_file() else []

    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except (GigaChatError, VeaiError) as exc:
        print(f"Ошибка авторизации: {exc}", file=sys.stderr)
        return 1

    runs: list[AgentRun] = []
    branches: dict[str, History] = {}
    # Номер обращения к модели в сеансе: считается по всем, включая упавшие
    # с ошибкой — в HTTP-логе номера должны идти подряд.
    received = 0
    print()
    print_help(history, memory)
    if options.fill_words and memory.state.paused:
        print("Задача на паузе: начальное заполнение пропущено.")
    elif options.fill_words:
        received += 1
        history = fill_history(history, options, client, access_token, runs, received,
                               memory, [*profile.block(), *memory.state.block()])
    while True:
        try:
            query = ask_command(f"\nЗапрос (или {EXIT_COMMAND}): ", options.command_prefix)
        except AppError as exc:
            # Конец ввода (Ctrl+D, закрытый поток) — тот же выход, что и /exit.
            print(f"\n{exc}")
            break
        try:
            query = resolve_command(query, options.command_prefix)
        except AppError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
            continue
        command = query.lower()
        if command == "/profile" or command.startswith("/profile "):
            try:
                if command == "/profile select" or command.startswith("/profile use "):
                    if command == "/profile select":
                        names = profile_names()
                        if not names:
                            print("Нет профилей для выбора.")
                            continue
                        print_profiles(names, profile.name)
                        choice = ask("Номер профиля (Enter — отмена): ")
                        if not choice:
                            continue
                        if not choice.isascii() or not choice.isdecimal() or not 1 <= int(choice) <= len(names):
                            raise AppError("Укажите номер профиля из списка.")
                        name = names[int(choice) - 1]
                    else:
                        name = query[len("/profile use "):].strip()
                    new_options = profile_options(options, name)
                    new_profile = load_profile(name)
                    new_history = load_history(new_options.history_file, new_options.summary_file,
                                               new_options.keep_turns, new_options.strategy)
                    new_memory = load_memory(new_options.work_file, new_options.long_file)
                    options, profile, history, memory = new_options, new_profile, new_history, new_memory
                    branches = {}
                    print(f"Активный профиль: {profile.name}")
                else:
                    profile = profile_command(query, profile)
            except AppError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
            continue
        if command == "/state" or command.startswith("/state "):
            try:
                updated = state_command(query, memory)
                if updated != memory:
                    save_memory(updated, options.work_file, options.long_file)
                    memory = updated
                    print("Состояние сохранено.")
            except AppError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
            continue
        if command == "/personal-test":
            try:
                run_personalization_test(client, access_token, options)
            except AppError as exc:
                print(f"Ошибка опыта: {exc}", file=sys.stderr)
            continue
        if command == EXIT_COMMAND:
            break
        if command == CLEAR_COMMAND:
            # /clear — это и есть «забыть краткосрочную память»: история и сводка
            # удаляются, рабочая задача закрывается, долговременная память цела.
            try:
                clear_history(options.history_file)
                clear_summary(options.summary_file)
            except AppError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
                continue
            new_history = History(datetime.now().astimezone().strftime(TIME_FORMAT),
                                  keep=options.keep_turns, strategy=options.strategy)
            new_memory = memory.open_task("")
            try:
                save_memory(new_memory, options.work_file, options.long_file)
            except AppError as exc:
                print(f"Ошибка сохранения задачи: {exc}", file=sys.stderr)
                break
            memory, history = new_memory, new_history
            print("История и сводка очищены, рабочая память ушла в архив — "
                  "следующий запрос начнёт беседу заново.")
            continue
        if command == HELP_COMMAND:
            print_commands()
            continue
        if command == CLEAR_LAYER_COMMAND or command.startswith(CLEAR_LAYER_COMMAND + " "):
            try:
                memory, history = clear_layer(query, memory, history, options)
            except AppError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
            continue
        if command == MEMORY_COMMAND:
            print_memory(memory, history, options)
            continue
        if command == LIMITS_COMMAND:
            print_session_limits(runs, history, options, memory)
            continue
        if command == COMPARE_COMMAND:
            compare_cleanup(options.work_file.parent / CLEANUP_FILE.name)
            continue
        if command.startswith(REMEMBER_COMMAND):
            memory = _remember(query, memory, options)
            continue
        if command.startswith(FORGET_COMMAND):
            memory = _forget(query, memory, options)
            continue
        if command.startswith(TASK_COMMAND):
            try:
                memory, history = task_command(query, memory, history, options)
            except AppError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
            continue
        if command == CLEANUP_COMMAND:
            if memory.state.paused:
                print("Задача на паузе: сначала выполните /state resume.")
                continue
            if not details:
                print(f"Нет файла деталей {DETAILS_FILE.name} — проверять нечего.")
                continue
            # Порог для опыта: автoсжатие выключено, извлечение выключено —
            # иначе различия по токенам смешались бы с этими запросами.
            probe = replace(options, extract=False, keep_turns=None,
                            strategy=STRATEGY_SLIDING)
            probe = Options(**{**probe.__dict__, "extract": False})
            results = run_cleanup(details, client, runs, history, memory, probe,
                                 received)
            received += len(results) + len(CLEANUP_STATES)
            if results:
                try:
                    save_cleanup(options.work_file.parent / CLEANUP_FILE.name, results, memory, options)
                except AppError as exc:
                    print(f"  предупреждение: {exc}", file=sys.stderr)
                print(f"  результаты: {CLEANUP_FILE.name}, "
                      f"сравнение — {COMPARE_COMMAND}")
            continue
        if command == CHECK_COMMAND or command.startswith(CHECK_COMMAND + " "):
            if memory.state.paused:
                print("Задача на паузе: сначала выполните /state resume.")
                continue
            if not details:
                print(f"Нет файла деталей {DETAILS_FILE.name} — проверять нечего.")
                continue
            if not history.turns:
                print("История пуста, проверка памяти не имеет смысла.")
                continue
            checked = check_details(details, client, runs, history, memory,
                                    options, received)
            received += len(checked)
            print_table(("#", "Деталь", "Ожидалось", "Ответ модели", "Помнит"),
                        [(item["id"], item["kind"], item["fact"], item["answer"][:60],
                          "да" if item["hit"] else "нет") for item in checked])
            found = sum(1 for item in checked if item["hit"])
            print(f"  найдено деталей: {found} из {len(checked)}, вход на вопросы: "
                  f"{sum(item['prompt_tokens'] for item in checked)} токенов")
            continue
        if command.startswith("/checkpoint"):
            name = query[len("/checkpoint"):].strip() or "checkpoint"
            try:
                save_checkpoint(history, name, options.branches_file)
            except AppError as exc:
                print(f"Ошибка: {exc}")
            continue
        if command.startswith("/branch"):
            name = query[len("/branch"):].strip()
            if not name:
                print("Использование: /branch NAME")
                continue
            branches[name] = history
            try:
                save_checkpoint(history, name, options.branches_file)
            except AppError as exc:
                print(f"Ошибка: {exc}")
            print(f"Ветка «{name}» создана от текущей точки.")
            continue
        if command.startswith("/switch"):
            name = query[len("/switch"):].strip()
            if not name:
                print("Доступные ветки: " + (", ".join(branches) or "нет"))
                continue
            try:
                history = branches.get(name) or load_checkpoint(
                    name, options.branches_file, options.strategy, options.keep_turns)
                branches[name] = history
                print(f"Переключено на ветку «{name}»: {len(history.turns)} обменов.")
            except AppError as exc:
                print(f"Ошибка: {exc}")
            continue
        if command == COMPRESS_COMMAND or command.startswith(COMPRESS_COMMAND + " "):
            if not options.compression:
                print("Сжатие выключено (--no-compress): вся история уходит как есть.")
                continue
            limit = _parse_compress_count(query)
            if limit == 0:
                continue
            received += 1
            history = _compress_and_save(history, options, client, runs, received, limit)
            continue
        if command == FILL_COMMAND or command.startswith(FILL_COMMAND + " "):
            if memory.state.paused:
                print("Задача на паузе: сначала выполните /state resume.")
                continue
            words = options.fill_words or _parse_fill_words(query, options)
            if not words:
                continue
            received += 1
            history = fill_history(history, replace(options, fill_words=words),
                                   client, access_token, runs, received, memory,
                                   [*profile.block(), *memory.state.block()])
            continue
        if not query:
            continue
        if query.startswith("/"):
            print("Неизвестная команда или неверные аргументы. Введите /help.", file=sys.stderr)
            continue
        if memory.state.paused:
            print("Задача на паузе: сначала выполните /state resume.")
            continue

        received += 1
        try:
            profile = extract_preferences(query, profile, options, client, access_token,
                                          received, runs)
        except AppError as exc:
            print(f"Ошибка сохранения профиля: {exc}", file=sys.stderr)
            continue
        print("\nЦикл агента")
        try:
            run = run_agent(
                query, client, access_token, options.model, options.log_file,
                history, on_step=print_event, context=(received, query),
                layers=[*profile.block(), *memory.state.block(), *memory.blocks()],
            )
        except AppError as exc:
            # Отказ по одному запросу не закрывает диалог: причину видно сразу,
            # а следующий запрос можно задать снова.
            print(f"  ошибка: {exc}")
            continue
        runs.append(run)
        transcript = memory.state.transcript
        transcript += ("\n" if transcript else "") + f"Пользователь: {query}\nАссистент: {run.answer}"
        memory = dataclasses.replace(memory, state=dataclasses.replace(memory.state, transcript=transcript))
        print_answer(run, history_tokens(history, memory), options.model_limit)
        # Обмен сохраняется сразу после ответа: даже оборванный запуск
        # не теряет уже полученные ответы.
        turn = ChatTurn(
            query=query,
            answer=run.answer,
            steps=run.steps,
            at=datetime.now().astimezone().strftime(TIME_FORMAT),
            usage=run,
        )
        # Раскладываем в слои ДО записи в историю: если обмен нёс факт для
        # рабочего или долговременного слоя, он в историю уже не попадёт
        # и не дублируется в живом окне.
        history = history.extended(turn)
        memory, history = _extract(history, memory, options, client, access_token,
                                   runs, received)
        history = brief_history(history, SHORT_TERM_TURNS)
        try:
            save_history(history, options.history_file)
            save_memory(memory, options.work_file, options.long_file)
        except AppError as exc:
            print(f"  предупреждение: {exc}", file=sys.stderr)
        tokens = history_tokens(history, memory)
        print(f"  в истории разговора: {len(history.turns)} обменов, "
              + _context_line(tokens, options.model_limit))
        if options.strategy == STRATEGY_SUMMARY and len(history.pending_turns()) >= options.compress_every:
            received += 1
            history = _compress_and_save(history, options, client, runs, received,
                                         None)

    return finish(runs, history, options, memory)


def _compress_and_save(history: History, options: Options,
                       client: GigaChatClient, runs: list[AgentRun],
                       received: int, limit: int | None) -> History:
    """Сжимает историю и сохраняет историю со сводкой — двумя файлами."""
    compressed = compress_history(history, options, client, runs,
                                  limit=limit, context=(received, ""))
    if compressed is history:
        return history
    try:
        save_history(compressed, options.history_file)
        save_summary(compressed.summary, options.summary_file)
    except AppError as exc:
        print(f"  предупреждение: {exc}", file=sys.stderr)
    print(f"  в истории: {len(compressed.turns)} обменов, в сводке "
          f"{compressed.summary.compressed_turns}")
    return compressed


def _parse_compress_count(query: str) -> int | None:
    """Аргумент /compress: сколько обменов сжать; None — всё несжатое."""
    parts = query.split()
    if len(parts) == 1:
        return None
    try:
        count = int(parts[1])
    except ValueError:
        print(f"Не понимаю «{parts[1]}»: нужно {COMPRESS_COMMAND} или "
              f"{COMPRESS_COMMAND} N.")
        return 0
    if count < 1:
        print(f"Сжимать надо хотя бы один обмен.")
        return 0
    return count


def finish(runs: list[AgentRun], history: History, options: Options,
           memory: Memory | None = None) -> int:
    """Итог диалога: куда легли слои памяти и что с ними будет дальше."""
    memory = memory or Memory()
    print()
    if history.turns:
        print(f"Краткосрочная память: {len(history.turns)} обменов "
              f"({options.history_file.name})")
        print("Следующий запуск продолжит этот разговор, "
              f"{CLEAR_COMMAND} его сотрёт.")
        if options.compression:
            print(f"  в сводке сжато обменов: {history.summary.compressed_turns}; "
                  f"как есть уйдёт {len(history.context_turns())}.")
    else:
        print("Краткосрочная память пуста.")
    print(f"Рабочая память: {len(memory.working.facts)} записей, "
          f"задача №{memory.task} ({options.work_file.name})")
    print(f"Долговременная память: {len(memory.long.facts)} записей "
          f"({options.long_file.name})")
    if memory.archive:
        print(f"  в архиве задач: {len(memory.archive)}")
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
    prompt = sum(run.prompt_tokens for run in runs)
    completion = sum(run.completion_tokens for run in runs)
    print(f"Токенов за сеанс: {prompt} вход + {completion} выход = {prompt + completion}")
    try:
        json_path, csv_path = export(runs, options, "w3t3_" + options.model)
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    print(f"Результаты: {json_path.name}, {csv_path.name}")
    print(f"Журнал вызовов: {options.log_file}")
    print(f"HTTP-лог: {options.http_log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
