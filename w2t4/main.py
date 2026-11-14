"""Агент на GigaChat с управлением контекстом: сжатие истории в сводку.

Сценарий:
1. читает ключ GigaChat, лимит контекста модели, историю и её сводку;
2. спрашивает запрос у пользователя;
3. отправляет модели сводку сжатой части разговора, последние N обменов
   как есть и описание инструментов;
4. если модель просит вызвать инструмент, выполняет функцию локально и пишет
   вызов в журнал, а результат возвращает модели;
5. повторяет шаг 4, пока модель не ответит текстом, и показывает ответ
   вместе с расходом токенов: за шаги цикла, за сеанс, в истории
   и в самом ответе;
6. дописывает обмен в историю и, если несжатых обменов накопилось на сводку,
   сжимает их: модель пересказывает, пересказ дописывается в сводку, а сами
   обмены в контекст больше не идут — до /exit.

Команды: /clear стирает историю и сводку, /compress сжимает вручную,
/check прогоняет вопросы по деталям, /limits показывает расход и состав
контекста, /fill наполняет историю заполнителем, /help — справка.

Токены считаются дважды: по `usage` из ответа API (это факт и деньги) и своей
оценкой того, сколько уйдёт следующим запросом (API не говорит заранее).
Разница двух чисел показывает, насколько оценке можно доверять.
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
# История разговора: без неё каждый запуск начинал бы беседу заново.
DEFAULT_HISTORY_FILE = PROJECT_DIR / "history.json"
# Сводка сжатой части разговора — отдельный файл: история не переписывается,
# сжатые обмены остаются на диске, а в контекст идёт только сводка.
DEFAULT_SUMMARY_FILE = PROJECT_DIR / "summary.json"
# Сколько последних обменов уходит модели как есть. Всё, что старше,
# заменяется сводкой.
DEFAULT_KEEP_TURNS = 4
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
# Отступ в JSON-записи HTTP-лога: один обмен читается глазами, вложенные
# сообщения и тела ответов стоят на своих строках.
HTTP_LOG_INDENT = 2

# Запросов на один вопрос. Агент с тремя инструментами укладывается в два шага;
# лимит нужен, чтобы модель, зациклившаяся на вызовах, не тратила токены.
MAX_STEPS = 6
# Команды диалога.
EXIT_COMMAND = "/exit"
CLEAR_COMMAND = "/clear"
LIMITS_COMMAND = "/limits"
FILL_COMMAND = "/fill"
HELP_COMMAND = "/help"
COMPRESS_COMMAND = "/compress"
CHECK_COMMAND = "/check"
# Справка выводится из таблиц, а обработка команд — из тех же констант, которые
# печатает таблица, поэтому справка не может разойтись с поведением.
COMMAND_HELP = (
    (EXIT_COMMAND, "выйти; история и сводка остаются на диске"),
    (CLEAR_COMMAND, "забыть разговор: удалить историю и сводку"),
    (LIMITS_COMMAND, "расход токенов, состав контекста и сводка"),
    (COMPRESS_COMMAND, "сжать старые обмены в сводку: /compress или /compress N"),
    (CHECK_COMMAND, "прогнать вопросы по деталям и сохранить ответы для сравнения"),
    (FILL_COMMAND, "наполнить историю словами: до лимита или /fill N слов"),
    (HELP_COMMAND, "эта справка"),
)
OPTION_HELP = (
    ("--env PATH", "файл с ключом вместо поиска .env вверх по дереву"),
    ("--model NAME", f"модель GigaChat (по умолчанию {DEFAULT_MODEL})"),
    ("--limit TOKENS", "лимит контекста модели для оценки истории"),
    ("--keep N", f"сколько обменов держать как есть (по умолчанию {DEFAULT_KEEP_TURNS})"),
    ("--every N", f"через сколько обменов сжимать (по умолчанию {DEFAULT_COMPRESS_EVERY})"),
    ("--no-compress", "выключить сжатие: контекст как в w2t3, вся история"),
    ("--history PATH", "другой файл истории разговора"),
    ("--summary PATH", "другой файл сводки"),
    ("--log PATH", "другой журнал вызовов инструментов"),
    ("--http-log PATH", "другой HTTP-лог диалога с моделью"),
    ("--no-cache", "переавторизоваться, минуя кэш токена"),
)

# Инструмент, результат которого остаётся в истории как часть разговора: текст
# страницы кладётся в контекст и живёт там до /clear. Для исследования это
# сознательный выбор вопреки экономии: такая страница платится входными
# токенами на каждом следующем запросе.
PERSISTED_TOOLS = frozenset({"fetch_url"})

TIME_FORMAT = "%Y-%m-%d %H:%M:%S %Z"
SYSTEM_PROMPT = (
    "Ты агент с инструментами. Если для ответа нужен случайный бросок, текущее "
    "время или перевод числа в шестнадцатеричную систему, вызови подходящую "
    "функцию. Не выдумывай результаты броска и не считай системы счисления "
    "сам — вызывай функцию. Начало разговора может быть сжато в сводку: "
    "используй её как память о прошлом. Отвечай кратко, по-русски, называя "
    "полученное число."
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
class History:
    """История разговора: начало беседы и все её обмены по порядку.

    Сжатие историю не удаляет: `turns` — весь разговор, а в контекст уходят
    только последние `keep` обменов, всё старшее заменяется `summary`.
    """

    started: str
    turns: list[ChatTurn] = field(default_factory=list)
    summary: Summary = field(default_factory=Summary)
    # Последние обмены уходят как есть; None — весь разговор целиком.
    keep: int | None = None

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

    def messages(self) -> list[dict[str, Any]]:
        """Контекст истории для модели: сводка, затем последние обмены.

        Сводка идёт первым сообщением user/assistant — так она не путается
        с текущим вопросом и видна модели как прошлый опыт. Дальше — обычные
        обмены, а у обменов с fetch_url — ещё и текст страницы.
        """
        messages: list[dict[str, Any]] = []
        if self.summary.text:
            messages.extend(summary_messages(self.summary))
        for turn in self.context_turns():
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
        return History(self.started, [*self.turns, turn], self.summary, self.keep)

    def compressed(self, summary: Summary) -> History:
        return History(self.started, self.turns, summary, self.keep)


@dataclass(frozen=True)
class Options:
    env_path: Path
    model: str
    log_file: Path
    http_log_file: Path
    history_file: Path
    summary_file: Path
    use_cache: bool
    model_limit: int = DEFAULT_MODEL_LIMIT
    # Сколько последних обменов держать как есть; None — не сжимать вообще.
    keep_turns: int | None = DEFAULT_KEEP_TURNS
    # Через сколько несжатых обменов срабатывает автосжатие.
    compress_every: int = DEFAULT_COMPRESS_EVERY
    # Сколько слов-заполнителей вбросить в историю сразу после запуска
    # (режим /fill до первой строки): нужно для сценария «за лимитом».
    fill_words: int = 0

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


def summary_messages(summary: Summary) -> list[dict[str, Any]]:
    """Сводка в виде пары сообщений: она подаётся как прошлый диалог."""
    return [
        {
            "role": "user",
            "content": ("Сводка нашего разговора до этого места, чтобы ты помнил "
                        "детали:\n" + summary.text),
        },
        {"role": "assistant", "content": "Понял сводку, помню всё перечисленное."},
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


def parse_args(argv: list[str]) -> Options:
    env_path: Path | None = None
    model = DEFAULT_MODEL
    log_file = DEFAULT_LOG_FILE
    http_log_file = DEFAULT_HTTP_LOG_FILE
    history_file = DEFAULT_HISTORY_FILE
    summary_file = DEFAULT_SUMMARY_FILE
    model_limit = DEFAULT_MODEL_LIMIT
    fill_words = 0
    keep_turns: int | None = DEFAULT_KEEP_TURNS
    compress_every = DEFAULT_COMPRESS_EVERY
    use_cache = True
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--no-cache":
            use_cache = False
        elif arg == "--no-compress":
            keep_turns = None
        elif arg in ("--env", "--model", "--log", "--http-log", "--history",
                     "--summary", "--limit", "--fill", "--keep", "--every"):
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
            elif arg == "--summary":
                summary_file = Path(value).expanduser()
            elif arg == "--limit":
                model_limit = _positive_int(arg, value)
            elif arg == "--fill":
                fill_words = _positive_int(arg, value)
            elif arg == "--keep":
                keep_turns = _positive_int(arg, value)
            elif arg == "--every":
                compress_every = _positive_int(arg, value)
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
        elif arg.startswith("--summary="):
            summary_file = Path(arg[len("--summary=") :]).expanduser()
        elif arg.startswith("--limit="):
            model_limit = _positive_int("--limit", arg[len("--limit=") :])
        elif arg.startswith("--fill="):
            fill_words = _positive_int("--fill", arg[len("--fill=") :])
        elif arg.startswith("--keep="):
            keep_turns = _positive_int("--keep", arg[len("--keep=") :])
        elif arg.startswith("--every="):
            compress_every = _positive_int("--every", arg[len("--every=") :])
        elif arg.startswith("--http-log="):
            http_log_file = Path(arg[len("--http-log=") :]).expanduser()
        else:
            raise AppError(
                f"Неизвестный аргумент: {arg}. Доступны: "
                + ", ".join(flag.split()[0] if " " in flag else flag
                            for flag, _ in OPTION_HELP)
                + "."
            )
    return Options(
        env_path or find_env_file(ENV_SEARCH_START), model, log_file,
        http_log_file, history_file, summary_file, use_cache, model_limit,
        keep_turns, compress_every, fill_words,
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


def run_agent(
    query: str,
    client: GigaChatClient,
    access_token: str,
    model: str,
    log_file: Path,
    history: History,
    on_step: Callable[[Step, int], None] | None = None,
    context: tuple[int, str] = (0, ""),
    include_tools: bool = True,
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

    Оценка входа считается на каждом шаге по тем же сообщениям, что идут в API,
    и сохраняется рядом с фактом того же шага (`StepTokenUsage.estimated`).
    Сравнивать их надо именно на одном шаге: после вызова инструмента
    в сообщениях появляется его результат, и «оценка до цикла» против
    «факта после цикла» — уже разные запросы.
    """
    query_number, user_query = context
    started = time.time()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history.messages(),
        {"role": "user", "content": user_query},
    ]
    steps: list[Step] = []
    usages: list[StepTokenUsage] = []
    prompt_tokens = completion_tokens = cached_tokens = 0

    for step_number in range(1, MAX_STEPS + 1):
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if include_tools:
            body["functions"] = [tool.schema() for tool in TOOLS]
            body["function_call"] = "auto"
        client.log_context = {
            "query_number": query_number,
            "query": user_query,
            "step": step_number,
        }
        try:
            # Токен не передаём: клиент сам обновит его, если GigaChat
            # ответит 401 после долгого простоя.
            response = client.chat_messages(None, body)
        except GigaChatError as exc:
            raise AppError(f"GigaChat не ответил: {exc}") from exc
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
                cached_tokens, usages,
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
    return text if len(text) <= limit else text[:limit] + "…"


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


def check_details(details: list[dict[str, str]], client: GigaChatClient,
                  runs: list[AgentRun], history: History, options: Options,
                  received: int) -> tuple[History, list[dict[str, Any]]]:
    """Задаёт модели вопрос по каждой детали и сохраняет ответы файлом.

    Запросы идут обычным путём агента, поэтому обмены попадают в историю:
    сначала все вопросы без сжатия, потом все со сжатием — на одной беседе.
    """
    print(f"\nПроверка деталей: {len(details)} вопросов, "
          f"сжатие {'включено' if options.compression else 'выключено'}")
    results: list[dict[str, Any]] = []
    for item in details:
        question = str(item.get("question") or "")
        run = run_agent(question, client, "", options.model, options.log_file,
                        history, context=(received, question))
        runs.append(run)
        history = history.extended(ChatTurn(
            query=question, answer=run.answer, steps=run.steps,
            at=datetime.now().astimezone().strftime(TIME_FORMAT), usage=run,
        ))
        fact = str(item.get("expected") or "")
        answer = run.answer.strip()
        results.append({
            "id": item.get("id", ""),
            "kind": item.get("kind", ""),
            "fact": fact,
            "question": question,
            "answer": answer,
            "hit": _mentions(answer, fact),
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
        })
    print_table(
        ("#", "Деталь", "Ожидалось", "Ответ модели", "Помнит"),
        [(item["id"], item["kind"], item["fact"], item["answer"][:60],
          "да" if item["hit"] else "нет") for item in results],
    )
    found = sum(1 for item in results if item["hit"])
    exact = sum(1 for item in results if item["hit"] == "прямо")
    print(f"  найдено деталей: {found} из {len(results)} "
          f"(дословно {exact}), вход на вопросы: "
          f"{sum(item['prompt_tokens'] for item in results)} токенов")
    return history, results


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


def save_checks(path: Path, results: list[dict[str, Any]]) -> None:
    """Результаты проверки — файлом, чтобы их можно было разобрать потом."""
    payload = {
        "version": 1,
        "checked": datetime.now().astimezone().strftime(TIME_FORMAT),
        "results": results,
        "summary": {
            "details": len(results),
            "found": sum(1 for item in results if item["hit"]),
            "exact": sum(1 for item in results if item["hit"] == "прямо"),
            "prompt_tokens": sum(item["prompt_tokens"] for item in results),
        },
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    except OSError as exc:
        raise AppError(f"Не удалось сохранить {path}: {exc}") from exc


def print_help(history: History) -> None:
    print(f"Введите запрос и нажмите Enter. {HELP_COMMAND} — все команды "
          f"и параметры запуска, {EXIT_COMMAND} — выход.")
    print("Например:")
    for query in EXAMPLE_QUERIES:
        print(f"  {query}")
    if history.turns:
        print(f"Продолжается разговор из {len(history.turns)} обменов "
              f"(начат {history.started}).")


def print_commands() -> None:
    """Полная справка: команды диалога и параметры запуска."""
    print("Команды диалога:")
    for command, about in COMMAND_HELP:
        print(f"  {command:<12} {about}")
    print()
    print("Параметры запуска (для --flag=value и --flag value):")
    for flag, about in OPTION_HELP:
        print(f"  {flag:<20} {about}")
    print()
    print("Файлы по умолчанию:")
    print(f"  история разговора   {DEFAULT_HISTORY_FILE}")
    print(f"  сводка контекста    {DEFAULT_SUMMARY_FILE}")
    print(f"  контрольные детали  {DETAILS_FILE}")
    print(f"  журнал инструментов {DEFAULT_LOG_FILE}")
    print(f"  HTTP-лог диалога    {DEFAULT_HTTP_LOG_FILE}")
    print(f"  модель по умолчанию {DEFAULT_MODEL}, лимит {DEFAULT_MODEL_LIMIT} токенов")
    print("Ключ GIGACHAT_CREDENTIALS берётся из .env в поиске вверх от папки задачи.")


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
                 keep: int | None = None) -> History:
    """Читает сохранённый разговор и его сводку; битый файл — ошибка, а не потеря."""
    if not history_file.is_file():
        return History(datetime.now().astimezone().strftime(TIME_FORMAT),
                       summary=load_summary(summary_file) if summary_file else Summary(),
                       keep=keep)
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
    return History(str(payload.get("started", "")), turns,
                   load_summary(summary_file) if summary_file else Summary(), keep)


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


def history_tokens(history: History) -> int:
    """Оценка того, сколько токенов займёт история в следующем запросе.

    Считается по тем же сообщениям, что уйдут модели: системная подсказка,
    сводка и живые обмены (текст нового запроса добавится потом — это нижняя
    граница). Описания инструментов прибавляются отдельно.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history.messages()]
    return estimate_messages_tokens(messages) + estimate_tokens(
        json.dumps([tool.schema() for tool in TOOLS], ensure_ascii=False)
    )


def full_history_tokens(history: History) -> int:
    """То же без сжатия: столько было бы, если бы ушла вся история целиком."""
    everything = dataclasses.replace(history, summary=Summary(), keep=None)
    return history_tokens(everything)


def compression_lines(history: History, options: Options) -> list[str]:
    """Сколько сейчас стоит контекст и сколько сэкономило сжатие.

    Первая строка — без отступа (её печатают после слова «Сжатие:»),
    остальные — с отступом в готовом виде.
    """
    now = history_tokens(history)
    full = full_history_tokens(history)
    saved = max(0, full - now)
    summary_tokens = (estimate_messages_tokens(summary_messages(history.summary))
                      if history.summary.text else 0)
    live = len(history.context_turns())
    saved_share = f"{100.0 * saved / full:.0f}%" if full else "0%"
    lines = [
        f" ~{now} токенов, без сжатия было бы ~{full} токенов",
        f"  сжато: {history.summary.compressed_turns} обменов → сводка "
        f"~{summary_tokens} токенов, экономия ~{saved} токенов ({saved_share})",
        f"  как есть: {live} обменов из {len(history.turns)}"
        + (f", до автосжатия: {max(0, options.compress_every - len(history.pending_turns()))} обменов"
           if options.compression else ", сжатие выключено"),
    ]
    if history.summary.text:
        lines.append("  сводка: " + history.summary.text.replace("\n", " / "))
    return lines


def print_session_limits(runs: list[AgentRun], history: History,
                        options: Options) -> None:
    """/limits: сжатие, история, сеанс и каждый запрос сеанса."""
    print("\nТокены")
    print(f"  модель: {options.model}, лимит контекста {options.model_limit}")
    print_share_line(history_tokens(history), options.model_limit)
    print("Сжатие:" + compression_lines(history, options)[0])
    for line in compression_lines(history, options)[1:]:
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
                 received: int = 0) -> History:
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
    print(f"  оценка запроса: ~{history_tokens(history)} токенов "
          f"без заполнителя + ~{estimate_tokens(query)}")
    run = run_agent(query, client, access_token, options.model,
                    options.log_file, history,
                    context=(received or len(runs) + 1, query))
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
    print_share_line(history_tokens(history), options.model_limit)
    return history


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
        print(f"Сводка: {options.summary_file}")
        print(f"Сжатие: " + (
            f"последние {options.keep_turns} обменов как есть, "
            f"остальное в сводку через каждые {options.compress_every}"
            if options.compression else "выключено (--no-compress)"))
        print(f"Лимит контекста: {options.model_limit} токенов")
        history = load_history(options.history_file, options.summary_file,
                              options.keep_turns)
        details = load_details(DETAILS_FILE) if DETAILS_FILE.is_file() else []

        access_token = client.get_access_token(options.use_cache)
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except GigaChatError as exc:
        print(f"Ошибка авторизации: {exc}", file=sys.stderr)
        return 1

    runs: list[AgentRun] = []
    # Номер обращения к модели в сеансе: считается по всем, включая упавшие
    # с ошибкой — в HTTP-логе номера должны идти подряд.
    received = 0
    print()
    print_help(history)
    if options.fill_words:
        received += 1
        history = fill_history(history, options, client, access_token, runs, received)
    while True:
        try:
            query = ask(f"\nЗапрос (или {EXIT_COMMAND}): ")
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
                clear_summary(options.summary_file)
            except AppError as exc:
                print(f"Ошибка: {exc}", file=sys.stderr)
                continue
            history = History(datetime.now().astimezone().strftime(TIME_FORMAT),
                              keep=options.keep_turns)
            print("История и сводка очищены — следующий запрос начнёт беседу заново.")
            continue
        if command == LIMITS_COMMAND:
            print_session_limits(runs, history, options)
            continue
        if command == HELP_COMMAND:
            print_commands()
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
        if command == CHECK_COMMAND or command.startswith(CHECK_COMMAND + " "):
            if not details:
                print(f"Нет файла деталей {DETAILS_FILE.name} — проверять нечего.")
                continue
            if not history.turns:
                print("История пуста, проверка памяти не имеет смысла.")
                continue
            received += 1
            try:
                history, checked = check_details(details, client, runs, history,
                                                options, received)
            except AppError as exc:
                print(f"  ошибка: {exc}")
                continue
            try:
                save_history(history, options.history_file)
                save_checks(CHECKS_FILE, checked)
            except AppError as exc:
                print(f"  предупреждение: {exc}", file=sys.stderr)
            print(f"  результаты: {CHECKS_FILE.name}")
            continue
        if command == FILL_COMMAND or command.startswith(FILL_COMMAND + " "):
            words = options.fill_words or _parse_fill_words(query, options)
            if not words:
                continue
            received += 1
            history = fill_history(history, replace(options, fill_words=words),
                                   client, access_token, runs, received)
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
        print_answer(run, history_tokens(history), options.model_limit)
        # Обмен сохраняется сразу после ответа: даже оборванный запуск
        # не теряет уже полученные ответы.
        history = history.extended(ChatTurn(
            query=query,
            answer=run.answer,
            steps=run.steps,
            at=datetime.now().astimezone().strftime(TIME_FORMAT),
            usage=run,
        ))
        try:
            save_history(history, options.history_file)
        except AppError as exc:
            print(f"  предупреждение: {exc}", file=sys.stderr)
        tokens = history_tokens(history)
        print(f"  в истории разговора: {len(history.turns)} обменов, "
              + _context_line(tokens, options.model_limit))
        if options.compression and len(history.pending_turns()) >= options.compress_every:
            received += 1
            history = _compress_and_save(history, options, client, runs, received,
                                         None)

    return finish(runs, history, options)


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


def finish(runs: list[AgentRun], history: History, options: Options) -> int:
    """Итог диалога: куда легла история и что с ней будет дальше."""
    print()
    if history.turns:
        print(f"В истории разговора: {len(history.turns)} обменов "
              f"({options.history_file.name})")
        print("Следующий запуск продолжит этот разговор, "
              f"{CLEAR_COMMAND} его сотрёт.")
        if options.compression:
            print(f"В сводке сжато обменов: {history.summary.compressed_turns}; "
                  f"как есть уйдёт {len(history.context_turns())}.")
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
    prompt = sum(run.prompt_tokens for run in runs)
    completion = sum(run.completion_tokens for run in runs)
    print(f"Токенов за сеанс: {prompt} вход + {completion} выход = {prompt + completion}")
    try:
        json_path, csv_path = export(runs, options, "w2t4_" + options.model)
    except AppError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    print(f"Результаты: {json_path.name}, {csv_path.name}")
    print(f"Журнал вызовов: {options.log_file}")
    print(f"HTTP-лог: {options.http_log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
