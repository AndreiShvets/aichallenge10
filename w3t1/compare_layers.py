"""Сравнивает качество распределения 30 естественных фраз по слоям.

Пример:
    python w3t1/compare_layers.py --models \
      gigachat:GigaChat,gigachat:GigaChat-Max,veai:openai/gpt-4o-mini,veai:openai/gpt-4o

Ключи берутся из .env: GigaChat — GIGACHAT_CREDENTIALS, Veai —
VEAI_CREDENTIALS. Список моделей Veai можно вывести так:
    python w3t1/compare_layers.py --list-veai
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main
from gigachat_client import GigaChatClient
from veai_client import VeaiClient


@dataclass(frozen=True)
class Example:
    number: int
    expected: str
    text: str


EXAMPLES = [
    Example(1, "short", "Давай обсудим, как лучше организовать запуск нового проекта."),
    Example(2, "short", "Какие риски обычно стоит проверить в самом начале работы?"),
    Example(3, "short", "Спасибо, теперь понятно, продолжим с этого места позже."),
    Example(4, "short", "Расскажи подробнее, как сравнивать варианты решения задачи."),
    Example(5, "short", "Мне нужно обдумать это предложение и вернуться к обсуждению."),
    Example(6, "short", "Какие вопросы стоит задать команде перед следующим обсуждением?"),
    Example(7, "short", "Хорошо, тогда перейдём к следующему пункту плана."),
    Example(8, "short", "Объясни простыми словами, чем отличаются эти два подхода."),
    Example(9, "short", "Я пока не принял решение, давай рассмотрим ещё один вариант."),
    Example(10, "short", "Сформулируй краткий вывод по тому, что мы обсудили."),
    Example(11, "work", "Сдать первую версию нужно до 15 июня 2027 года."),
    Example(12, "work", "На этот проект выделили 480000 рублей."),
    Example(13, "work", "В готовом решении нельзя использовать сторонние библиотеки."),
    Example(14, "work", "Встретимся с командой во вторник в 11:30."),
    Example(15, "work", "Запуск запланирован для Санкт-Петербурга."),
    Example(16, "work", "К пятнице нужно подготовить прототип для демонстрации."),
    Example(17, "work", "На первом этапе команда должна проверить три гипотезы."),
    Example(18, "work", "Клиент прислал текущую задачу: обработать 120 заявок."),
    Example(19, "work", "Для этой версии продукта нужен только русский интерфейс."),
    Example(20, "work", "Следующая встреча с заказчиком состоится 4 июля в 10:00."),
    Example(21, "long", "Меня зовут Марина."),
    Example(22, "long", "Я работаю аналитиком и обычно проверяю цифры перед выводами."),
    Example(23, "long", "Мне удобнее получать итоговые отчёты в формате CSV."),
    Example(24, "long", "Предпочитаю краткие ответы без длинных вступлений."),
    Example(25, "long", "Мы решили использовать GigaChat для этого класса задач."),
    Example(26, "long", "Запомни, что я не работаю по воскресеньям."),
    Example(27, "long", "В нашей команде принято сначала показывать прототип, а потом писать документацию."),
    Example(28, "long", "Пользователь предпочитает общаться на русском языке."),
    Example(29, "long", "Для будущих проектов лучше начинать с небольшого пилота."),
    Example(30, "long", "Я живу в Екатеринбурге и часто работаю удалённо."),
]


def parse_specs(value: str) -> list[tuple[str, str]]:
    specs = []
    for raw in value.split(","):
        provider, sep, model = raw.strip().partition(":")
        if not sep or provider not in main.PROVIDERS or not model:
            raise SystemExit(f"Неверная модель {raw!r}; ожидается provider:model")
        specs.append((provider, model))
    return specs


def make_options(provider: str, model: str, root: Path) -> main.Options:
    return main.Options(
        env_path=Path(os.environ.get("W3T1_ENV", "/home/mactep/ai/.env")),
        provider=provider, model=model,
        log_file=root / f"{provider}_{model.replace('/', '_')}.tool.log",
        http_log_file=root / f"{provider}_{model.replace('/', '_')}.http.log",
        history_file=root / "history.json", summary_file=root / "summary.json",
        work_file=root / "work.json", long_file=root / "long.json", use_cache=True,
        extract=True, rules_only=True, sync_short=False,
    )


def create_client(provider: str) -> tuple[Any, str]:
    main.load_env_file(Path(os.environ.get("W3T1_ENV", "/home/mactep/ai/.env")))
    if provider == "veai":
        key = os.environ.get(main.VEAI_CREDENTIALS_ENV)
        if not key:
            raise SystemExit("Для Veai нужен VEAI_CREDENTIALS в .env")
        return VeaiClient(key), ""
    key = os.environ.get(main.GIGACHAT_CREDENTIALS_ENV) or os.environ.get(main.GIGACHAT_FALLBACK_CREDENTIALS_ENV)
    if not key:
        raise SystemExit("Для GigaChat нужен GIGACHAT_CREDENTIALS в .env")
    client = GigaChatClient(credentials=key, scope=main.DEFAULT_GIGACHAT_SCOPE,
                            ca_cert_files=main.default_ca_cert_files(),
                            token_cache_file=main.TOKEN_CACHE_FILE)
    return client, client.get_access_token(True)


def list_veai() -> int:
    main.load_env_file(Path(os.environ.get("W3T1_ENV", "/home/mactep/ai/.env")))
    key = os.environ.get(main.VEAI_CREDENTIALS_ENV)
    if not key:
        print("Для Veai нужен VEAI_CREDENTIALS в .env", file=sys.stderr)
        return 1
    models = VeaiClient(key).list_models()
    for item in models:
        model_id = str(item.get("id") or "")
        if "openai" in model_id.lower():
            print(model_id)
    return 0


def run_model(provider: str, model: str, root: Path) -> dict[str, Any]:
    client, token = create_client(provider)
    options = make_options(provider, model, root)
    results = []
    for example in EXAMPLES:
        history = main.History("layer-check", keep=options.keep_turns,
                               strategy=options.strategy)
        memory = main.Memory()
        answer = main.run_agent(example.text, client, token, model, options.log_file,
                                history, context=(example.number, example.text),
                                include_tools=False, layers=memory.blocks())
        history = history.extended(main.ChatTurn(example.text, answer.answer,
            at=main.datetime.now().astimezone().strftime(main.TIME_FORMAT), usage=answer))
        extracted_runs: list[main.AgentRun] = []
        actual_memory, _ = main._extract(history, memory, options, client, token,
                                         extracted_runs, example.number)
        actual = []
        actual.extend({"layer": "long", "key": f.key, "value": f.value}
                      for f in actual_memory.long.facts.values())
        actual.extend({"layer": "work", "key": f.key, "value": f.value}
                      for f in actual_memory.working.facts.values())
        layers = ",".join(sorted({item["layer"] for item in actual})) or "short"
        results.append({"number": example.number, "expected": example.expected,
                        "actual": layers, "match": layers == example.expected,
                        "text": example.text, "facts": actual,
                        "answer": answer.answer})
        print(f"{provider}:{model} [{example.number:02d}] {example.expected}->{layers}", flush=True)
    return {"provider": provider, "model": model, "results": results}


def main_run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="provider:model,...")
    parser.add_argument("--list-veai", action="store_true")
    args = parser.parse_args()
    if args.list_veai:
        return list_veai()
    if not args.models:
        parser.error("укажите --models provider:model,... или --list-veai")
    output = Path(__file__).resolve().parent.parent / "w3t1_layer_compare.json"
    payload = {"models": []}
    for provider, model in parse_specs(args.models):
        payload["models"].append(run_model(provider, model, output.parent / ".cache"))
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_run())
