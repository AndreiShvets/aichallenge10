"""Авторизуется в GigaChat API, выводит список моделей и остаток токенов."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from gigachat_client import (
    TOKEN_CACHE_FILE,
    GigaChatClient,
    GigaChatError,
    default_ca_cert_files,
)

CREDENTIALS_ENV = "GIGACHAT_CREDENTIALS"
FALLBACK_CREDENTIALS_ENV = "GIGACHAT_API_KEY"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
PROJECT_DIR = Path(__file__).resolve().parent
# Ключ не хранится в репозитории, поэтому ищем .env от папки задачи вверх по дереву.
ENV_SEARCH_START = PROJECT_DIR


def find_env_file(start: Path) -> Path:
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise GigaChatError(f"Файл .env не найден ни в {start}, ни в его родительских папках.")


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
                raise GigaChatError("После --env требуется путь к файлу .env.")
            env_path = Path(args.pop(0)).expanduser()
        elif arg.startswith("--env="):
            env_path = Path(arg[len("--env=") :]).expanduser()
        else:
            raise GigaChatError(
                f"Неизвестный аргумент: {arg}. Доступны: --env PATH, --no-cache."
            )
    return env_path or find_env_file(ENV_SEARCH_START), use_cache


def load_env_file(path: Path) -> None:
    """Читает .env в os.environ, не перетирая уже заданные переменные."""
    if not path.is_file():
        raise GigaChatError(f"Файл с ключом не найден: {path}")
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


def print_models(models: list[dict]) -> None:
    rows = [
        (str(m.get("id", "")), str(m.get("type", "")), str(m.get("owned_by", "")))
        for m in models
    ]
    print_table(("ID", "Тип", "Владелец"), rows)


def print_balance(balance: list[dict]) -> None:
    rows = [
        (str(item.get("usage", "")), f"{int(item.get('value') or 0):,}".replace(",", " "))
        for item in balance
    ]
    print_table(("Модель", "Остаток токенов"), rows)


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и портит кириллицу.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    try:
        env_file, use_cache = parse_args(sys.argv[1:])
        load_env_file(env_file)
        credentials = os.environ.get(CREDENTIALS_ENV) or os.environ.get(FALLBACK_CREDENTIALS_ENV)
        if not credentials:
            raise GigaChatError(
                f"В {env_file} не найден ключ {CREDENTIALS_ENV} (или {FALLBACK_CREDENTIALS_ENV})."
            )

        scope = os.environ.get("GIGACHAT_SCOPE", "").strip() or DEFAULT_SCOPE
        client = GigaChatClient(
            credentials=credentials,
            scope=scope,
            ca_cert_files=default_ca_cert_files(),
            token_cache_file=TOKEN_CACHE_FILE,
        )

        print(f"Ключ: {env_file}")
        print(f"Авторизация в GigaChat API (scope {scope})...")
        token = client.get_access_token(use_cache=use_cache)
        print(f"Токен получен ({client.token_source}): {token[:16]}...\n")

        models = client.list_models(token)
        balance = client.get_balance(token)
    except GigaChatError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"Доступно моделей: {len(models)}\n")
    print_models(models)

    print("\nОстаток токенов:")
    if balance:
        print_balance(balance)
    else:
        print("  API вернул пустой баланс.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
