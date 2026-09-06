import json
from typing import Any, Type, TypeVar
from pydantic import BaseModel, ValidationError
from .errors import InputError


T = TypeVar("T", bound=BaseModel)


def load_json_file(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except FileNotFoundError as exc:
        raise InputError(f"File not found: {path}") from exc
    except IsADirectoryError as exc:
        raise InputError(f"Path is a directory, not a file: {path}") from exc
    except PermissionError as exc:
        raise InputError(f"Permission denied: {path}") from exc
    except OSError as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"{path} is not valid UTF-8 text: {exc}") from exc
    if not content.strip():
        raise InputError(f"Empty JSON file: {path}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc}") from exc


def parse_models(raw_data: Any, model: Type[T], source: str) -> list[T]:
    if not isinstance(raw_data, list):
        raise InputError(f"Expected a JSON array in {source}")
    result: list[T] = []
    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise InputError(
                f"Item {index} in {source} is a {type(item).__name__}, "
                "expected a JSON object"
            )
        try:
            result.append(model(**item))
        except ValidationError as exc:
            raise InputError(
                f"Validation error in {source}, item {index}: {exc}"
            ) from exc
    return result
