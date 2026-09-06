import json

from llm_sdk import Small_LLM_Model

from .errors import GenerationError
from .models import FunctionDefinition
from .prompt import build_prompt


def get_function_names(functions: list[FunctionDefinition]) -> list[str]:
    names = []
    for function in functions:
        names.append(function.name)
    names.append("NO_FUNCTION")
    return names


def build_function_candidate(
    model: Small_LLM_Model, function_name: str
) -> list[int]:
    text = '{"name": "' + function_name + '"}'
    return [int(x) for x in model.encode(text).squeeze(0).tolist()]


def build_function_candidates(
    model: Small_LLM_Model, function_names: list[str]
) -> list[list[int]]:
    candidates = []
    for function_name in function_names:
        candidate = build_function_candidate(model, function_name)
        candidates.append(candidate)
    return candidates


def get_valid_next_tokens(
    candidates: list[list[int]], generated_ids: list[int]
) -> set[int]:
    valid_token_ids: set[int] = set()
    for candidate in candidates:
        prefix_length = len(generated_ids)
        if candidate[:prefix_length] == generated_ids:
            if prefix_length < len(candidate):
                valid_token_ids.add(candidate[prefix_length])
    return valid_token_ids


def select_next_token(logits: list[float], valid_tokens: set[int]) -> int:
    return max(valid_tokens, key=lambda token_id: logits[token_id])


def generate_function_name(
    model: Small_LLM_Model, prompt: str, functions: list[FunctionDefinition]
) -> str:
    prompt_text = build_prompt(functions, prompt)
    input_ids = [int(x) for x in model.encode(prompt_text).squeeze(0).tolist()]
    function_names = get_function_names(functions)
    candidates = build_function_candidates(model, function_names)
    generated_ids: list[int] = []
    while True:
        valid_tokens = get_valid_next_tokens(candidates, generated_ids)
        if not valid_tokens:
            break
        if len(valid_tokens) == 1:
            # Every remaining candidate agrees on the next token, so there is
            # nothing to decide. Skipping the forward pass here removes about
            # 88 of the 99 passes a full run would otherwise make.
            next_token = next(iter(valid_tokens))
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            next_token = select_next_token(logits, valid_tokens)
        generated_ids.append(next_token)
        input_ids.append(next_token)
        if generated_ids in candidates:
            break
    return str(model.decode(generated_ids))


def extract_function_name(raw_snippet: str) -> str:
    """Parse the '{"name": "..."}' snippet from generate_function_name."""
    try:
        data = json.loads(raw_snippet)
        return str(data["name"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GenerationError(
            f"Could not extract a function name from: {raw_snippet!r}"
        ) from exc
