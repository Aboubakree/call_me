import json
from typing import Any

from llm_sdk import Small_LLM_Model

from .errors import GenerationError, VocabError
from .models import FunctionDefinition
from .prompt import build_argument_prompt


DOT_TOKEN = 13
ZERO_TOKEN = 15

# [NEW] Qwen merges the space before a value into the opening token: `": -2`
# is `":` + `Ġ-` + `2`, and a bare `-` never follows a space. So the key
# prefix stops at `":` and the number decoder picks its own opening token,
# which makes the sign an explicit choice instead of an unreachable one.
SPACE_TOKEN = 220
SPACE_MINUS_TOKEN = 481

# [NEW] Hard caps so a stubborn model can never spin forever.
MAX_NUMBER_TOKENS = 32
MAX_STRING_TOKENS = 64

# [NEW] A token may not sit raw inside a JSON string if it contains a quote,
# a backslash, or a control byte (0x00-0x1F). The byte-level BPE alphabet
# writes those control bytes as chr(256)..chr(287).
FORBIDDEN_STRING_CHARS = set('"\\') | {chr(c) for c in range(256, 288)}

# [NEW] The vocabulary is read once and reused for every prompt.
_string_tokens: tuple[set[int], set[int]] | None = None


def get_number_digit_tokens() -> set[int]:
    """Return the token ids of the digits 0-9."""
    return set(range(15, 25))


def get_number_end_tokens(is_last_parameter: bool) -> set[int]:
    """Return the token that must follow a finished number."""
    if is_last_parameter:
        return {92}
    return {11}


# [NEW] One source of truth for the mask. Previously the mask and the state
# machine were written separately and disagreed, which let the decoder emit
# a bare "-" or a trailing "3." -- both invalid JSON.
def get_number_valid_tokens(state: str, is_last_parameter: bool) -> set[int]:
    """Return the token ids that keep a JSON number valid from `state`."""
    digits = get_number_digit_tokens()
    ends = get_number_end_tokens(is_last_parameter)
    if state == "start":
        return {SPACE_TOKEN, SPACE_MINUS_TOKEN}  # [MOD] was digits | {"-"}
    if state in {"sign", "point"}:
        return digits
    if state == "zero":
        return {DOT_TOKEN} | ends
    if state == "digits":
        return digits | {DOT_TOKEN} | ends
    if state == "decimal":
        return digits | ends
    raise GenerationError(f"Invalid number state: {state}")


# [MOD] "sign" and "point" now demand a digit before the number can end, and
# "zero" forbids the leading zeros that JSON rejects.
def get_next_number_state(state: str, token_id: int) -> str:
    """Advance the JSON-number state machine with the token just emitted."""
    if state == "start":  # [MOD] the opening " " or " -" needs a digit next
        return "sign"
    if token_id == DOT_TOKEN:
        return "point"
    if state in {"point", "decimal"}:
        return "decimal"
    if token_id == ZERO_TOKEN and state == "sign":
        return "zero"
    return "digits"


# [MOD] Also collects the closing tokens. Qwen writes the end of a JSON
# string as a merged token (`",`, `"}`, `":`), so offering only the bare
# quote left the model unable to stop and it repeated until the cap.
def get_string_tokens(
    model: Small_LLM_Model,
) -> tuple[set[int], set[int]]:
    """Return the string content tokens and the tokens that close a string."""
    global _string_tokens
    if _string_tokens is None:
        try:
            path = model.get_path_to_vocab_file()
            with open(path, "r", encoding="utf-8") as handle:
                vocab = json.load(handle)
        except Exception as exc:
            raise VocabError(f"Could not read the vocabulary: {exc}") from exc
        content: set[int] = set()
        closing: set[int] = set()
        for token, token_id in vocab.items():
            if not set(token) & FORBIDDEN_STRING_CHARS:
                content.add(token_id)
            elif token.startswith('"'):
                closing.add(token_id)
        _string_tokens = (content, closing)
    return _string_tokens


def get_parameter_decoder(parameter_type: str) -> str:
    """Map a declared parameter type to the decoder that can generate it."""
    if parameter_type == "number":
        return "number"
    if parameter_type == "string":
        return "string"
    if parameter_type == "boolean":  # [NEW]
        return "boolean"
    raise GenerationError(f"Unsupported parameter type: {parameter_type}")


def emit(
    model: Small_LLM_Model,
    input_ids: list[int],
    generated_ids: list[int],
    text: str,
) -> None:
    """Append literal `text` to both the model context and the output."""
    ids = model.encode(text).squeeze(0).tolist()
    input_ids.extend(ids)
    generated_ids.extend(ids)


def select_token(logits: list[float], valid_tokens: set[int]) -> int:
    """Return the highest-scoring token among `valid_tokens`."""
    return max(valid_tokens, key=lambda token_id: logits[token_id])


def generate_number(
    model: Small_LLM_Model, input_ids: list[int], is_last_parameter: bool
) -> list[int]:
    """Decode one JSON number, stopping just before its separator."""
    generated_ids: list[int] = []
    state = "start"
    end_tokens = get_number_end_tokens(is_last_parameter)
    while len(generated_ids) < MAX_NUMBER_TOKENS:  # [MOD] was `while True`
        valid_tokens = get_number_valid_tokens(state, is_last_parameter)
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = select_token(logits, valid_tokens)
        if next_token in end_tokens:
            break
        generated_ids.append(next_token)
        input_ids.append(next_token)
        state = get_next_number_state(state, next_token)
    if state in {"start", "sign", "point"}:  # [NEW] cap hit mid-number
        generated_ids.append(ZERO_TOKEN)
        input_ids.append(ZERO_TOKEN)
    return generated_ids


# [NEW] The string decoder was missing entirely.
def generate_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    content_tokens: set[int],
    closing_tokens: set[int],
) -> list[int]:
    """Decode the contents of a JSON string, stopping at the closing quote."""
    generated_ids: list[int] = []
    valid_tokens = content_tokens | closing_tokens
    while len(generated_ids) < MAX_STRING_TOKENS:
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = select_token(logits, valid_tokens)
        if next_token in closing_tokens:  # [MOD] the caller writes the quote
            break
        generated_ids.append(next_token)
        input_ids.append(next_token)
    return generated_ids


# [NEW] Booleans are named in the subject's validation rules.
def generate_boolean(
    model: Small_LLM_Model, input_ids: list[int]
) -> list[int]:
    """Decode `true` or `false`, whichever the model scores higher."""
    candidates = [  # [MOD] leading space merged in, as Qwen writes it
        model.encode(" true").squeeze(0).tolist(),
        model.encode(" false").squeeze(0).tolist(),
    ]
    logits = model.get_logits_from_input_ids(input_ids)
    chosen: list[int] = max(candidates, key=lambda ids: logits[ids[0]])
    input_ids.extend(chosen)
    return chosen


# [NEW] The raw JSON text is parsed here so callers get a ready-to-write dict.
def parse_arguments(
    text: str, function: FunctionDefinition
) -> dict[str, Any]:
    """Parse the generated JSON and coerce numbers to float."""
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Generated invalid JSON {text!r}") from exc
    if not isinstance(values, dict):
        raise GenerationError(f"Generated arguments are not JSON: {text!r}")
    for name, spec in function.parameters.items():
        if spec.type == "number" and name in values:
            values[name] = float(values[name])
    return values


# [MOD] Returns a parsed dict instead of a raw string, handles zero-argument
# functions, and dispatches to the string and boolean decoders.
def generate_arguments(
    model: Small_LLM_Model, function: FunctionDefinition, request: str
) -> dict[str, Any]:
    """Constrained-decode this function's arguments from the request."""
    prompt = build_argument_prompt(function, request)
    input_ids = model.encode(prompt).squeeze(0).tolist()
    parameters = list(function.parameters.items())
    if not parameters:  # [MOD] used to return "" instead of {}
        return {}
    generated_ids: list[int] = []
    for index, (parameter_name, parameter_spec) in enumerate(parameters):
        is_last = index == len(parameters) - 1
        decoder = get_parameter_decoder(parameter_spec.type)
        # [MOD] stops at `":` so the value decoder owns the space token
        prefix = ('{"' if index == 0 else ', "') + parameter_name + '":'
        if decoder == "string":
            prefix += ' "'
        emit(model, input_ids, generated_ids, prefix)
        if decoder == "number":
            generated_ids.extend(
                generate_number(model, input_ids, is_last)
            )
        elif decoder == "string":
            content, closing = get_string_tokens(model)
            generated_ids.extend(
                generate_string(model, input_ids, content, closing)
            )
            emit(model, input_ids, generated_ids, '"')
        else:
            generated_ids.extend(generate_boolean(model, input_ids))
    emit(model, input_ids, generated_ids, "}")
    return parse_arguments(model.decode(generated_ids), function)
