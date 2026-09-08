"""Constrained decoding of a function's arguments as a JSON object.

Every value is produced token by token with the candidate set masked down to
the tokens that keep the output valid JSON and compliant with the schema
declared in functions_definition.json.

The ids below are Qwen's byte-level BPE ids. Two tokenizer details shape the
decoders: the space before a value is merged into the token that follows it
(`": -2` is `":` + `Ġ-` + `2`, never a bare `-`), and a string is normally
closed by a merged token such as `",` or `"}` rather than a lone quote.
"""

import json
from typing import Any

from llm_sdk import Small_LLM_Model

from .errors import GenerationError, VocabError
from .models import FunctionDefinition
from .prompt import build_argument_prompt


DOT_TOKEN = 13
ZERO_TOKEN = 15
SPACE_TOKEN = 220
SPACE_MINUS_TOKEN = 481

MAX_NUMBER_TOKENS = 32
MAX_STRING_TOKENS = 64

FORBIDDEN_STRING_CHARS = set('"\\') | {chr(c) for c in range(256, 288)}

_string_tokens: tuple[set[int], set[int]] | None = None


def get_number_digit_tokens() -> set[int]:
    """List the tokens that carry a single digit.

    Returns:
        The ids of the tokens "0" through "9".
    """
    return set(range(15, 25))


def get_number_end_tokens(is_last_parameter: bool) -> set[int]:
    """List the tokens that may follow a finished number.

    Args:
        is_last_parameter: True if this is the object's last argument.

    Returns:
        The id of "}" for the last argument, otherwise the id of ",".
    """
    if is_last_parameter:
        return {92}
    return {11}


def get_number_valid_tokens(state: str, is_last_parameter: bool) -> set[int]:
    """List the tokens that keep a JSON number valid from a given state.

    Args:
        state: Current state of the number machine.
        is_last_parameter: True if this is the object's last argument.

    Returns:
        The ids the decoder is allowed to choose from.

    Raises:
        GenerationError: If the state is not one the machine defines.
    """
    digits = get_number_digit_tokens()
    ends = get_number_end_tokens(is_last_parameter)
    if state == "start":
        return {SPACE_TOKEN, SPACE_MINUS_TOKEN}
    if state in {"sign", "point"}:
        return digits
    if state == "zero":
        return {DOT_TOKEN} | ends
    if state == "digits":
        return digits | {DOT_TOKEN} | ends
    if state == "decimal":
        return digits | ends
    raise GenerationError(f"Invalid number state: {state}")


def get_next_number_state(state: str, token_id: int) -> str:
    """Advance the number machine with the token that was just emitted.

    The "sign" and "point" states still owe a digit, and "zero" exists
    because JSON rejects the leading zeros of a value such as 007.

    Args:
        state: Current state of the number machine.
        token_id: Id of the token appended to the number.

    Returns:
        The state the machine moves to.
    """
    if state == "start":
        return "sign"
    if token_id == DOT_TOKEN:
        return "point"
    if state in {"point", "decimal"}:
        return "decimal"
    if token_id == ZERO_TOKEN and state == "sign":
        return "zero"
    return "digits"


def get_string_tokens(
    model: Small_LLM_Model,
) -> tuple[set[int], set[int]]:
    """Split the vocabulary into string content and string closing tokens.

    A token can sit unescaped inside a JSON string when it holds no quote,
    no backslash and no control byte; the byte-level alphabet writes the
    control bytes 0x00-0x1F as chr(256) to chr(287). Any token opening with
    a quote is read as the model asking to close the string. The result is
    cached, so the vocabulary file is only read once per run.

    Args:
        model: SDK wrapper used to locate the vocabulary file.

    Returns:
        The content token ids and the closing token ids.

    Raises:
        VocabError: If the vocabulary file cannot be read or parsed.
    """
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
    """Map a declared parameter type to the decoder that can generate it.

    Args:
        parameter_type: Type named in functions_definition.json.

    Returns:
        The name of the matching decoder.

    Raises:
        GenerationError: If no decoder handles that type.
    """
    if parameter_type == "number":
        return "number"
    if parameter_type == "string":
        return "string"
    if parameter_type == "boolean":
        return "boolean"
    raise GenerationError(f"Unsupported parameter type: {parameter_type}")


def emit(
    model: Small_LLM_Model,
    input_ids: list[int],
    generated_ids: list[int],
    text: str,
) -> None:
    """Append fixed text to both the model context and the output.

    Args:
        model: SDK wrapper used to encode the text.
        input_ids: Context the model reads from; extended in place.
        generated_ids: Output built so far; extended in place.
        text: Literal text to append, such as a key or a brace.
    """
    ids = model.encode(text).squeeze(0).tolist()
    input_ids.extend(ids)
    generated_ids.extend(ids)


def select_token(logits: list[float], valid_tokens: set[int]) -> int:
    """Pick the best token the mask still allows.

    Reading the maximum over the allowed ids alone is the same as setting
    every other logit to negative infinity before the argmax.

    Args:
        logits: Score the model gave each token of the vocabulary.
        valid_tokens: Ids that would keep the output valid.

    Returns:
        The id of the highest scoring allowed token.
    """
    return max(valid_tokens, key=lambda token_id: logits[token_id])


def generate_number(
    model: Small_LLM_Model, input_ids: list[int], is_last_parameter: bool
) -> list[int]:
    """Decode one JSON number, stopping just before its separator.

    The separator is used only as a stop signal and is not emitted; the
    caller writes it. If the cap is reached while a digit is still owed, a
    zero is appended so the number stays parseable.

    Args:
        model: SDK wrapper used to score each step.
        input_ids: Context the model reads from; extended in place.
        is_last_parameter: True if this is the object's last argument.

    Returns:
        The ids that spell the number.
    """
    generated_ids: list[int] = []
    state = "start"
    end_tokens = get_number_end_tokens(is_last_parameter)
    while len(generated_ids) < MAX_NUMBER_TOKENS:
        valid_tokens = get_number_valid_tokens(state, is_last_parameter)
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = select_token(logits, valid_tokens)
        if next_token in end_tokens:
            break
        generated_ids.append(next_token)
        input_ids.append(next_token)
        state = get_next_number_state(state, next_token)
    if state in {"start", "sign", "point"}:
        generated_ids.append(ZERO_TOKEN)
        input_ids.append(ZERO_TOKEN)
    return generated_ids


def generate_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    content_tokens: set[int],
    closing_tokens: set[int],
) -> list[int]:
    """Decode the body of a JSON string, stopping at the closing quote.

    The closing token is a stop signal only; the caller writes the quote so
    that the result always carries the plain one.

    Args:
        model: SDK wrapper used to score each step.
        input_ids: Context the model reads from; extended in place.
        content_tokens: Ids allowed inside a string.
        closing_tokens: Ids that mean the string should end.

    Returns:
        The ids that spell the string, without its quotes.
    """
    generated_ids: list[int] = []
    valid_tokens = content_tokens | closing_tokens
    while len(generated_ids) < MAX_STRING_TOKENS:
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = select_token(logits, valid_tokens)
        if next_token in closing_tokens:
            break
        generated_ids.append(next_token)
        input_ids.append(next_token)
    return generated_ids


def generate_boolean(
    model: Small_LLM_Model, input_ids: list[int]
) -> list[int]:
    """Decode a JSON boolean by scoring both spellings against each other.

    Args:
        model: SDK wrapper used to score the choice.
        input_ids: Context the model reads from; extended in place.

    Returns:
        The ids spelling either true or false.
    """
    candidates = [
        model.encode(" true").squeeze(0).tolist(),
        model.encode(" false").squeeze(0).tolist(),
    ]
    logits = model.get_logits_from_input_ids(input_ids)
    chosen: list[int] = max(candidates, key=lambda ids: logits[ids[0]])
    input_ids.extend(chosen)
    return chosen


def parse_arguments(
    text: str, function: FunctionDefinition
) -> dict[str, Any]:
    """Turn the generated JSON text into typed argument values.

    Numbers are cast to float so the output matches the declared type
    rather than whatever Python inferred while parsing.

    Args:
        text: JSON object produced by the decoders.
        function: Definition the arguments belong to.

    Returns:
        The argument names mapped to their values.

    Raises:
        GenerationError: If the text is not a JSON object.
    """
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


def generate_arguments(
    model: Small_LLM_Model, function: FunctionDefinition, request: str
) -> dict[str, Any]:
    """Extract one function's arguments from a request.

    Keys, separators and braces are written directly; only the values are
    decoded. Each key prefix stops at `":` so the value decoder owns the
    space token and can open a number with either " " or " -".

    Args:
        model: SDK wrapper used to encode and score.
        function: Definition whose arguments are wanted.
        request: Natural-language prompt to read the values from.

    Returns:
        The argument names mapped to their values, empty if the function
        takes none.
    """
    prompt = build_argument_prompt(function, request)
    input_ids = model.encode(prompt).squeeze(0).tolist()
    parameters = list(function.parameters.items())
    if not parameters:
        return {}
    generated_ids: list[int] = []
    for index, (parameter_name, parameter_spec) in enumerate(parameters):
        is_last = index == len(parameters) - 1
        decoder = get_parameter_decoder(parameter_spec.type)
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
