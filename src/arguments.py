import json
from functools import lru_cache
from typing import Any

from llm_sdk import Small_LLM_Model

from .errors import GenerationError
from .models import FunctionDefinition
from .prompt import build_argument_prompt

# Qwen keeps every single character in the first hundred token ids, so these
# are stable: 1 is '"', 11 is ',', 12 is '-', 13 is '.', 15..24 are '0'..'9',
# and 92 is '}'. Numbers are always split into single digits by this
# tokenizer, so a digit-at-a-time state machine matches it exactly.
MAX_STRING_TOKENS = 40
MAX_NUMBER_TOKENS = 20
MAX_CYCLE_TOKENS = 10


def get_quote_token() -> int:
    return 1


def get_repeated_tail_size(texts: list[str]) -> int:
    """Return the size of a block that has just repeated itself, else 0.

    Greedy decoding has no way out of a loop: if the likeliest continuation
    of a pattern is the pattern again, it repeats until the length cap.  On
    one regex the model wrote '\\d+\\s+' seven times over.

    A block only counts once it is three characters wide, because ordinary
    text repeats single characters all the time: "233" arrives as the tokens
    '2', '3', '3', and cutting there would lose the rest of the value.
    """
    for size in range(1, MAX_CYCLE_TOKENS + 1):
        if len(texts) < 2 * size:
            break
        tail = texts[-size:]
        if tail == texts[-2 * size:-size] and len("".join(tail)) >= 3:
            return size
    return 0


def get_number_start_tokens() -> set[int]:
    return set(range(15, 25)) | {12}


def get_number_digit_tokens() -> set[int]:
    return set(range(15, 25))


def get_zero_token() -> int:
    return 15


def get_nonzero_digit_tokens() -> set[int]:
    return set(range(16, 25))


# States in which the number so far is a complete JSON value and may end.
COMPLETE_NUMBER_STATES = {"zero", "digits", "decimal"}


def get_number_end_tokens(is_last_parameter: bool) -> set[int]:
    if is_last_parameter:
        return {92}
    return {11}


def get_decimal_point_tokens() -> set[int]:
    return {13}


def get_next_number_state(
    state: str, token_id: int, is_last_parameter: bool = False
) -> str:
    end_tokens = get_number_end_tokens(is_last_parameter)
    if state == "start" and token_id == 12:
        # A minus sign alone is not a number, so it gets its own state: the
        # next token must be a digit, and the value cannot end here.
        return "sign"
    if state in {"start", "sign"}:
        if token_id == get_zero_token():
            # JSON forbids leading zeros, so "0" gets its own state where no
            # further digit may follow: 007 would not parse.
            return "zero"
        if token_id in get_nonzero_digit_tokens():
            return "digits"
    if state == "zero":
        if token_id in get_decimal_point_tokens():
            return "point"
        if token_id in end_tokens:
            return "finished"
    if state == "digits":
        if token_id in get_number_digit_tokens():
            return "digits"
        if token_id in get_decimal_point_tokens():
            return "point"
        if token_id in end_tokens:
            return "finished"
    if state == "point":
        # A decimal point must be followed by at least one digit: "2." would
        # not parse either.
        if token_id in get_number_digit_tokens():
            return "decimal"
    if state == "decimal":
        if token_id in get_number_digit_tokens():
            return "decimal"
        if token_id in end_tokens:
            return "finished"
    return "invalid"


def get_number_valid_tokens(state: str, is_last_parameter: bool) -> set[int]:
    """Return the tokens that may legally follow the number so far."""
    end_tokens = get_number_end_tokens(is_last_parameter)
    if state == "start":
        return get_number_start_tokens()
    if state == "sign":
        return get_number_digit_tokens()
    if state == "zero":
        return get_decimal_point_tokens() | end_tokens
    if state == "digits":
        return (
            get_number_digit_tokens()
            | get_decimal_point_tokens()
            | end_tokens
        )
    if state == "point":
        return get_number_digit_tokens()
    if state == "decimal":
        return get_number_digit_tokens() | end_tokens
    raise ValueError(f"Invalid number state: {state}")


def get_parameter_decoder(parameter_type: str) -> str:
    if parameter_type == "number":
        return "number"
    if parameter_type == "string":
        return "string"
    raise ValueError(
        f"Unsupported parameter type: {parameter_type}"
    )


def generate_number(
    model: Small_LLM_Model, input_ids: list[int], is_last_parameter: bool
) -> list[int]:
    generated_ids: list[int] = []
    state = "start"
    end_tokens = get_number_end_tokens(is_last_parameter)
    # Bounded, not `while True`: nothing guarantees the model ever chooses to
    # end the number, and a run of digits would otherwise never stop.
    for _ in range(MAX_NUMBER_TOKENS):
        valid_tokens = get_number_valid_tokens(state, is_last_parameter)
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = max(valid_tokens, key=lambda token_id: logits[token_id])
        if state in COMPLETE_NUMBER_STATES and next_token in end_tokens:
            return generated_ids
        generated_ids.append(next_token)
        input_ids.append(next_token)
        state = get_next_number_state(state, next_token, is_last_parameter)
    if state not in COMPLETE_NUMBER_STATES:
        raise GenerationError(
            "Could not generate a complete number within "
            f"{MAX_NUMBER_TOKENS} tokens"
        )
    return generated_ids


@lru_cache(maxsize=1)
def get_string_tokens(
    model: Small_LLM_Model,
) -> tuple[frozenset[int], frozenset[int]]:
    """Split the vocabulary into tokens usable inside a string value.

    Returns the tokens that may be chosen at all, and the subset of those
    that close the string.  A closing token is any token *starting* with a
    quote, because this tokenizer merges the closing quote with whatever
    follows it: the model reaches for '",' or '"}', never a bare '"'.

    Backslashes are deliberately allowed.  Excluding them would make every
    regex escape, '\\d' included, impossible to produce.
    """
    path = model.get_path_to_vocab_file()
    with open(path, encoding="utf-8") as handle:
        vocab = json.load(handle)
    valid: set[int] = set()
    closing: set[int] = set()
    for token_id in vocab.values():
        text = model.decode([token_id])
        if not text or not text.isprintable():
            continue
        if text.startswith('"'):
            closing.add(token_id)
            valid.add(token_id)
        elif '"' not in text:
            valid.add(token_id)
    return frozenset(valid), frozenset(closing)


def generate_string(model: Small_LLM_Model, input_ids: list[int]) -> list[int]:
    """Generate one quoted string value, including both of its quotes."""
    valid_tokens, closing_tokens = get_string_tokens(model)
    quote = get_quote_token()
    generated_ids = [quote]
    input_ids.append(quote)
    texts: list[str] = []
    for _ in range(MAX_STRING_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)
        next_token = max(valid_tokens, key=lambda token_id: logits[token_id])
        if next_token in closing_tokens:
            break
        generated_ids.append(next_token)
        input_ids.append(next_token)
        texts.append(model.decode([next_token]))
        cycle = get_repeated_tail_size(texts)
        if cycle:
            # Keep the first occurrence and drop the repeat. One token was
            # appended per text, so the tails line up.
            del generated_ids[-cycle:]
            del input_ids[-cycle:]
            break
    # Close it ourselves: only the quote is kept, never the characters the
    # tokenizer merged onto it, so the enclosing template stays in control.
    generated_ids.append(quote)
    input_ids.append(quote)
    return generated_ids


def generate_arguments(
    model: Small_LLM_Model, function: FunctionDefinition, request: str
) -> str:
    parameters = list(function.parameters.items())
    if not parameters:
        # Nothing to generate, but the value still has to be an object.
        return "{}"
    prompt = build_argument_prompt(function, request)
    input_ids = [int(x) for x in model.encode(prompt).squeeze(0).tolist()]
    # Show the model that it is building a call to this function, not
    # answering the request. Without this it sees a bare '{"s": ' and helpfully
    # supplies the answer: 'Reverse the string hello' yields s="olleh".
    # This text is context only, so it is not added to generated_ids.
    opening = '{"name": "' + function.name + '", "parameters": '
    input_ids.extend(
        int(x) for x in model.encode(opening).squeeze(0).tolist()
    )
    generated_ids: list[int] = []
    for index, (parameter_name, parameter_spec) in enumerate(parameters):
        is_first = index == 0
        is_last = index == len(parameters) - 1
        if is_first:
            prefix = '{"' + parameter_name + '": '
        else:
            prefix = ', "' + parameter_name + '": '
        prefix_ids = [int(x) for x in model.encode(prefix).squeeze(0).tolist()]
        generated_ids.extend(prefix_ids)
        input_ids.extend(prefix_ids)
        decoder_type = get_parameter_decoder(parameter_spec.type)
        if decoder_type == "number":
            number_ids = generate_number(
                model, input_ids, is_last_parameter=is_last
            )
            generated_ids.extend(number_ids)
        if decoder_type == "string":
            # A string closes with its own quote; the comma that follows is
            # supplied by the next parameter's prefix, exactly as for numbers.
            string_ids = generate_string(model, input_ids)
            generated_ids.extend(string_ids)
    closing_ids = [int(x) for x in model.encode("}").squeeze(0).tolist()]
    generated_ids.extend(closing_ids)
    return str(model.decode(generated_ids))


def extract_arguments(raw_snippet: str) -> dict[str, Any]:
    """Parse the object produced by generate_arguments."""
    try:
        data = json.loads(raw_snippet)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"Generated arguments are not valid JSON: {raw_snippet!r}"
        ) from exc
    if not isinstance(data, dict):
        raise GenerationError(
            f"Generated arguments are not a JSON object: {raw_snippet!r}"
        )
    return data
