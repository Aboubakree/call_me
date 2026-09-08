"""Constrained selection of the function a request is asking for.

The choice is left to the LLM: every known name is turned into the full
token sequence that spells it, and generation is masked so that only tokens
keeping at least one of those sequences reachable can be picked. The model
therefore cannot name a function that does not exist.
"""

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition
from .prompt import build_prompt


def get_function_names(functions: list[FunctionDefinition]) -> list[str]:
    """List every name the model is allowed to answer with.

    Args:
        functions: Definitions loaded from functions_definition.json.

    Returns:
        The declared names plus the NO_FUNCTION fallback.
    """
    names = []
    for function in functions:
        names.append(function.name)
    names.append("NO_FUNCTION")
    return names


def build_function_candidate(
    model: Small_LLM_Model, function_name: str
) -> list[int]:
    """Encode the JSON answer naming one function.

    Args:
        model: SDK wrapper used to encode the text.
        function_name: Name to wrap.

    Returns:
        The ids spelling the complete answer for that name.
    """
    text = '{"name": "' + function_name + '"}'
    ids: list[int] = model.encode(text).squeeze(0).tolist()
    return ids


def build_function_candidates(
    model: Small_LLM_Model, function_names: list[str]
) -> list[list[int]]:
    """Encode one candidate answer per name.

    Args:
        model: SDK wrapper used to encode the text.
        function_names: Names the model may choose between.

    Returns:
        One token sequence per name, in the same order.
    """
    candidates = []
    for function_name in function_names:
        candidate = build_function_candidate(model, function_name)
        candidates.append(candidate)
    return candidates


def get_valid_next_tokens(
    candidates: list[list[int]], generated_ids: list[int]
) -> set[int]:
    """List the tokens that keep at least one candidate reachable.

    Args:
        candidates: Token sequences the answer may still become.
        generated_ids: Tokens chosen so far.

    Returns:
        The ids that continue a candidate, empty once one is complete.
    """
    valid_token_ids: set[int] = set()
    for candidate in candidates:
        prefix_length = len(generated_ids)
        if candidate[:prefix_length] == generated_ids:
            if prefix_length < len(candidate):
                valid_token_ids.add(candidate[prefix_length])
    return valid_token_ids


def select_next_token(logits: list[float], valid_tokens: set[int]) -> int:
    """Pick the best token the mask still allows.

    Args:
        logits: Score the model gave each token of the vocabulary.
        valid_tokens: Ids that would keep a candidate reachable.

    Returns:
        The id of the highest scoring allowed token.
    """
    return max(valid_tokens, key=lambda token_id: logits[token_id])


def find_function(
    functions: list[FunctionDefinition], name: str
) -> FunctionDefinition | None:
    """Look a chosen name up among the definitions.

    Args:
        functions: Definitions loaded from functions_definition.json.
        name: Name the model picked.

    Returns:
        The matching definition, or None for NO_FUNCTION.
    """
    for function in functions:
        if function.name == name:
            return function
    return None


def generate_function_name(
    model: Small_LLM_Model, prompt: str, functions: list[FunctionDefinition]
) -> str:
    """Let the model choose which function answers a request.

    Steps where the mask leaves a single option are taken without asking
    the model, which skips a forward pass over the shared prefix.

    Args:
        model: SDK wrapper used to encode and score.
        prompt: Natural-language request to answer.
        functions: Definitions the model may choose between.

    Returns:
        The chosen function name, or NO_FUNCTION if none applies.
    """
    text = build_prompt(functions, prompt)
    input_ids = model.encode(text).squeeze(0).tolist()
    function_names = get_function_names(functions)
    candidates = build_function_candidates(model, function_names)
    generated_ids: list[int] = []
    while True:
        valid_tokens = get_valid_next_tokens(candidates, generated_ids)
        if not valid_tokens:
            break
        if len(valid_tokens) == 1:
            next_token = valid_tokens.pop()
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            next_token = select_next_token(logits, valid_tokens)
        generated_ids.append(next_token)
        input_ids.append(next_token)
        if generated_ids in candidates:
            break
    return function_names[candidates.index(generated_ids)]
