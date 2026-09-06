from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition
from .prompt import build_prompt


def get_function_names(functions: list[FunctionDefinition]) -> list[str]:
    """Return every callable name plus the NO_FUNCTION fallback."""
    names = []
    for function in functions:
        names.append(function.name)
    names.append("NO_FUNCTION")
    return names


def build_function_candidate(
    model: Small_LLM_Model, function_name: str
) -> list[int]:
    """Encode the full JSON envelope naming one function."""
    text = '{"name": "' + function_name + '"}'
    ids: list[int] = model.encode(text).squeeze(0).tolist()  # [MOD] typed
    return ids


def build_function_candidates(
    model: Small_LLM_Model, function_names: list[str]
) -> list[list[int]]:
    """Encode one candidate token sequence per function name."""
    candidates = []
    for function_name in function_names:
        candidate = build_function_candidate(model, function_name)
        candidates.append(candidate)
    return candidates


def get_valid_next_tokens(
    candidates: list[list[int]], generated_ids: list[int]
) -> set[int]:
    """Return the tokens that keep at least one candidate reachable."""
    valid_token_ids: set[int] = set()
    for candidate in candidates:
        prefix_length = len(generated_ids)
        if candidate[:prefix_length] == generated_ids:
            if prefix_length < len(candidate):
                valid_token_ids.add(candidate[prefix_length])
    return valid_token_ids


def select_next_token(logits: list[float], valid_tokens: set[int]) -> int:
    """Return the highest-scoring token among `valid_tokens`."""
    return max(valid_tokens, key=lambda token_id: logits[token_id])


# [NEW] Look a name up in the definitions; None means NO_FUNCTION.
def find_function(
    functions: list[FunctionDefinition], name: str
) -> FunctionDefinition | None:
    """Return the definition called `name`, or None if there is none."""
    for function in functions:
        if function.name == name:
            return function
    return None


# [MOD] Returns the bare function name instead of the raw JSON envelope, and
# skips the forward pass whenever only one token is possible.
def generate_function_name(
    model: Small_LLM_Model, prompt: str, functions: list[FunctionDefinition]
) -> str:
    """Let the LLM choose one function, constrained to the known names."""
    text = build_prompt(functions, prompt)
    input_ids = model.encode(text).squeeze(0).tolist()
    function_names = get_function_names(functions)
    candidates = build_function_candidates(model, function_names)
    generated_ids: list[int] = []
    while True:
        valid_tokens = get_valid_next_tokens(candidates, generated_ids)
        if not valid_tokens:
            break
        if len(valid_tokens) == 1:  # [NEW] forced token, no need to ask
            next_token = valid_tokens.pop()
        else:
            logits = model.get_logits_from_input_ids(input_ids)
            next_token = select_next_token(logits, valid_tokens)
        generated_ids.append(next_token)
        input_ids.append(next_token)
        if generated_ids in candidates:
            break
    return function_names[candidates.index(generated_ids)]
