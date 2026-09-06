from .models import FunctionDefinition


SELECT_INSTRUCTION = (
    "Choose the one function that best answers the request. "
    "If none of the available functions can answer the request, "
    "choose NO_FUNCTION. JSON only."
)
ARGUMENT_INSTRUCTION = (
    "Extract this function's arguments from the request. JSON only."
)
# [MOD] Qwen3 is a hybrid-thinking model: an empty think block is what its
# chat template emits for non-thinking mode, and it stops the model from
# trying to reason before the constrained JSON.
THINK_BLOCK = "<think>\n\n</think>\n\n"


def describe_function(function: FunctionDefinition) -> str:
    """Render one function as a single-line signature for the prompt."""
    signature = ", ".join(
        f"{name}: {spec.type}" for name, spec in function.parameters.items()
    )
    return f"- {function.name}({signature}): {function.description}"


def _wrap(system: str, request: str) -> str:
    """Wrap a system message and a request in Qwen's ChatML format."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_BLOCK}"
    )


def build_prompt(
    functions: list[FunctionDefinition], request: str
) -> str:
    """Build the prompt asking the model to pick one function."""
    catalogue = "\n".join(describe_function(fn) for fn in functions)
    return _wrap(f"{SELECT_INSTRUCTION}\n{catalogue}", request)


def build_argument_prompt(
    function: FunctionDefinition, request: str
) -> str:
    """Build the prompt asking the model to extract the arguments."""
    return _wrap(
        f"{ARGUMENT_INSTRUCTION}\n{describe_function(function)}", request
    )
