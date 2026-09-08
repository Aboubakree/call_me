"""Building the prompts sent to the model.

Both prompts follow Qwen's ChatML layout and end on an empty think block,
which is what the model's own chat template emits for non-thinking mode.
They only steer the model; correctness of the output comes from the
constrained decoders, never from the wording here.
"""

from .models import FunctionDefinition


SELECT_INSTRUCTION = (
    "Choose the one function that best answers the request. "
    "If none of the available functions can answer the request, "
    "choose NO_FUNCTION. JSON only."
)
ARGUMENT_INSTRUCTION = (
    "Extract this function's arguments from the request. JSON only."
)
THINK_BLOCK = "<think>\n\n</think>\n\n"


def describe_function(function: FunctionDefinition) -> str:
    """Render a function as a one-line signature for the prompt.

    Args:
        function: Definition to describe.

    Returns:
        A line holding the name, the typed arguments and the description.
    """
    signature = ", ".join(
        f"{name}: {spec.type}" for name, spec in function.parameters.items()
    )
    return f"- {function.name}({signature}): {function.description}"


def _wrap(system: str, request: str) -> str:
    """Lay a system message and a request out in ChatML.

    Args:
        system: Instructions the model should follow.
        request: Natural-language request to answer.

    Returns:
        The prompt text, ending where the answer begins.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_BLOCK}"
    )


def build_prompt(
    functions: list[FunctionDefinition], request: str
) -> str:
    """Build the prompt asking the model to choose a function.

    Args:
        functions: Definitions to offer as a catalogue.
        request: Natural-language request to answer.

    Returns:
        The prompt text.
    """
    catalogue = "\n".join(describe_function(fn) for fn in functions)
    return _wrap(f"{SELECT_INSTRUCTION}\n{catalogue}", request)


def build_argument_prompt(
    function: FunctionDefinition, request: str
) -> str:
    """Build the prompt asking the model to extract the arguments.

    Args:
        function: Definition whose arguments are wanted.
        request: Natural-language request to read the values from.

    Returns:
        The prompt text.
    """
    return _wrap(
        f"{ARGUMENT_INSTRUCTION}\n{describe_function(function)}", request
    )
