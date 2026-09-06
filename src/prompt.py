from .models import FunctionDefinition


SELECT_INSTRUCTION = (
    "Choose the one function that best answers the request. "
    "If none of the available functions can answer the request, "
    "choose NO_FUNCTION. JSON only."
)
ARGUMENT_INSTRUCTION = (
    "Extract this function's arguments from the request. JSON only. "
    "If the request gives no value for an argument, use 0 for a number "
    "and an empty string for a string. Never invent a value."
)
THINK_BLOCK = ""


def describe_function(function: FunctionDefinition) -> str:
    signature = ", ".join(
        f"{name}: {spec.type}" for name, spec in function.parameters.items()
    )
    return f"- {function.name}({signature}): {function.description}"


def _wrap(system: str, request: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_BLOCK}"
    )


def build_prompt(
    functions: list[FunctionDefinition], request: str
) -> str:
    catalogue = "\n".join(describe_function(fn) for fn in functions)
    return _wrap(f"{SELECT_INSTRUCTION}\n{catalogue}", request)


def build_argument_prompt(
    function: FunctionDefinition, request: str
) -> str:
    return _wrap(
        f"{ARGUMENT_INSTRUCTION}\n{describe_function(function)}", request
    )
