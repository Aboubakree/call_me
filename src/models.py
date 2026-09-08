"""Pydantic models validating the input and output files."""

from typing import Any

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """One declared argument of a function.

    Attributes:
        type: JSON type of the argument, such as number or string.
        description: Optional explanation of what the argument means.
    """

    type: str
    description: str | None = None


class ReturnSpec(BaseModel):
    """The declared return value of a function.

    Attributes:
        type: JSON type the function returns.
    """

    type: str


class FunctionDefinition(BaseModel):
    """One entry of functions_definition.json.

    Attributes:
        name: Name used to call the function.
        description: What the function does, shown to the model.
        parameters: Declared arguments, keyed by name.
        returns: Declared return value.
    """

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec


class PromptEntry(BaseModel):
    """One entry of function_calling_tests.json.

    Attributes:
        prompt: Natural-language request to answer.
    """

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the generated results file.

    Attributes:
        prompt: Request this call answers.
        name: Function chosen for it.
        parameters: Arguments extracted from the request.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
