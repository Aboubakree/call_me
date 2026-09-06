from typing import Any
from pydantic import BaseModel


class ParameterSpec(BaseModel):
    """One declared argument of a function."""

    type: str
    description: str | None = None


class ReturnSpec(BaseModel):
    """The declared return type of a function."""

    type: str


class FunctionDefinition(BaseModel):
    """One entry of functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec


class PromptEntry(BaseModel):
    """One entry of function_calling_tests.json."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the generated results file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
