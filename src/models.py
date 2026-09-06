from typing import Any
from pydantic import BaseModel


class ParameterSpec(BaseModel):
    type: str
    description: str | None = None


class ReturnSpec(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec


class PromptEntry(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
