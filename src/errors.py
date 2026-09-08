"""Exceptions raised across the project.

They share a base class so the entry point can catch every expected
failure in one place and report it without a traceback.
"""


class CallMeMaybeError(Exception):
    """Base class for every error this project raises."""


class InputError(CallMeMaybeError):
    """An input file is missing, unreadable or malformed."""


class VocabError(CallMeMaybeError):
    """The tokenizer vocabulary could not be loaded."""


class GenerationError(CallMeMaybeError):
    """The model or a decoder could not produce a valid call."""


class OutputError(CallMeMaybeError):
    """The results file could not be written."""
