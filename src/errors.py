class CallMeMaybeError(Exception):
    """Base class for every error this project raises."""


class InputError(CallMeMaybeError):
    """An input file is missing, unreadable, or malformed."""


class VocabError(CallMeMaybeError):
    """The tokenizer vocabulary could not be loaded."""


class GenerationError(CallMeMaybeError):
    """The model or the decoder could not produce a valid call."""


class OutputError(CallMeMaybeError):
    """The result file could not be written."""
