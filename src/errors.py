class CallMeMaybeError(Exception):
    pass


class InputError(CallMeMaybeError):
    pass


class VocabError(CallMeMaybeError):
    pass


class GenerationError(CallMeMaybeError):
    pass


class OutputError(CallMeMaybeError):
    pass
