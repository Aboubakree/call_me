"""Entry point: read the input files, generate the calls, write the results."""

import sys

from llm_sdk import Small_LLM_Model

from .arguments import generate_arguments
from .errors import CallMeMaybeError, GenerationError, InputError
from .function_name import find_function, generate_function_name
from .json_loader import load_json_file, parse_models
from .models import FunctionCallResult, FunctionDefinition, PromptEntry
from .output_writer import write_results


def parse_args() -> tuple[str, str, str]:
    """Read the optional path arguments from the command line.

    Returns:
        The functions definition, input and output paths, each falling back
        to its default under data/.

    Raises:
        InputError: If a known flag is given without a value.
    """
    functions_definition = "data/input/functions_definition.json"
    input_file = "data/input/function_calling_tests.json"
    output = "data/output/function_calling_results.json"
    known_flags = ["--functions_definition", "--input", "--output"]
    for i, arg in enumerate(sys.argv):
        if arg not in known_flags:
            continue
        if i + 1 >= len(sys.argv):
            raise InputError(f"Missing value for argument: {arg}")
        if arg == "--functions_definition":
            functions_definition = sys.argv[i + 1]
        elif arg == "--input":
            input_file = sys.argv[i + 1]
        elif arg == "--output":
            output = sys.argv[i + 1]
    return functions_definition, input_file, output


def load_model() -> Small_LLM_Model:
    """Load the LLM, reporting any failure as a project error.

    Returns:
        The ready-to-use SDK wrapper.

    Raises:
        GenerationError: If the model cannot be downloaded or loaded.
    """
    try:
        return Small_LLM_Model()
    except Exception as exc:
        raise GenerationError(f"Could not load the model: {exc}") from exc


def run(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[PromptEntry],
) -> list[FunctionCallResult]:
    """Turn every prompt into one function call.

    A prompt that cannot be answered is reported on the error stream and
    skipped, so one bad case never costs the whole run.

    Args:
        model: SDK wrapper used to encode and score.
        functions: Definitions the model may choose between.
        prompts: Requests to answer.

    Returns:
        One result per prompt that could be answered.
    """
    results: list[FunctionCallResult] = []
    for entry in prompts:
        try:
            name = generate_function_name(model, entry.prompt, functions)
            function = find_function(functions, name)
            parameters = {}
            if function is not None:
                parameters = generate_arguments(model, function, entry.prompt)
        except Exception as exc:
            print(f"Skipping {entry.prompt!r}: {exc}", file=sys.stderr)
            continue
        results.append(
            FunctionCallResult(
                prompt=entry.prompt, name=name, parameters=parameters
            )
        )
    return results


def main() -> None:
    """Run the whole pipeline, reporting any failure without a traceback."""
    try:
        functions_definition, input_file, output = parse_args()
        raw_functions = load_json_file(functions_definition)
        raw_prompts = load_json_file(input_file)
        functions = parse_models(
            raw_functions, FunctionDefinition, functions_definition
        )
        prompts = parse_models(raw_prompts, PromptEntry, input_file)
        model = load_model()
        results = run(model, functions, prompts)
        write_results(results, output)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
