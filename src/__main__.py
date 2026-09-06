import sys

from llm_sdk import Small_LLM_Model

from .arguments import extract_arguments, generate_arguments
from .errors import CallMeMaybeError, GenerationError
from .function_name import extract_function_name, generate_function_name
from .json_loader import load_json_file, parse_models
from .models import FunctionCallResult, FunctionDefinition, PromptEntry
from .output_writer import write_results


def parse_args() -> tuple[str, str, str]:
    functions_definition = "data/input/functions_definition.json"
    input_file = "data/input/function_calling_tests.json"
    output = "data/output/function_calling_results.json"
    known_flags = ["--functions_definition", "--input", "--output"]
    for i, arg in enumerate(sys.argv):
        if arg not in known_flags:
            continue
        if i + 1 >= len(sys.argv):
            raise ValueError(f"Missing value for argument: {arg}")
        if arg == "--functions_definition":
            functions_definition = sys.argv[i + 1]
        elif arg == "--input":
            input_file = sys.argv[i + 1]
        elif arg == "--output":
            output = sys.argv[i + 1]
    return functions_definition, input_file, output


def build_result(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    request: str,
) -> FunctionCallResult:
    """Generate one complete function call for a single request."""
    name = extract_function_name(
        generate_function_name(model, request, functions)
    )
    function = None
    for candidate in functions:
        if candidate.name == name:
            function = candidate
    if function is None:
        raise GenerationError(f"No definition for generated name: {name}")
    parameters = extract_arguments(
        generate_arguments(model, function, request)
    )
    return FunctionCallResult(
        prompt=request, name=name, parameters=parameters
    )


def main() -> None:
    try:
        functions_definition, input_file, output = parse_args()
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    try:
        raw_functions = load_json_file(functions_definition)
        raw_prompts = load_json_file(input_file)
        functions = parse_models(
            raw_functions, FunctionDefinition, functions_definition
        )
        prompts = parse_models(
            raw_prompts, PromptEntry, input_file
        )
        # The evaluation machine has no GPU, and asking the SDK to
        # auto-detect one pulls in `accelerate`, which is not allowed.
        model = Small_LLM_Model(device="cpu")
        results: list[FunctionCallResult] = []
        for entry in prompts:
            try:
                results.append(build_result(model, functions, entry.prompt))
            except GenerationError as exc:
                # One bad prompt must not abort the whole batch.
                print(
                    f"Skipping {entry.prompt!r}: {exc}", file=sys.stderr
                )
        write_results(results, output)
        print(
            f"Wrote {len(results)} of {len(prompts)} results to {output}",
            file=sys.stderr,
        )
    except CallMeMaybeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
