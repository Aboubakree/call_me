import sys

from llm_sdk import Small_LLM_Model

from .arguments import generate_arguments
from .errors import CallMeMaybeError, GenerationError, InputError
from .function_name import find_function, generate_function_name
from .json_loader import load_json_file, parse_models
from .models import FunctionCallResult, FunctionDefinition, PromptEntry
from .output_writer import write_results


def parse_args() -> tuple[str, str, str]:
    """Read the optional --functions_definition/--input/--output flags."""
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


# [NEW] Loading pulls the model from the hub, so failures are turned into a
# clear message instead of a traceback.
def load_model() -> Small_LLM_Model:
    """Load the LLM, reporting any failure as a project error."""
    try:
        return Small_LLM_Model()
    except Exception as exc:
        raise GenerationError(f"Could not load the model: {exc}") from exc


# [NEW] The generation loop; previously main() built an empty result list.
def run(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    prompts: list[PromptEntry],
) -> list[FunctionCallResult]:
    """Turn every prompt into one function call."""
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


# [MOD] Everything now runs inside one handler, so a bad input file or a
# failed download prints a message and exits 1 instead of crashing.
def main() -> None:
    """Load the inputs, generate every call, and write the results."""
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
