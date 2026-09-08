"""Writing the generated calls to the results file."""

import json
from pathlib import Path

from .errors import OutputError
from .models import FunctionCallResult


def write_results(
    results: list[FunctionCallResult], output_path: str
) -> None:
    """Write the results as a JSON array, creating the directory if needed.

    Args:
        results: Calls generated for the prompts.
        output_path: Path of the file to write.

    Raises:
        OutputError: If the directory or the file cannot be written.
    """
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(
            f"Could not create output directory for {path}: {exc}"
        ) from exc
    data = []
    for result in results:
        data.append(result.model_dump())
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise OutputError(
            f"Could not write output file {path}: {exc}"
        ) from exc
