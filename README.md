*This project has been created as part of the 42 curriculum by \<login1\>, \<login2\>.*

# call me maybe

## Description

`call me maybe` turns a natural-language request into a structured function
call. Asked *"What is the sum of 40 and 2?"*, it does not answer `42`; it
answers which function to call and with which arguments:

```json
{ "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 } }
```

The point of the project is **constrained decoding**. The model is never asked
to produce JSON and trusted to get it right. Instead the program drives
generation one token at a time and, at every step, masks the model's choices
down to the tokens that keep the answer both syntactically valid and compliant
with the schema in `functions_definition.json`. Anything else is unreachable,
so the output parses every time — even from Qwen3-0.6B, a 0.6B-parameter model.

The program reads a list of prompts and a list of function definitions, and
writes one JSON object per prompt to `data/output/function_calling_results.json`.

## Instructions

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
make install        # uv sync
make run            # uv run python -m src
make lint           # flake8 . and mypy . with the required flags
make lint-strict    # flake8 . and mypy . --strict
make debug          # runs under pdb
make clean          # removes caches
```

The first run downloads Qwen3-0.6B from the Hugging Face hub (about 1.5 GB) and
caches it; later runs start from disk.

## Example usage

Default paths (`data/input/` in, `data/output/` out):

```sh
uv run python -m src
```

Custom paths — each flag is optional and falls back to its default:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Input:

```json
[ { "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat'" } ]
```

Output:

```json
[
  {
    "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat'",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "The cat sat on the mat",
      "regex": "cat",
      "replacement": "dog"
    }
  }
]
```

Failures are reported on stderr and never raise a traceback:

```sh
$ uv run python -m src --input missing.json
Error: File not found: missing.json
```

## Algorithm explanation

Generation is a loop: the model turns the context into a **logit** per token of
the vocabulary, one token is chosen, it is appended to the context, and the loop
repeats. Constrained decoding intervenes at the choice. At every step the
program works out which tokens could still lead to a valid answer and picks the
best-scoring one among those alone. Taking the maximum over the allowed ids is
the same operation as setting every other logit to negative infinity first, and
it avoids building a 151k-element mask per token:

```python
next_token = max(valid_tokens, key=lambda token_id: logits[token_id])
```

The work is split in two stages, because the two questions have different
shapes.

### Stage 1 — which function? (`src/function_name.py`)

Every allowed answer is known in advance, so each one is encoded up front into
the complete token sequence that spells it:

```
{"name": "fn_add_numbers"}  ->  [4913, 606, 788, 330, 8822, 2891, 32964, 9207]
{"name": "fn_greet"}        ->  [4913, 606, 788, 330, 8822, 1889, 3744, 9207]
{"name": "NO_FUNCTION"}     ->  [4913, 606, 788, 330, 8996, 18490, 9207]
```

At each step the valid set is *the next token of every candidate still matching
what has been generated*. The model can only walk down this tree, so it cannot
invent a name, misspell one, or wander off into prose. The chosen name is read
back from the candidate that was completed, not parsed out of text.

Note how much is shared: all three candidates open with the same four tokens,
and the two `fn_` names share a fifth. Wherever the mask leaves exactly one
legal token the step is taken without calling the model at all, which removes a
forward pass each time.

`NO_FUNCTION` is an extra candidate for requests no function can answer.

### Stage 2 — which arguments? (`src/arguments.py`)

The shape of the object is already known from the definition, so the program
**writes the structure itself** and only decodes the values. For
`fn_add_numbers(a: number, b: number)` it emits `{"a":`, decodes a number,
emits `, "b":`, decodes a number, emits `}`. Braces, keys, colons and commas
are never up to the model, which removes a whole class of failure.

Each type has its own decoder.

**Numbers** are a state machine whose states say what may come next:

| state | meaning | allowed next |
|---|---|---|
| `start` | nothing written yet | `" "`, `" -"` |
| `sign` | a digit is owed | `0`-`9` |
| `zero` | value opened with `0` | `.`, separator |
| `digits` | inside the integer part | `0`-`9`, `.`, separator |
| `point` | a decimal digit is owed | `0`-`9` |
| `decimal` | inside the fraction | `0`-`9`, separator |

The `sign` and `point` states exist so a lone `-` or a trailing `3.` can never
be produced; `zero` exists because JSON rejects leading zeros such as `007`.
The mask and the transitions are derived from the same function, so the two can
never drift apart. The separator is used only as a stop signal and is written
by the caller.

**Strings** cannot be enumerated, so the allowed set is computed from the
vocabulary file, as `get_path_to_vocab_file()` invites. A token may sit
unescaped inside a JSON string when it contains no quote, no backslash and no
control byte. In the byte-level BPE alphabet the control bytes `0x00`-`0x1F`
are written as `chr(256)` to `chr(287)`, which makes the test one set
intersection over the vocabulary — done once per run and cached. Separately,
every token *opening* with a quote is collected as a closing signal.

**Booleans** are the two spellings scored against each other.

Finally the generated text is parsed back with `json.loads` and the values are
coerced to their declared types.

## Design decisions

**Structure is written, not generated.** Only values are decoded. This is
simpler and faster than a full JSON grammar, and it makes schema compliance
structural rather than something to check afterwards.

**The function is chosen by the model, not by heuristics.** No keyword matching
or similarity scoring anywhere; the constraint only limits *which* names are
spellable, and the model picks among them from its own logits.

**Masks are derived from the vocabulary, not hardcoded per prompt.** The string
decoder reads `vocab.json`; nothing is tailored to the sample functions, so a
different `functions_definition.json` works unchanged.

**Greedy argmax, no sampling.** Function calling wants the most likely answer,
not a varied one. It also makes runs reproducible: the GPU and CPU runs below
produce byte-identical output.

**`number` is coerced to `float`.** JSON does not distinguish them, and it
matches the output format the subject shows (`2.0`, not `2`).

**Hard caps on every decoder loop** (32 tokens for a number, 64 for a string).
If a cap is reached mid-number a `0` is appended so the JSON still parses. This
bounds the runtime whatever the model does.

**A failed prompt is skipped, not fatal.** It is reported on stderr and the
remaining prompts still produce output.

**Errors share a base class** (`CallMeMaybeError`), so `main()` catches every
expected failure in one place and prints a message instead of a traceback.

## Challenges faced

**The tokenizer does not spell things the way you assume — twice.**
Both real bugs in this project came from building a mask out of the bare
single-character token, without checking how Qwen actually writes that
character in context.

*Strings would not stop.* Allowing only the lone `"` (id 1) to close a string
looked right, but Qwen ends a JSON string with a *merged* token — `",`, `"}`,
`":`. Since every token containing a quote had been excluded from the content
set, the only exit was a spelling the model never uses, so it kept generating
until the 64-token cap. A `regex` argument came out as
`"34|233|233|34|233|34|..."`. The fix was to treat *any* token opening with a
quote as the request to close, and write the canonical quote from the caller.

*Negative numbers were impossible.* `{"a": ` tokenizes as `{"`, `a`, `":`, `Ġ`
— the trailing space is its own token. But Qwen writes a negative value after
`":` as the merged token `Ġ-` (`" -"`, id 481), and a bare `-` (id 12) never
follows a space. Dumping the logits after the prefix showed exactly that:

```
  17  '2'    32.219   <- picked
 481  'Ġ-'   22.234   <- what the model wanted, not in the mask
  12  '-'     2.000   <- the only minus offered
```

`-2` therefore decoded as `2`. The fix was to stop the key prefix at `":` and
let the number decoder emit its own opening token, choosing between `" "` and
`" -"`. That turns the sign into an explicit constrained decision. (`" -"` is a
single token but `" 0"` through `" 9"` are not, which is why positives worked
and negatives did not.)

The lesson, and the habit worth keeping: when the model "gets it wrong", print
the top unconstrained logits at that step before blaming the model. Both bugs
were obvious within a minute of doing so.

**Mask and state machine drifting apart.** An earlier version of the number
decoder built the allowed set in one place and the transitions in another. They
disagreed, and `{"a": -}` and `{"a": 3.}` were both reachable. Deriving both
from one function removed the whole class of bug.

**`uv sync` was not enough on a CUDA machine.** The provided SDK takes a
`device_map="auto"` path when CUDA is present, which transformers refuses to
run without `accelerate`. It is now declared as a dependency so a plain
`uv sync` works on GPU and CPU machines alike.


## Resources

- [JSON specification](https://www.json.org/json-en.html) — the grammar the decoders enforce
- [Qwen3 chat template deep dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive) — ChatML layout and the empty think block for non-thinking mode
- [Pydantic validation docs](https://docs.pydantic.dev/latest/) — model validation of the input and output files
- [Hugging Face tokenizers summary](https://huggingface.co/docs/transformers/tokenizer_summary) — byte-level BPE, and why `Ġ` prefixes matter
- [GPT-2 `bytes_to_unicode`](https://github.com/openai/gpt-2/blob/master/src/encoder.py) — the byte-to-character mapping behind the `chr(256)`-`chr(287)` control-byte range
- [Outlines: efficient guided generation](https://arxiv.org/abs/2307.09702) — background reading on constrained decoding (not used; the project implements its own)

### Use of AI

> Adjust this section to cover your own use of AI before submitting.

AI (Claude) was used as a reviewing and debugging partner, not as a code
generator left unchecked. Specifically:

- **Reviewing the implementation against the subject**, which surfaced the
  missing string decoder, the unhandled exceptions in `main()`, and the lint
  rules that did not match Chapter IV.
- **Diagnosing the two tokenizer bugs** described above. The method — dumping
  the top unconstrained logits at the failing step — is what identified both.
- **Hardening the number state machine** and writing the exhaustive check that
  proves no invalid JSON number is reachable.
- **Docstrings and this README.**

Every change was run and verified: the program executes end to end, the output
is re-validated against the schema, and both linters pass.
