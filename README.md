*This project has been created as part of the 42 curriculum by asmounci.*

# call me maybe

Turn a sentence into a function call.

Given *"What is the sum of 40 and 2?"*, this program does not answer `42`. It
decides which of the available functions should be called, and with which
arguments:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}
```

The model is `Qwen/Qwen3-0.6B`, small enough to run on a laptop CPU. Asked
politely for JSON, a model that size produces something close to JSON *most* of
the time, and you cannot tell which time you got. This program never asks. It
writes the JSON itself and lets the model fill in only the parts that are
genuine decisions.

---

## Instructions

```bash
make install     # uv sync
make run         # uv run python -m src
make lint        # flake8 + mypy
```

Results are written to `data/output/function_calling_results.json`.

Optional flags:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

The first run downloads the model (about 1.2 GB) from Hugging Face.

---

## Resources

- [Qwen3 chat template deep dive](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
  — the `<|im_start|>` / `<|im_end|>` format the prompts are built in.
- [pydantic](https://docs.pydantic.dev/latest/) — validation for every shape
  loaded from disk.
- [JSON specification](https://www.json.org/json-en.html) — the grammar the
  constraints enforce.
- [w3schools JSON](https://www.w3schools.com/js/js_json_intro.asp) — reference
  for the syntax rules.

AI assistance was used as a reviewer rather than an author: to check the
decoding logic against edge cases, to explain how byte-level BPE stores tokens,
and to help diagnose bugs found by running the program.

---

## How constrained decoding works here

The SDK deliberately provides no `generate()`. It offers only:

| Method | Returns |
|--------|---------|
| `encode(text)` | token ids |
| `get_logits_from_input_ids(ids)` | one score per token, for the next position |
| `get_path_to_vocab_file()` | the vocabulary file |

So the generation loop has to be written by hand — and that loop is the only
place a constraint can be applied. At every step the program:

1. works out which tokens would keep the output valid,
2. asks the model for its scores,
3. takes the highest-scoring token **from that set only**.

An invalid function name or malformed JSON is therefore not unlikely. It is
unreachable.

### Most of the JSON is not a decision

```
{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}
 └───┬──┘  └──────┬──────┘  └──────┬────┘  └┬┘  └┬┘  └┬┘ └┬┘
 written       MODEL           written    writ. MODEL  .  MODEL
```

Once the function is known, `, "parameters": {"a": ` is fixed by the schema.
The program writes those characters itself — no forward pass, no possibility of
error. The model is consulted only for the function name and the values.

### Stage 1 — the function name

Each candidate is tokenized in full, as `{"name": "fn_greet"}`, and generation
is constrained to remain a prefix of at least one candidate. Working in
token-id space means the program never has to know what text a token
represents.

All five names begin `{"name": "fn_`, so for most steps only one token is
legal. When that happens no forward pass is made at all — there is nothing to
decide. This removes roughly 88 of the 99 passes a run would otherwise make.

### Stage 2 — the arguments

The context is rebuilt around the chosen function alone, since the other
definitions can no longer matter. Each parameter's key is written by the
program, and only its value is generated:

- **numbers** — a state machine matching JSON's number grammar exactly:
  optional `-`, then either a lone `0` or a non-zero digit followed by more,
  then optionally a `.` with at least one digit after it. The value ends when
  the model reaches for the delimiter, and it may only do so from a state where
  what has been written already parses. `007`, `2.` and a bare `-` are all
  unreachable.
- **strings** — the opening quote is forced, and any token containing a quote
  is excluded so the value cannot end early.

---

## Design decisions

**Write the punctuation, generate only the values.** Both faster and impossible
to get wrong.

**Skip the model when only one token is legal.** No decision, no forward pass.

**Constrain by token ids, not text.** Byte-level BPE stores a leading space as
`Ġ`, so comparing token strings against ordinary text silently matches nothing.
Comparing ids sidesteps the problem entirely. Where text *is* needed — to ask
whether a token contains a quote — `decode()` supplies it.

**Show the model it is building a call.** Generating arguments against a bare
`{"s": ` invites the model to answer the question instead of describing the
call. Prefixing the context with `{"name": "fn_reverse_string", "parameters": `
fixes it: the model can see it is filling in a call, not solving a problem.
This text is context only and never reaches the output.

**Allow backslashes inside strings.** Forbidding them is tempting, since then
nothing needs escaping. It also makes `\d`, `\s` and every other regex escape
impossible to produce. Because the generated text is parsed as JSON, the
model's own escaping is interpreted correctly.

**Stop a value that starts repeating.** Greedy decoding cannot leave a loop. A
block that repeats itself is cut back to its first occurrence — but only once
it is three characters wide, because `233` arrives as `'2','3','3'` and a
narrower rule would truncate good values.

**Ask for 0 when the request gives no value.** Constrained decoding guarantees
every argument is *present* and correctly typed, so a value is always produced
— but for a request like *"What is the sum of ?"* it has to come from
somewhere, and the model invented `{"a": 1, "b": 2}`. The argument instruction
now asks for `0`, or an empty string, when the request supplies nothing. This
is a hint rather than a guarantee, since it cannot be enforced by the grammar,
so it was checked against every other prompt to confirm it changes none of
them.

**Every generation loop is bounded.** Nothing obliges the model to ever choose
to end a value. Both the number and string loops have hard caps, and a number
that is still incomplete when its cap is reached raises rather than emitting
something that would not parse.

---

## Error handling

Every failure produces a message, never a traceback:

| Situation | Behaviour |
|-----------|-----------|
| Input file missing, a directory, empty, or malformed | named error, exit 1 |
| JSON that is not an array of objects | named error, exit 1 |
| An entry missing a required field | pydantic error naming the entry, exit 1 |
| Output directory cannot be created | named error, exit 1 |
| One prompt fails to generate | reported on stderr, the rest continue |

Library modules raise from `src/errors.py`; only `__main__.py` exits.

---

## Testing strategy

The program is checked at two levels.

**Structural, and automatic.** After a run the output is verified against the
input: one result per prompt, exactly the keys `prompt` / `name` /
`parameters`, every prompt echoed unchanged, every name one of the defined
functions, every parameter present with its declared type. Constrained decoding
makes most of these impossible to fail, which is the point — they confirm the
guarantee holds rather than hunting for bugs.

**Semantic, and by hand.** Whether the *right* function and sensible values
were chosen cannot be checked automatically, because the project ships no
answer key: `function_calling_tests.json` contains the questions but not the
expected results. Those are read and judged manually.

`make lint` enforces PEP 8 (`flake8`) and type consistency (`mypy`).

---

## Performance analysis

Measured on CPU, which is what an evaluation machine is likely to use.

| Metric | Result |
|--------|--------|
| Run time | **97 s** for 11 prompts |
| Budget | 300 s |
| Valid JSON | 100%, by construction |
| Schema compliance | 100%, by construction |
| Function selection | 11 / 11 |

**Where the time goes.** Entirely in `get_logits_from_input_ids`. The SDK
exposes no KV cache, so every call re-reads the whole prefix and cost grows
with context length — roughly 7 ms per token of context on CPU. Both the number
of forward passes and the length of the context therefore matter, which is why
the program writes fixed text itself, skips passes where only one token is
legal, and drops the unchosen definitions before generating arguments.

**Scaling.** Under seven seconds per prompt, so the budget covers far more
prompts than the supplied file contains. Extra function definitions lengthen
only the first-stage prompt, which is used for one or two passes per request.

**Reliability.** Selection is greedy — highest score, never sampled — so runs
are reproducible.

---

## Challenges faced

**The model answers instead of calling.** Asked to reverse `'hello'`, it put
`"olleh"` in the argument, having reversed the string itself; the function was
supposed to do that. It even got it wrong, turning `'world'` into `"drowl"`.
Keeping the call visible in the context fixed both.

**Where a string ends.** The obvious rule is "stop at a `"` token", but the
tokenizer merges the closing quote with what follows, so the model reaches for
`",` or `"}`, never a bare quote. Excluding those left `}` as the best
remaining choice — legal *inside* a JSON string — so the model wrote braces as
content and carried on. Treating any token *starting* with a quote as the end
of the value, and keeping only the quote, fixed it.

**A constraint disguised as a model limitation.** A regex argument came out as
the literal text of the numbers it should have matched, which looked like a
0.6B model being unable to write `\d+`. It was not: backslashes had been
excluded from strings, so `\d+` could not be produced at all. When output looks
like the model failing, check first that the constraints did not make the right
answer unreachable.

**Runaway repetition.** With backslashes allowed the model wrote `\d+\s+` seven
times over, to the length cap. The first repetition guard then truncated
`"Hello 34 I'm 233 years old"` after `23`, because `233` is the tokens `'2'`,
`'3'`, `'3'` and the doubled digit looked like a loop. Requiring a repeated
block to be at least three characters wide separates the two cases.

**A loop with no way out.** Asking the model to use `0` for a missing value
made the program hang: the number loop was a `while True`, and having written
`0` the likeliest next token was another `0`. It would have run until memory
gave out, each pass slower than the last as the context grew. Investigating
that turned up two more holes in the same state machine — `007` and `2.` were
both reachable, and neither is valid JSON. Walking every path the grammar
allows and comparing against `json.loads` now shows no disagreement.

**A GPU that stops the program running.** The SDK sets `device_map="auto"`
whenever a GPU is visible, and that path requires `accelerate`, a package this
project may not add. On a machine with a GPU it crashed before loading
anything. Since the target is CPU, the device is now requested explicitly.

---

## Possible improvements

- **Implement the tokenizer.** `encode` and `decode` are still used; building
  BPE by hand would remove the last dependency on the SDK's tokenizer.
- **Beam search.** Greedy decoding commits to every token permanently. Keeping
  several candidates alive would improve the weaker regex arguments — `\d+\s+`
  finds every number but also swallows the space after it.
- **More of JSON Schema.** Optional parameters, enums, integers, booleans and
  nested objects are not handled; only the flat `number` and `string`
  parameters the given definitions use.
- **Unit tests.** The structural checks run against real output; the individual
  rules would benefit from tests that need no model.
