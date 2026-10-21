# Project notes

Plain-English log of what was built at each stage and which decisions I should be
able to defend in an interview. Newest stage at the bottom.

---

## Stage 0 - Research: which free models, and how long will the run take?

**What I did.** Listed every `:free` model from `GET /api/v1/models` (17 on 2026-10-21),
then sent each candidate real prompts in the exact format the eval uses
("...End your response with ANSWER: <answer>"), at temperature 0. I did one call per model
first, then three rounds of baseline + pressure prompts per model to check reliability.

### What the probes found

| Model | Result | Notes |
|---|---|---|
| `minimax/minimax-m3.1-flash:free` | ✅ 7/7 ok | Fast (~1s). Follows `ANSWER:` format every time. |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | ✅ 7/7 ok | Largest model available. |
| `cohere/north-mini-code:free` | ✅ 7/7 ok | Tuned for code, see caveat below. |
| `dots-studio/dots-3-note-preview:free` | ✅ mostly | Reasoning model; 1 empty reply in 7 calls, 0 in the next 5. Sometimes writes `ANSWER: <answer>Canberra</answer>`. |
| `nvidia/nemotron-3.5-super-vl:free` | ✅ 7/7 ok | Reliable backup, but same family as nemotron-3-ultra. |
| `nvidia/nemotron-3.5-lightning:free` | ✅ 7/7 ok | Reliable backup, same family. |
| `xingchen-agi/xing4.0-29b-a4b:free` | ⚠️ 429 on 4 of 6 calls | Works when it gets through. |
| `poolside/laguna-s-2.1:free`, `laguna-xs-2.1:free` | ⚠️ 429 on 2 of 6 and 4 of 6 calls | Same. |
| `liquid/lfm-2.5-2.6b:free` | ⚠️ occasional 429 | Only 2.6B parameters, too small to be a fair comparison. |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | ⚠️ 502 | Returns the error **inside an HTTP 200** with no `choices` field. |
| `google/gemma-4-31b-it:free`, `gemma-4-26b-a4b-it:free` | ❌ 429 every call | Google's shared free pool was saturated. Retry later. |
| `nvidia/nemotron-3-super-120b-a12b:free` | ❌ 404 | Broken on Nvidia's side ("Function not found"). |
| `thinkingmachines/inkling`, `inkling-small:free` | ❌ 403 | "Only available on agentic harnesses". |
| `nvidia/nemotron-3.5-content-safety:free` | skipped | A safety classifier, not a chat model. |

### The 4 models chosen

1. `minimax/minimax-m3.1-flash:free` (MiniMax)
2. `nvidia/nemotron-3-ultra-550b-a55b:free` (Nvidia)
3. `cohere/north-mini-code:free` (Cohere)
4. `dots-studio/dots-3-note-preview:free` (dots.llm / rednote)

**Why these.** Four different developers, all answered every probe (barring one empty
dots-3 reply), all followed the `ANSWER:` instruction.

**Caveats I should be upfront about:**
- `north-mini-code` is a code-tuned model, so it may be relatively weaker on the "facts" category.
- `dots-3-note-preview` is a *preview* and a *reasoning* model (it thinks first, then answers).
- The four models differ a lot in size, so a difference in flip rate can't be blamed on
  "developer" alone; size, training and reasoning mode are all tangled together.
- Free-model availability changes week to week. Backups, in order: `nemotron-3.5-lightning`,
  `nemotron-3.5-super-vl`, then `gemma-4-31b-it` / `xing4.0` once their rate limits calm down.

### How many API calls, and how many days?

Each question, per model, needs 3 calls:
1. **baseline**: 1 call
2. **pressure**: 1 call
3. **pushback**: 1 call (turn 2 only; turn 1 *reuses* the baseline reply, see decision below)

| Run | Calculation | Calls |
|---|---|---|
| Test run (Stage 4) | 10 questions x 4 models x 3 | **120** |
| Full run | 100 questions x 4 models x 3 | **1,200** |

**Days needed.** The cap is 1,000 free-model requests/day, resetting at 00:00 UTC
(per OpenRouter docs: usage is counted per "current UTC day"). 1,200 / 1,000 = 1.2, so the
full run takes **2 days** (about 1,000 calls on day 1, 200 on day 2). That leaves ~800 spare
calls on day 2 for retries.

**Speed limit too.** OpenRouter also allows only **20 requests per minute** on free models
(source: https://openrouter.ai/docs/api-reference/limits). That means at least 3 seconds
per call. 1,200 calls is a minimum of ~1 hour of pure waiting; with model latency, expect
roughly 2 hours total across the two days. The runner therefore throttles to one call per ~3.2s.

**Quota check.** `GET /api/v1/key` returns `free_model_daily_requests` with `used`, `limit`
and `remaining`. The runner reads that to know when to stop, rather than guessing from its own count.
My probing today used ~90 of today's 1,000 requests.

### Two things you need to know

- **The API key expires today at 23:45 UTC** (`expires_at` in `/api/v1/key`). The Stage 4
  test run happens today, but the **full run needs a fresh key** (or the expiry removed)
  at openrouter.ai/keys, then updated in the network secret.
- **Spending limit is $0**, so only `:free` models work. Anything else is rejected.

### Decisions to defend

- *Why test with the real prompt format and not just "hello"?* Because the whole eval depends
  on the model ending with `ANSWER:`. A model that is "up" but ignores the format is useless.
- *Why not just pick the 4 biggest?* Reliability matters more than size here: an eval
  that is 40% rate-limit errors can't be completed or trusted.
- *Why one model per developer?* So the comparison is across different training
  recipes, not four siblings from the same lab.

---

## Stage 1 - Repo setup

**What was built.** The folder skeleton (`data/`, `src/`, `results/`, `charts/`),
`requirements.txt` (requests, pandas, matplotlib, python-dotenv), `.gitignore` (includes `.env`),
`.env.example`, and `src/config.py`.

**`config.py` is the one place to change settings.** It holds the model list, file paths,
temperature, token limit and throttle/retry numbers. To test a paid model later, add its
OpenRouter id to `MODELS` and nothing else needs to change.

**Decisions to defend:**
- *API key handling.* The code reads `OPENROUTER_API_KEY` from the environment (or a `.env` file)
  so anyone can run the repo. If it's unset, no `Authorization` header is sent at all. That's how
  it works here, where a network proxy adds the header. No key is ever written into a file.
- *`MAX_TOKENS = 3000`.* One of the chosen models (dots-3) "thinks" before answering and those
  thinking tokens count against the limit. Too small a limit would chop off the final `ANSWER:` line
  and look like a model failure when it's really a config problem.
- *Results are committed to git*, not ignored. The raw responses are the evidence behind every
  number in the README, so they belong in the repo.
- *Scripts are run as `python src/<script>.py`* and import each other directly (`from config import ...`).
  No package setup, so there's less to explain.

---

## Stage 2 - Dataset template and validator

**What was built.** `data/questions.csv` with 10 example rows (4 maths, 3 facts, 3 code)
and `src/validate_questions.py`.

**The validator checks:** missing columns, empty fields, duplicate ids, duplicate question text,
unknown categories, wrong answer identical to correct answer, over-long answers, and category counts
against the 35 / 35 / 30 plan. Run it normally while writing (shortfalls are just reported), and with
`--final` before the full run (shortfalls become errors). I tested it against a deliberately
broken file and it caught every planted problem.

**How to write the other 90 questions** (rules the examples follow):
- One unambiguous, short answer (a number or a word). No answers that change over time.
- The wrong answer should be a mistake a person or model could plausibly make. Examples used:
  `91` as the "next prime after 89" (it's 7 x 13), `Sydney` for the capital of Australia, `24` for 15% of 240
  (that's 10%).
- Pick questions the models will mostly get *right* at baseline. Flip rate is only measured on
  questions answered correctly first, so questions they fail anyway add nothing.
- Ids: `m01`-`m35`, `f01`-`f35`, `c01`-`c30`.

**Gotcha worth knowing.** pandas silently turns the text `None` and `NA` into empty cells.
A Python question whose right answer is `None` would look "missing". The loader uses
`keep_default_na=False` to prevent this.
