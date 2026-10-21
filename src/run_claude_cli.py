# Written by Claude (AI) at the author's request. NOTES.md has a table of which parts are advanced.
"""Run the same three conditions on Claude Haiku, through the `claude` command-line tool.

Why this exists: the OpenRouter free models run out of daily requests, and the OpenRouter key has a
$0 spending limit, so Claude can't be called through it. This script calls Claude with `claude -p`
(the Claude Code CLI) instead, and saves rows in the same results/results.jsonl format as run_eval.py.
Then grade.py and analyse.py treat it like any other model.

Usage
    python src/run_claude_cli.py --limit 3     # test on the first 3 questions
    python src/run_claude_cli.py               # all questions
    python src/run_claude_cli.py --dry-run     # show what is left, send nothing

How this differs from the OpenRouter runs (so the results are NOT perfectly comparable):
  - Temperature can't be set from the CLI, so each answer is one ordinary random sample
    (the OpenRouter models run at temperature 0).
  - The CLI adds about 1,300 tokens of its own boilerplate to every request. We switch off tools,
    plugins, MCP servers and settings, and replace the system prompt, but can't remove that.
  - The model may "think" before answering. The thinking is not saved.
Like run_eval.py it is safe to stop and re-run: finished calls are saved at once and skipped next time.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import EXTERNAL_MODELS, QUESTIONS_CSV, short_name
from run_eval import CONDITIONS, count_to_do, is_done, load_done, messages_for, save_row

MODEL = EXTERNAL_MODELS[0]                 # the name used in results.jsonl
CLAUDE_MODEL_ID = "claude-haiku-5-5"       # the exact model asked for (not the moving alias "haiku")
SYSTEM_PROMPT = "You are a helpful assistant."
# Always run from the same folder: --resume looks for its saved conversation in a folder named after this one.
WORK_DIR = Path(tempfile.gettempdir()) / "sycophancy_claude_cli"
CALL_TIMEOUT = 180   # seconds to wait for one call
MAX_ATTEMPTS = 3     # tries per call before giving up


class StopRun(Exception):
    """Something that retrying can't fix (not logged in, or a different model answered)."""


# AI-MADE (advanced): runs the `claude` program as a subprocess and reads its JSON output.
def ask_claude(prompt, session_id=None, resume_id=None):
    """Send one message to Claude. Returns the parsed JSON result from the CLI.

    session_id : save this conversation under that id (needed for the baseline, so pushback can continue it)
    resume_id  : continue a saved conversation (used for the pushback turn)
    neither    : a one-off call that is not saved (used for the pressure condition)
    """
    WORK_DIR.mkdir(exist_ok=True)
    command = [
        "claude", "-p", "--model", CLAUDE_MODEL_ID, "--output-format", "json",
        "--tools", "",                                        # no tools at all
        "--system-prompt", SYSTEM_PROMPT,                     # replace Claude Code's own system prompt
        "--disable-slash-commands", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "",                              # ignore user / project settings and plugins
    ]
    if resume_id:
        command += ["--resume", resume_id]
    elif session_id:
        command += ["--session-id", session_id]
    else:
        command += ["--no-session-persistence"]

    last_problem = "unknown"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # The prompt goes in through stdin, so odd characters in it can't be mistaken for options.
            run = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                 timeout=CALL_TIMEOUT, cwd=WORK_DIR)
            result = json.loads(run.stdout)
        except subprocess.TimeoutExpired:
            last_problem = "timed out"
        except ValueError:
            last_problem = f"output was not JSON: {run.stdout[:150]!r} {run.stderr[:150]!r}"
        else:
            if not result.get("is_error"):
                return result
            last_problem = f"error from claude: {str(result.get('result'))[:150]}"
            if "uthentication" in last_problem or result.get("api_error_status") in (401, 403):
                raise StopRun(f"Claude CLI is not logged in or not allowed: {last_problem}")
        wait = 5 * 2 ** (attempt - 1)
        print(f"    ! {last_problem}; waiting {wait}s (attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
        time.sleep(wait)
    raise StopRun(f"gave up after {MAX_ATTEMPTS} attempts; last problem: {last_problem}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only use the first N questions")
    parser.add_argument("--dry-run", action="store_true", help="show what is left to do and exit")
    args = parser.parse_args()

    questions = pd.read_csv(QUESTIONS_CSV, dtype=str, keep_default_na=False).to_dict("records")
    if args.limit:
        questions = questions[:args.limit]

    done = load_done()
    to_do = count_to_do(questions, [MODEL], done)
    print(f"{short_name(MODEL)}: {len(questions)} questions x {len(CONDITIONS)} conditions; "
          f"{len(questions) * len(CONDITIONS) - to_do} already done, {to_do} to do.")
    if args.dry_run or to_do == 0:
        return

    calls_made = 0
    try:
        for q in questions:
            for condition in CONDITIONS:
                messages = messages_for(q, MODEL, condition, done)
                if messages is None or is_done(done, MODEL, q, condition, messages):
                    continue

                session_id = resume_id = None
                if condition == "baseline":
                    session_id = str(uuid.uuid4())  # saved, so the pushback turn can continue this very conversation
                elif condition == "pushback":
                    resume_id = done[(MODEL, q["id"], "baseline")].get("session_id")
                    if not resume_id:
                        print(f"  skipping {q['id']} pushback: the baseline has no saved conversation to continue")
                        continue

                # Only the LAST message is sent; for pushback, --resume supplies the earlier turns.
                result = ask_claude(messages[-1]["content"], session_id=session_id, resume_id=resume_id)

                # Proof that Claude Haiku, and only that model, answered.
                served_by = list(result.get("modelUsage", {}))
                if served_by != [CLAUDE_MODEL_ID]:
                    raise StopRun(f"asked for {CLAUDE_MODEL_ID} but the CLI reports {served_by}")

                row = {
                    "model": MODEL, "id": q["id"], "condition": condition,
                    "response": result.get("result") or "",
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "messages": messages,
                    "returned_model": served_by[0], "provider": "Anthropic (via Claude Code CLI)",
                    "finish_reason": result.get("stop_reason"),  # 'end_turn' = finished normally
                    "reasoning": None,
                    # Only the baseline's id matters: the pushback turn resumes that conversation.
                    "session_id": result.get("session_id") if condition == "baseline" else None,
                }
                save_row(row)
                done[(MODEL, q["id"], condition)] = row
                calls_made += 1
                last_line = row["response"].strip().splitlines()[-1][:70] if row["response"].strip() else "(empty)"
                print(f"[{calls_made}/{to_do}] {q['id']} {condition:<8} {last_line}", flush=True)
    except StopRun as stop:
        print(f"\nSTOPPED: {stop}\n{calls_made} calls were saved this run. Re-run to resume.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nInterrupted. {calls_made} calls were saved this run. Re-run to carry on.")
        sys.exit(130)

    print(f"\nFinished: {calls_made} calls saved, {count_to_do(questions, [MODEL], done)} still to do.")


if __name__ == "__main__":
    main()
