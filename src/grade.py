# Written by Claude (AI) at the author's request. NOTES.md has a table of which parts are advanced.
"""Turn the raw replies in results/results.jsonl into verdicts (results/graded.csv).

Grading is plain code (see answers.py). No AI is used to judge anything.

Usage
    python src/grade.py

Also writes results/manual_check.csv: 20 random graded rows for you to check by hand.
"""
import json

import pandas as pd

from answers import grade_response
from config import ALL_MODELS, GRADED_CSV, MANUAL_CHECK_CSV, QUESTIONS_CSV, RESULTS_JSONL
from run_eval import build_messages

MANUAL_CHECK_ROWS = 20
RANDOM_SEED = 42  # fixed so the same 20 rows are picked every time the data is the same


def load_latest_rows():
    """Read results.jsonl keeping only the LAST row for each (model, question, condition).

    A re-run appends a new row instead of rewriting the file, so the last one is the current one.
    """
    latest = {}
    with open(RESULTS_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                latest[(row["model"], row["id"], row["condition"])] = row
    return list(latest.values())


# AI-MADE (advanced): rebuilds the prompt each row should have used and checks it matches.
def prompt_matches_current_setup(row, question):
    """True if this row was produced by the same prompt the current questions.csv and run_eval.py would build.

    Catches the case where you edited a question (or the pushback wording) after running it.
    Such a row would be graded against an answer key it wasn't asked about.
    """
    baseline_response = None
    if row["condition"] == "pushback":
        baseline_response = row["messages"][1]["content"]  # the saved baseline reply that was replayed
    return row["messages"] == build_messages(row["condition"], question, baseline_response)


def main():
    questions = pd.read_csv(QUESTIONS_CSV, dtype=str, keep_default_na=False).set_index("id")
    rows = load_latest_rows()

    graded = []
    ignored_models = set()
    stale = 0
    for row in rows:
        if row["model"] not in ALL_MODELS:
            ignored_models.add(row["model"])  # e.g. a model that was swapped out of config.py
            continue
        if row["id"] not in questions.index:
            stale += 1
            continue
        question = {"id": row["id"], **questions.loc[row["id"]].to_dict()}
        if not prompt_matches_current_setup(row, question):
            stale += 1
            continue

        extracted, verdict = grade_response(row["response"], question["correct_answer"], question["wrong_answer"])
        if row["finish_reason"] == "length":
            # The reply hit the token limit, so it never finished (one model rambled about its own instructions for
            # 12,000 characters). Whatever "ANSWER:" fragment is lying around in the text is not a real answer.
            extracted, verdict = None, "unparseable"
        graded.append({
            "model": row["model"], "id": row["id"], "category": question["category"],
            "condition": row["condition"], "question": question["question"],
            "correct_answer": question["correct_answer"], "wrong_answer": question["wrong_answer"],
            "extracted_answer": extracted, "verdict": verdict,
            "finish_reason": row["finish_reason"], "response": row["response"],
        })

    if ignored_models:
        print(f"Ignored rows from models not in config.ALL_MODELS: {sorted(ignored_models)}")
    if stale:
        print(f"WARNING: skipped {stale} row(s) made with an old version of a question or prompt. "
              f"Re-run run_eval.py to refresh them.")
    if not graded:
        raise SystemExit("Nothing to grade.")

    df = pd.DataFrame(graded).sort_values(["model", "id", "condition"])
    df.to_csv(GRADED_CSV, index=False)
    print(f"Graded {len(df)} rows -> {GRADED_CSV}\n")

    # How are the verdicts spread? 'unparseable' is a formatting problem, not a flip.
    print("Verdicts by condition:")
    print(pd.crosstab(df["condition"], df["verdict"]).to_string(), "\n")
    unparseable = df[df["verdict"] == "unparseable"]
    if len(unparseable):
        print("Unparseable replies by model and condition:")
        print(unparseable.groupby(["model", "condition"]).size().to_string(), "\n")

    write_manual_check(df)


def write_manual_check(df):
    """Save a random sample for you to check by hand, without overwriting work you've already done."""
    if MANUAL_CHECK_CSV.exists():
        existing = pd.read_csv(MANUAL_CHECK_CSV, dtype=str, keep_default_na=False)
        if "human_verdict" in existing.columns and (existing["human_verdict"].str.strip() != "").any():
            print(f"{MANUAL_CHECK_CSV.name} already has your verdicts, so it was NOT overwritten. "
                  f"Delete it to pick a new sample.")
            return

    sample = df.sample(n=min(MANUAL_CHECK_ROWS, len(df)), random_state=RANDOM_SEED).copy()
    # The end of the reply is where the ANSWER: line is. Newlines would make the CSV hard to read.
    sample["response_end"] = sample["response"].str.strip().str[-300:].str.replace("\n", " / ", regex=False)
    sample["human_verdict"] = ""  # you fill in: correct / suggested_wrong / other_wrong / unparseable
    sample["notes"] = ""
    columns = ["model", "id", "condition", "question", "correct_answer", "wrong_answer",
               "extracted_answer", "verdict", "response_end", "human_verdict", "notes"]
    sample[columns].to_csv(MANUAL_CHECK_CSV, index=False)
    print(f"Wrote {len(sample)} random rows to {MANUAL_CHECK_CSV} - fill in 'human_verdict' and compare it with 'verdict'.")


if __name__ == "__main__":
    main()
