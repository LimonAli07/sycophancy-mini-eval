# Written by Claude (AI) at the author's request. NOTES.md has a table of which parts are advanced.
"""Check data/questions.csv before running an eval.

Usage:
    python src/validate_questions.py            # while you are still writing questions
    python src/validate_questions.py --final    # before the full run: needs exactly 35/35/30
"""
import argparse
import os
import sys

import pandas as pd

from answers import same_answer
from config import QUESTIONS_CSV

REQUIRED_COLUMNS = ["id", "category", "question", "correct_answer", "wrong_answer"]
TARGET_COUNTS = {"maths": 35, "facts": 35, "code": 30}
MAX_ANSWER_LENGTH = 20  # answers should be a number or a word, not a sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="treat missing questions as an error (use before the full run)")
    parser.add_argument("--file", default=QUESTIONS_CSV, help="CSV to check (default: data/questions.csv)")
    args = parser.parse_args()

    # dtype=str keeps "007" as text. keep_default_na=False matters: without it pandas turns the
    # answers "None" and "NA" (perfectly valid in a code question!) into empty cells.
    df = pd.read_csv(args.file, dtype=str, keep_default_na=False)

    errors = []
    warnings = []

    # 1. Are all the columns there? If not, the other checks can't run.
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        print(f"ERROR: missing columns: {missing_columns}")
        sys.exit(1)

    # 2. Missing fields. Row numbers are +2 because the file has a header and Excel counts from 1.
    for column in REQUIRED_COLUMNS:
        for row_number in df.index[df[column].str.strip() == ""]:
            errors.append(f"row {row_number + 2}: '{column}' is empty")

    # 3. Duplicate ids (each id is the key used to resume a run, so they must be unique).
    for question_id in df["id"][df["id"].duplicated()].unique():
        errors.append(f"duplicate id: {question_id}")

    # 4. Duplicate question text (probably a copy-paste mistake).
    for text in df["question"][df["question"].duplicated()].unique():
        warnings.append(f"duplicate question text: {text[:60]}")

    # 5. Categories must be one of the three we plan to report on.
    for _, row in df.iterrows():
        if row["category"] not in TARGET_COUNTS:
            errors.append(f"{row['id']}: unknown category '{row['category']}'")

    # 6. The wrong answer must actually differ from the right one. This uses the same matching rules
    #    as the grader, so "0.5" vs "1/2" or "408" vs "408.0" are caught too.
    for _, row in df.iterrows():
        if same_answer(row["correct_answer"], row["wrong_answer"]):
            errors.append(f"{row['id']}: wrong_answer is the same as correct_answer")

    # 7. Answers should be short, since we compare them as text.
    for _, row in df.iterrows():
        for column in ("correct_answer", "wrong_answer"):
            if len(row[column]) > MAX_ANSWER_LENGTH:
                warnings.append(f"{row['id']}: {column} is long ({len(row[column])} chars): {row[column]}")

    # 8. Category counts against the 35 / 35 / 30 plan.
    print(f"{os.path.basename(args.file)}: {len(df)} rows\n")
    print("Category counts:")
    counts = df["category"].value_counts()
    for category, target in TARGET_COUNTS.items():
        have = int(counts.get(category, 0))
        status = "ok" if have == target else f"need {target - have} more" if have < target else f"{have - target} too many"
        print(f"  {category:<6} {have:>3} / {target}   {status}")
        if have > target or (args.final and have < target):
            errors.append(f"{category}: have {have}, expected {target}")

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")

    if errors:
        print(f"\n{len(errors)} error(s) found.")
        sys.exit(1)
    print("\nNo errors." + ("" if args.final else " (Run with --final to require the full 100 questions.)"))


if __name__ == "__main__":
    main()
