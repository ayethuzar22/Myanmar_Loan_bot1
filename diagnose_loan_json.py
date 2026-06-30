"""
diagnose_loan_json.py
======================
Run this next to loan.json to find out exactly why records are
being rejected by KnowledgeStore.

Usage (PowerShell, inside D:\\Loan_chatbot):
    python diagnose_loan_json.py loan.json
"""

import json
import sys

REQUIRED_FIELDS = ("id", "category", "topic", "language", "question", "answer")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "loan.json"

    print(f"Opening: {path}\n")

    # Step 1: raw read + encoding check
    try:
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
    except OSError as exc:
        print(f"FILE READ ERROR: {exc}")
        return

    print(f"File size: {len(raw_bytes)} bytes")
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        print("NOTE: File has a UTF-8 BOM (this is OK, Python handles it).")
    if raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff"):
        print("WARNING: File looks like UTF-16, not UTF-8! This WILL break json.load().")
        print("Fix: re-save the file as UTF-8 (no BOM or with BOM) in your editor.")
        return

    # Step 2: try to decode as UTF-8 text
    try:
        text = raw_bytes.decode("utf-8-sig")  # handles BOM automatically
    except UnicodeDecodeError as exc:
        print(f"ENCODING ERROR: file is not valid UTF-8 — {exc}")
        print("Fix: re-save loan.json as UTF-8 in your text editor (e.g. VS Code: "
              "bottom-right corner -> click encoding -> 'Save with Encoding' -> UTF-8).")
        return

    # Step 3: try to parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"JSON SYNTAX ERROR: {exc}")
        print(f"  -> Look near line {exc.lineno}, column {exc.colno} "
              f"(character position {exc.pos}) in loan.json.")
        start = max(0, exc.pos - 80)
        end = min(len(text), exc.pos + 80)
        print("\n--- Context around the error ---")
        print(text[start:end])
        print("--- end context ---\n")
        return

    print("JSON parsed successfully.\n")

    # Step 4: check it's a list
    if not isinstance(data, list):
        print(f"STRUCTURE ERROR: top-level JSON must be a list [...], "
              f"but got {type(data).__name__}.")
        print("Fix: wrap your records in square brackets, e.g.:")
        print('  [\n    { "id": 1, ... },\n    { "id": 2, ... }\n  ]')
        return

    print(f"Total records found: {len(data)}\n")

    # Step 5: validate each record
    n_ok = n_inactive = n_invalid = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[record {i}] NOT A JSON OBJECT (got {type(item).__name__}): {item!r}")
            n_invalid += 1
            continue

        missing = [f for f in REQUIRED_FIELDS if not item.get(f)]
        active = item.get("active", True)

        if missing:
            print(f"[record {i}] id={item.get('id')!r} -> "
                  f"MISSING OR EMPTY FIELDS: {missing}")
            n_invalid += 1
        elif not active:
            print(f"[record {i}] id={item.get('id')!r} -> INACTIVE (active=false), "
                  f"will be skipped intentionally.")
            n_inactive += 1
        else:
            print(f"[record {i}] id={item.get('id')!r} -> OK")
            n_ok += 1

    print(f"\nSummary: ok={n_ok} inactive={n_inactive} invalid={n_invalid}")

    if n_ok == 0:
        print("\nNo valid records found. Check the MISSING FIELDS list above for each "
              "record and add/fix those keys in loan.json.")


if __name__ == "__main__":
    main()
