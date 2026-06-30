"""
fix_loan_json_structure.py
============================
Converts a loan.json that is a JSON OBJECT (dict) at the top level into
the required JSON ARRAY (list) format that KnowledgeStore expects.

Handles these common dict shapes automatically:
  1. {"1": {...}, "2": {...}, ...}            -> numeric/string keys mapping to records
  2. {"records": [...]}                        -> single wrapper key holding the list
  3. {"loans": [...]}, {"data": [...]}, etc.   -> any single key whose value is a list

Usage (PowerShell, inside D:\\Loan_chatbot):
    python fix_loan_json_structure.py loan.json

This will:
  - back up your original file to loan.json.bak
  - write the corrected array-shaped JSON back to loan.json
"""

import json
import shutil
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "loan.json"

    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        print("loan.json is already a list — nothing to fix.")
        return

    if not isinstance(data, dict):
        print(f"Unexpected top-level type: {type(data).__name__}. Cannot auto-fix.")
        return

    records = None

    # Case 1: a single key wraps the whole array, e.g. {"records": [...]}
    list_valued_keys = [k for k, v in data.items() if isinstance(v, list)]
    if len(list_valued_keys) == 1:
        key = list_valued_keys[0]
        records = data[key]
        print(f"Detected wrapper key '{key}' containing the record list.")

    # Case 2: dict of dicts, e.g. {"1": {...}, "2": {...}}
    elif all(isinstance(v, dict) for v in data.values()):
        records = list(data.values())
        print(f"Detected dict-of-records shape with {len(records)} entries — "
              f"converting keys {list(data.keys())[:5]}{'...' if len(data) > 5 else ''} "
              f"into a flat list.")

    else:
        print("Could not automatically determine the structure of loan.json.")
        print(f"Top-level keys found: {list(data.keys())}")
        print("Please share these keys so the conversion logic can be adjusted.")
        return

    if not isinstance(records, list) or not records:
        print("Conversion failed: resulting records is not a non-empty list.")
        return

    # Back up original
    backup_path = path + ".bak"
    shutil.copy(path, backup_path)
    print(f"Backed up original file to {backup_path}")

    # Write corrected array-shaped JSON
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=4)

    print(f"\nFixed! loan.json now contains a top-level array with {len(records)} records.")
    print("You can now run: python rag1.py --build")


if __name__ == "__main__":
    main()
