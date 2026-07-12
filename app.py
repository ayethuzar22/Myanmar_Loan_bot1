"""
app.py — CLI entry point for the Wonderami Bilingual Loan RAG Chatbot.

Usage (unchanged from the original single-file rag1.py):
  python app.py --build              # Build / rebuild FAISS index
  python app.py --query "your text"  # Single query then exit
  python app.py                      # Interactive REPL

This module owns only orchestration glue: argument parsing, the
interactive REPL loop, and the calculator state machine. All business
logic (retrieval, generation, loan-mode handling) lives in
rag.rag_pipeline.
"""

from __future__ import annotations
import argparse
import os
import sys

from config import HISTORY_WINDOW, RAW_JSON_PATH, _RE_DIGITS, log
from models.chat_turn import ChatTurn
from rag.rag_pipeline import _get_pipeline, build_index, retrieve
from utils.loan_utils import calculate_microfinance_loan
from rag.conversation_memory import ConversationMemory

def _run_repl(json_path: str) -> None:
    """Blocking interactive REPL.  Not used in production Django."""
    # ── Gemini API key check ──────────────────────────────────────────────────
    # NOTE (carried over unmodified from rag1.py): this default-value
    # fallback re-introduces the hardcoded placeholder key that FIX #1
    # elsewhere in this project deliberately removed from GeminiClient.
    # It has NO effect on authentication (GeminiClient reads the env var
    # itself and does not accept this value), but as a stale/dead
    # credential string it should be deleted, not preserved, the next
    # time this file is touched.
    if not os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6KpC-CNFRqWm6m6_FwRKDc0jLlI5PnNoR7LC1jPeUypVw").strip():
        print("\n" + "!" * 62)
        print("  WARNING: GEMINI_API_KEY is not set.")
        print("  The bot will answer from the knowledge base directly,")
        print("  but full LLM-powered responses require a Gemini API key.")
        print()
        print("  To enable Gemini:")
        print("  1. Get a key from https://aistudio.google.com/apikey")
        print("  2. Set it as an environment variable, e.g.:")
        print("     Windows PowerShell:")
        print('       $env:GEMINI_API_KEY = "<your key here>"')
        print("     Windows CMD:")
        print('       set GEMINI_API_KEY=<your key here>')
        print("     macOS/Linux:")
        print('       export GEMINI_API_KEY="<your key here>"')
        print("  Or add it permanently via System Properties > Environment Variables")
        print("!" * 62 + "\n")

    pipeline = _get_pipeline(json_path)
    history: list[ChatTurn] = []
    memory = ConversationMemory()
    is_calculating = False
    calc_step      = 0
    p_amt          = 0.0

    print("\n" + "\u2550" * 62)
    print("  WONDERAMI SMART LOAN AI ASSISTANT  (Bilingual REPL)")
    print("\u2550" * 62)
    print("\u2022 \u1018\u102c\u101e\u102c\u1015\u103c\u1014\u103a\u101b\u1014\u103a  : 'translate with myanmar' \u101b\u102d\u102f\u1000\u103a\u1015\u102b")
    print("\u2022 \u1015\u102d\u1010\u103a\u101b\u1014\u103a      : 'exit' \u101e\u102d\u1037\u1019\u103d\u101f\u102f\u1037\u1000\u103a '\u1011\u103d\u1000\u103a\u1019\u101a\u103a' \u101b\u102d\u102f\u1000\u103a\u1015\u102b")
    print("\u2500" * 62)
    print(
        "AI: \u1019\u1004\u103a\u1039\u1002\u101c\u102c\u1015\u102b \u1001\u1004\u103a\u1017\u103b\u102c! \u1000\u103b\u103d\u1014\u103a\u1010\u102c\u1037\u1000\u103a Wonderami Smart Loan AI "
        "Assistant \u1016\u103c\u1005\u1015\u102b\u1078\u101a\u103a\u104d \u1018\u102c\u1019\u103b\u102c\u1038 \u1000\u1030\u100a\u102d\u1015\u1031\u1038\u101b\u1019\u101c\u1032\u1038 \u1001\u1004\u103a\u1017\u103b\u102c?\n"
    )

    while True:
        try:
            u_in = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAI: \u1000\u1031\u102c\u1004\u103a\u101e\u1031\u102c\u1014\u1031\u1037\u101c\u1031\u1038 \u1016\u103c\u1005\u1015\u102b\u1005\u1031 \u1001\u1004\u103a\u1017\u103b\u102c\u104d \u2728")
            break

        if not u_in:
            continue
        if u_in.lower() in {"exit", "\u1011\u103d\u1000\u103a\u1019\u101a\u103a", "bye", "goodbye"}:
            print("AI: \u1000\u1031\u102c\u1004\u103a\u101e\u1031\u102c\u1014\u1031\u1037\u101c\u1031\u1038 \u1016\u103c\u1005\u1015\u102b\u1005\u1031 \u1001\u1004\u103a\u1017\u103b\u102c\u104d \u2728")
            break

        # Calculator state machine
        if is_calculating:
            nums = _RE_DIGITS.findall(u_in.replace(",", ""))
            if not nums:
                print("AI: \u1000\u103b\u1031\u1038\u1007\u1030\u1038\u1015\u103c\u102f\u1024 \u1000\u1014\u103a\u1038\u1002\u100a\u103a\u1015\u102c\u1038 \u1021\u1078\u102d\u1021\u1000\u103b \u101b\u102d\u102f\u1000\u103a\u1011\u100a\u1037\u101e\u103d\u1004\u103a\u1038\u1015\u1031\u1038\u100a\u102c\u101c\u102c\u1038 \u1001\u1004\u103a\u1017\u103b\u102c\u104d")
                continue
            val = float(nums[0])
            if calc_step == 1:
                p_amt     = val
                calc_step = 2
                print(
                    "AI: \u1015\u103c\u1014\u103a\u1006\u1015\u103a\u101b\u1019\u100a\u103a\u1037 \u101e\u1000\u103a\u1010\u1019\u103a\u1038\u1000\u102d\u102f '\u101c' \u1021\u101c\u102d\u102f\u1000\u103a \u101b\u102d\u102f\u1000\u103a\u1015\u1031\u1038\u100a\u102c\u101c\u102c\u1038 (\u1041 \u1019\u103e \u1042\u1040 \u101c):"
                )
            elif calc_step == 2:
                if not 6 <= val <= 24:
                    print(
                        "AI: \u1001\u103b\u1031\u1038\u1004\u103a\u1040\u101e\u1000\u103a\u1010\u1019\u103a\u1038\u1000\u102d\u102f \u1041 \u101c \u1019\u103e \u1042\u1040 \u101c \u1021\u1078\u103d\u1004\u103a\u1038\u101e\u102c\u1019\u1037 \u1001\u103d\u1004\u103a\u1019\u1015\u1016\u102d\u102f\u1015\u102b\u101e\u100a\u103a \u1001\u1004\u103a\u1017\u103b\u102c\u104d \u1015\u103c\u1014\u103a\u101b\u102d\u102f\u1000\u103a\u1015\u1031\u1038\u100a\u102c\u101c\u102c\u1038:"
                    )
                    continue
                print("\n" + "\u2550" * 52)
                print("  \U0001f4ca \u1001\u103b\u1031\u1038\u1004\u103a\u1040 \u1078\u1000\u103a\u1001\u103b\u1000\u103a\u1019\u103e\u102f \u101b\u101c\u1012\u103a")
                print("\u2550" * 52)
                try:
                    result_str = calculate_microfinance_loan(p_amt, int(val))
                    print(result_str)
                    history.append(ChatTurn(role="assistant", content=result_str))
                except ValueError as exc:
                    print(f"AI: \u1078\u1000\u103a\u1001\u103b\u1000\u103a\u1019\u103e\u102f \u1019\u103e\u102c\u101a\u103d\u1004\u103a\u1038\u1014\u1031\u1015\u102b\u101e\u100a\u103a \u2014 {exc}")
                print("\u2550" * 52 + "\n")
                is_calculating = False
                calc_step      = 0
            continue

        # Standard query
        response = pipeline.run(u_in, history, memory=memory)

        if response.answer == "LAUNCH_CALCULATOR":
            is_calculating = True
            calc_step      = 1
            print(
                "AI: \u1040\u102f\u1000\u103a\u1000\u1032\u1037\u1015\u102b \u1001\u1004\u103a\u1017\u103b\u102c\u104a \u1001\u103b\u1031\u1038\u1004\u103a\u1040 \u1078\u1000\u103a\u1001\u103b\u1000\u103a\u1016\u102d\u1037\u1021\u1078\u103d\u1000\u103a "
                "\u1001\u103b\u1031\u1038\u101a\u1030\u101c\u102d\u101a\u101e\u1031\u102c '\u1004\u103a\u1040\u1015\u1019\u102c\u100f\u1014\u103a (\u1021\u101b\u1004\u103a\u1038)' \u1000\u102d\u102f \u1002\u100a\u103a\u1015\u102c\u1038\u1021\u1078\u102d\u101a\u102c\u1001\u103b\u1031\u1038\u1014\u102d\u102f\u1004\u103a\u1015\u102b \u1001\u1004\u103a\u1017\u103b\u102c:"
            )
        else:
            print(
                f"\n  [source={response.source} | "
                f"score={response.similarity_score:.3f} | "
                f"topic={response.matched_topic}]"
            )
            print(f"AI: {response.answer}\n")
            history.append(ChatTurn(role="user",      content=u_in))
            history.append(ChatTurn(role="assistant", content=response.answer))
            if len(history) > HISTORY_WINDOW * 2:
                history = history[-(HISTORY_WINDOW * 2):]


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate action."""
    if sys.platform == "win32":
        import io as _io
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Wonderami Bilingual Loan RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app.py --build\n"
            "  python app.py --query 'Agriculture loan interest?'\n"
            "  python app.py\n"
        ),
    )
    parser.add_argument(
        "--build", action="store_true",
        help="(Re-)build FAISS index from --json then exit",
    )
    parser.add_argument(
        "--json", default=RAW_JSON_PATH, metavar="PATH",
        help="Path to loan.json (default: %(default)s)",
    )
    parser.add_argument(
        "--query", metavar="TEXT",
        help="Run a single query then exit",
    )
    args = parser.parse_args()

    if args.build:
        build_index(args.json)
        return

    if args.query:
        res = retrieve(args.query, json_path=args.json)
        print(
            f"\n[source={res['source']} | score={res['similarity_score']:.3f} | "
            f"topic={res['matched_topic']}]"
        )
        print(f"AI: {res['answer']}\n")
        return

    _run_repl(args.json)


if __name__ == "__main__":
    main()