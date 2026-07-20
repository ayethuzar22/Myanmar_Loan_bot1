from __future__ import annotations
import argparse
import sys

from config import HISTORY_WINDOW, RAW_JSON_PATH, _RE_DIGITS
from models.chat_turn import ChatTurn
from rag.rag_pipeline import _get_pipeline, build_index, retrieve
from utils.loan_utils import calculate_microfinance_loan
from rag.conversation_memory import ConversationMemory

def _run_repl(json_path: str) -> None:
    """Blocking interactive REPL.  Not used in production Django."""

    from llm.qwen_client import QwenClient

    client = QwenClient()
    if not client.is_available():
        print("\n" + "=" * 62)
        print("WARNING")
        print("Ollama server is not running.")
        print()
        print("Please start Ollama first:")
        print()
        print("    ollama serve")
        print()
        print("Verify the model exists:")
        print()
        print("    ollama list")
        print()
        print("Expected model:")
        print()
        print("    qwen2.5:1.5b")
        print("=" * 62 + "\n")

    pipeline = _get_pipeline(json_path)
    history: list[ChatTurn] = []
    memory = ConversationMemory()
    is_calculating = False
    calc_step      = 0
    p_amt          = 0.0

    print("\n" + "\u2550" * 62)
    print("  WONDERAMI SMART LOAN AI ASSISTANT  (Bilingual REPL)")
    print("\u2550" * 62)
    print("  Running locally with Ollama + Qwen2.5:1.5B")
    print("  Knowledge Base + FAISS Retrieval")
    print("  Local LLM Generation")
    print("\u2500" * 62)
    print("• ဘာသာပြန်ရန်  : 'translate with myanmar' လို့ရိုက်ပါ")
    print("• ထွက်ရန်: 'exit' သို့မဟုတ် 'ထွက်မယ်' လို့ရိုက်ပါ")
    print("\u2500" * 62)
    print(
         "AI: မင်္ဂလာပါ ခင်ဗျာ! ကြိုဆိုပါတယ်။ Wonderami Smart Loan AI "
        "Assistant ဖြစ်ပါတယ်။ ဘာမေးလို့ရမလဲ ခင်ဗျာ?\n"
    )

    while True:
        try:
            u_in = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAI: ကောင်းပြီ နောက်မှဆုံမယ် ခင်ဗျာ။ ✨")
            break

        if not u_in:
            continue
        if u_in.lower() in {"exit","ထွက်မယ်", "bye", "goodbye"}:
            print("AI: ကောင်းပြီ နောက်မှဆုံမယ် ခင်ဗျာ။ ✨")
            break

        # Calculator state machine
        if is_calculating:
            nums = _RE_DIGITS.findall(u_in.replace(",", ""))
            if not nums:
                print("AI: ကျေးဇူးပြု၍ ကိန်းဂဏန်းများကို အတိအကျ ရိုက်ထည့်ပေးပါ ခင်ဗျာ။")
                continue
            val = float(nums[0])
            if calc_step == 1:
                p_amt     = val
                calc_step = 2
                print(
                    "AI: ပြန်ဆပ်ရမည့် သက်တမ်းကို 'လ' အလိုက် ရိုက်ထည့်ပေးပါ (၆ မှ ၂၄ လ):"
                )
            elif calc_step == 2:
                if not 6 <= val <= 24:
                    print(
                        "AI: ချေးငွေသက်တမ်းကို ၆ မှ ၂၄ လ အတွင်းသာ ရွေးချယ်နိုင်ပါတယ် ခင်ဗျာ။ ပြန်လည်ရိုက်ထည့်ပေးပါ:"
                    )
                    continue
                print("\n" + "\u2550" * 52)
                print("  📊 ချေးငွေ တွက်ချက်မှု ရလဒ်")
                print("\u2550" * 52)
                try:
                    result_str = calculate_microfinance_loan(p_amt, int(val))
                    print(result_str)
                    history.append(ChatTurn(role="assistant", content=result_str))
                except ValueError as exc:
                    print(f"AI: တွက်ချက်မှု မှားယွင်းနေပါတယ် _ {exc}")
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
                "AI: အိုကေပါခင်ဗျာ။ ချေးငွေ တွက်ချက်ဖို့အတွက် "
                "ချေးယူလိုသည့် 'ငွေပမာဏ (အရင်း)' ကို ဂဏန်းအတိအကျရိုက်ထည့်ပေးပါ ခင်ဗျာ:"
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
        description="Wonderami Bilingual Loan RAG Chatbot (Local Ollama + Qwen)",
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