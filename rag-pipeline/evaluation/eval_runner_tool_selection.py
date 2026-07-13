"""
Tool Selection Isolation Eval Runner
------------------------------------
Tests whether the LLM picks the right tool (rag_search / web_search / none)
WITHOUT actually running the tools. This is isolation testing:
we're checking the DECISION, not the EXECUTION.

How it works:
1. Load tool_selection_eval.json
2. For each query, send it to llm_with_tools (same LLM + same system prompt
   as rag_agent.py's llm_node) and call .invoke()
3. Read response.tool_calls — this tells us what the LLM decided to do,
   without it actually running rag_search or web_search
4. Compare decision to expected_tool, mark match True/False
5. Save results + print a summary
"""

import argparse
import json
import time
import sys
from pathlib import Path
import time

from groq import BadRequestError, RateLimitError

# -----------------------------------------------------------------------
# Path setup — same pattern as your other agent files
# -----------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Import the real tool objects so llm.bind_tools() sees the exact same
# tool names/descriptions/schemas your production agent uses.
from agents.rag_tool import rag_search
from agents.tavily_tool import web_search

# -----------------------------------------------------------------------
# Must be IDENTICAL to rag_agent.py's SYSTEM_PROMPT.
# Why: if this drifts from the real prompt, we're testing a different
# agent than the one that actually runs in production.
# -----------------------------------------------------------------------
SYSTEM_PROMPT = """You are a research assistant with access to two tools:

1. rag_search — use this for questions about AI, LLMs, RAG pipelines,
   retrieval systems, agents, multi-agent systems, LLM engineering,
   fine-tuning, and related research topics from academic papers.

2. web_search — use this for questions where the answer could be
   current, recent, or subject to change: current events, live prices,
   sports results, current officeholders, weather, or any topic where
   being out-of-date would make the answer wrong.

3. If the question is basic math, unit conversion, spelling, word
   definitions/opposites, casual conversation, or a stable fact that
   doesn't change over time (e.g. historical dates, scientific constants) —
   answer it directly yourself, without calling any tool.

4. EXCEPTION to rule 3: if the question asks about a specific research
   paper, its authors, findings, or claims — always use rag_search, even
   if you believe you already know the answer. This ensures the answer
   is grounded in the actual paper and properly cited.

RULES:
1. Always pass the user's COMPLETE original question to the tool. Never shorten it.
2. Pick the right tool based on the query topic.
3. After getting the tool result, check the 'action' field in rag_search results:
   - 'generate' → answer confidently
   - 'generate_cautiously' → answer but mention uncertainty
   - 'retry_or_abstain' → tell the user you cannot find relevant information
4. Always cite sources in your answer.
"""

llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools([rag_search, web_search])

EVAL_FILE = Path(__file__).parent / "tool_selection_eval.json"
RESULTS_FILE = Path(__file__).parent / "tool_selection_eval_results.json"


def get_actual_tool(query: str) -> tuple[str, list]:
    """
    Sends one query to the LLM and returns:
    - actual_tool: "rag_search" | "web_search" | "none" | "multiple"
    - raw_tool_calls: the raw list, kept for debugging if something looks off
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    max_attempts = 2
    last_error = None
    response = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm_with_tools.invoke(messages)
            break # success - stop retrying
        except BadRequestError as e:
            last_error = str(e)
            if attempt < max_attempts:
                time.sleep(1) # wait before retrying
                continue

            # Groq's tool-calling models occasionally emit a malformed function
            # call (e.g. raw "<function=...>" text) instead of a structured tool
            # call. Treat it as a failed decision for this query instead of
            # crashing the whole eval run.
            # both attempts failed, return error
            return "error", [{"error": last_error, "attempts": max_attempts}]

    if response is None:
        return "error", [{"error": last_error, "attempts": max_attempts}]

    tool_calls = response.tool_calls  # list, possibly empty

    if len(tool_calls) == 0:
        return "none", tool_calls
    if len(tool_calls) == 1:
        return tool_calls[0]["name"], tool_calls
    # More than one tool called in a single turn — flag it, don't silently pick one
    return "multiple", tool_calls


def run_eval(query_ids: set[str] | None = None):
    with open(EVAL_FILE, "r") as f:
        eval_set = json.load(f)

    # Resume support: if a previous (possibly partial) run already saved
    # results, reuse it as the starting point so completed queries aren't
    # re-sent to Groq (they'd just burn quota for no reason).
    # NOTE: eval_set always stays the FULL list here, and every save below
    # writes the FULL eval_set back out -- so filtering by --ids only
    # affects which queries get printed/re-run below, never what's on disk.
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            eval_set = json.load(f)
        already_done = sum(1 for item in eval_set if "actual_tool_called" in item)
        print(f"Resuming from {RESULTS_FILE.name} — {already_done}/{len(eval_set)} queries already done.")

    if query_ids:
        target_items = [item for item in eval_set if item["query_id"] in query_ids]
        print(f"--ids applied -- (re)running {len(target_items)} of {len(query_ids)} requested queries.")
    else:
        target_items = eval_set

    correct = 0
    category_totals = {}
    category_correct = {}
    mismatches = []

    processed = 0

    for item in target_items:
        query = item["query"]
        expected = item["expected_tool"]
        category = item["category"]
        raw_calls = None

        # When specific --ids are requested, the user wants a fresh run for
        # those queries, so always call the LLM instead of reusing a saved result.
        if query_ids is None and item.get("actual_tool_called") is not None:
            # Already evaluated in a previous run — reuse the saved result
            # instead of calling the LLM again.
            actual = item["actual_tool_called"]
            match = item["match"]
        else:
            try:
                actual, raw_calls = get_actual_tool(query)
            except RateLimitError:
                print(f"[{item['query_id']}] Rate limit hit — waiting 30s and retrying this query...")
                time.sleep(30)
                try:
                    actual, raw_calls = get_actual_tool(query)
                except RateLimitError:
                    # Still limited after the retry — likely the daily quota,
                    # not a short throttle. Save progress and stop instead of
                    # crashing; re-running the script later will resume here.
                    print(f"\nStill rate-limited after retry. Progress saved — "
                          f"rerun the script later to continue from query [{item['query_id']}].")
                    with open(RESULTS_FILE, "w") as f:
                        json.dump(eval_set, f, indent=2)
                    return

            match = (actual == expected)
            item["actual_tool_called"] = actual
            item["match"] = match

            # Checkpoint after every new query so a crash never loses progress.
            with open(RESULTS_FILE, "w") as f:
                json.dump(eval_set, f, indent=2)

            # Small delay to be gentle on Groq API rate limits
            time.sleep(2)

        processed += 1
        category_totals[category] = category_totals.get(category, 0) + 1
        if match:
            correct += 1
            category_correct[category] = category_correct.get(category, 0) + 1
        else:
            mismatches.append({
                "query_id": item["query_id"],
                "query": query,
                "expected": expected,
                "actual": actual,
                "raw_tool_calls": raw_calls
            })

        print(f"[{item['query_id']}] expected={expected:<12} actual={actual:<12} match={match}")

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------
    total = processed
    print("\n=== SUMMARY ===")
    print(f"Overall accuracy: {correct}/{total} = {correct/total:.1%}")
    print("\nPer-category accuracy:")
    for cat in category_totals:
        c = category_correct.get(cat, 0)
        t = category_totals[cat]
        print(f"  {cat:<12}: {c}/{t} = {c/t:.1%}")

    if mismatches:
        print(f"\n{len(mismatches)} mismatches — see below:")
        for m in mismatches:
            print(f"  [{m['query_id']}] '{m['query']}' -> expected {m['expected']}, got {m['actual']}")
    else:
        print("\nNo mismatches.")

    print(f"\nFull results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tool-selection eval, optionally for specific query_ids only.")
    parser.add_argument("--ids", type=str, default=None,
                         help="Comma-separated query_ids to run, e.g. --ids tool0012,tool005,tool017,tool021,tool022")
    args = parser.parse_args()

    ids = {qid.strip() for qid in args.ids.split(",")} if args.ids else None
    run_eval(query_ids=ids)