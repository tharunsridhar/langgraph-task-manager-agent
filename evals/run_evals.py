"""Labelled evaluation suite for the task manager agent.

Each case in cases.json is a short scripted conversation with labels for what must happen:
which tools ran, which were blocked by the guardrails, what the task table looks like at
the end, and whether the graph is paused on a delete confirmation.

Two modes:

  python -m evals.run_evals            offline. A scripted model plays the LLM, so this measures
                                       the agent harness (tools, validation, human-in-the-loop,
                                       guardrails), not the language model. Needs no key or network.
  python -m evals.run_evals --live     runs the cases that carry a "live" label against the real
                                       Groq model (needs GROQ_API_KEY) and scores tool choice.

Options: --json PATH writes the per-case results. Exit code is 1 if any case fails.
"""
import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

import db  # noqa: E402
from agent import build_graph  # noqa: E402
from guardrails import BLOCK_PREFIX, NOTICE  # noqa: E402

CASES_PATH = Path(__file__).parent / "cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


class ScriptedModel:
    """Stands in for the LLM: replays the case's scripted steps in order and records every
    message list it was shown, so cases can check what the model was told."""

    def __init__(self, steps: list[dict]):
        self.steps = list(steps)
        self.seen: list[list] = []
        self._n = 0

    def bind_tools(self, tools):
        return self

    def _call_id(self) -> str:
        self._n += 1
        return f"call_{self._n}"

    def invoke(self, messages):
        self.seen.append(list(messages))
        if not self.steps:
            raise AssertionError("scripted model ran out of steps (the agent looped more than the case expects)")
        step = self.steps.pop(0)
        if "say" in step:
            return AIMessage(content=step["say"])
        calls = step["tools"] if "tools" in step else [step]
        return AIMessage(content="", tool_calls=[
            {"name": c["tool"], "args": c.get("args", {}), "id": self._call_id()} for c in calls
        ])


def _text(message) -> str:
    c = message.content
    return c if isinstance(c, str) else "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def _fresh_db(seed_tasks: list[dict]) -> str:
    path = str(Path(tempfile.mkdtemp()) / "tasks.db")
    db.DB_PATH = path
    db.init_db()
    for t in seed_tasks:
        row = db.create_task(t["title"], t.get("priority", "medium"), t.get("due_date"))
        if t.get("status") == "done":
            db.update_task(row["id"], status="done")
    return path


def run_case(case: dict, model=None) -> dict:
    """Run one case and return observed facts. model=None means the real Groq model."""
    _fresh_db(case["seed_tasks"])
    scripted = None
    if model is None and "model" in case and not os.getenv("EVAL_LIVE"):
        scripted = model = ScriptedModel(case["model"])
    graph = build_graph(model=model, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"eval-{case['id']}"}}

    for turn in case["turns"]:
        if "user" in turn:
            graph.invoke({"messages": [{"role": "user", "content": turn["user"]}]}, config)
        else:
            graph.invoke(Command(resume=turn["resume"]), config)

    state = graph.get_state(config)
    tool_msgs = [m for m in state.values["messages"] if m.type == "tool"]
    blocked = [m.name for m in tool_msgs if _text(m).startswith(BLOCK_PREFIX)]
    executed = [m.name for m in tool_msgs if not _text(m).startswith(BLOCK_PREFIX)]
    tasks = db.list_tasks()
    saw_notice = None
    if scripted is not None:
        saw_notice = any(
            m.type == "tool" and _text(m).startswith(NOTICE) for call in scripted.seen for m in call
        )
    return {
        "executed": executed,
        "blocked": blocked,
        "confirm_pending": bool(state.interrupts),
        "tasks": tasks,
        "task_count": len(tasks),
        "tool_outputs": [_text(m) for m in tool_msgs],
        "model_saw_notice": saw_notice,
    }


def score(expect: dict, got: dict) -> list[str]:
    """Return a list of mismatch descriptions (empty means the case passed)."""
    problems = []
    for key in ("executed", "blocked", "confirm_pending", "task_count"):
        if key in expect and got[key] != expect[key]:
            problems.append(f"{key}: expected {expect[key]!r}, got {got[key]!r}")
    if "model_saw_notice" in expect and got["model_saw_notice"] != expect["model_saw_notice"]:
        problems.append(f"model_saw_notice: expected {expect['model_saw_notice']!r}, got {got['model_saw_notice']!r}")
    if "tasks" in expect:
        want = sorted(expect["tasks"], key=lambda t: t["title"])
        have = sorted(got["tasks"], key=lambda t: t["title"])
        if len(want) != len(have):
            problems.append(f"tasks: expected {len(want)} task(s), got {len(have)}: {[t['title'] for t in have]}")
        else:
            for w, h in zip(want, have):
                for k, v in w.items():
                    if h.get(k) != v:
                        problems.append(f"task {w['title']!r}: {k} expected {v!r}, got {h.get(k)!r}")
    for needle in expect.get("tool_outputs_contain", []):
        if not any(needle in out for out in got["tool_outputs"]):
            problems.append(f"no tool output contains {needle!r}")
    return problems


def evaluate(live: bool = False, model=None) -> list[dict]:
    results = []
    for case in load_cases():
        expect = case.get("live") if live else case["expect"]
        if expect is None:
            continue
        try:
            got = run_case(case, model=model)
            problems = score(expect, got)
        except Exception as exc:  # a crash is a failed case, not a crashed report
            problems = [f"{type(exc).__name__}: {exc}"]
        results.append({"id": case["id"], "category": case["category"], "passed": not problems,
                        "problems": problems, "description": case["description"]})
    return results


def report(results: list[dict], mode: str) -> str:
    lines = [f"Task manager agent evaluation ({mode})", ""]
    for r in results:
        lines.append(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['category']:<18} {r['id']}")
        for p in r["problems"]:
            lines.append(f"        - {p}")
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += r["passed"]
    lines += ["", "By category:"]
    for cat, (ok, total) in sorted(by_cat.items()):
        lines.append(f"  {cat:<18} {ok}/{total}")
    passed = sum(r["passed"] for r in results)
    lines += ["", f"Total: {passed}/{len(results)} passed"]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="run against the real Groq model")
    parser.add_argument("--json", metavar="PATH", help="write per-case results as JSON")
    args = parser.parse_args(argv)

    if args.live:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        if not os.getenv("GROQ_API_KEY"):
            print("--live needs GROQ_API_KEY (set it in .env)", file=sys.stderr)
            return 2
        os.environ["EVAL_LIVE"] = "1"

    results = evaluate(live=args.live)
    print(report(results, "live, Groq model" if args.live else "offline, scripted model"))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
