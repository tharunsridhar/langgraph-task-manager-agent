"""Runs the labelled evaluation suite (evals/cases.json) as tests. Offline, no key needed."""
import pytest

from evals import run_evals

CASES = {c["id"]: c for c in run_evals.load_cases()}


@pytest.mark.parametrize("case_id", list(CASES))
def test_case(case_id):
    case = CASES[case_id]
    got = run_evals.run_case(case)
    problems = run_evals.score(case["expect"], got)
    assert not problems, f"{case_id}: " + "; ".join(problems)


def test_suite_shape():
    """The suite stays a real suite: enough labelled cases, covering every behaviour area."""
    cats = {c["category"] for c in CASES.values()}
    assert {"tool_selection", "arg_validation", "human_in_the_loop", "prompt_injection"} <= cats
    assert len(CASES) >= 15
    assert all("expect" in c and c["turns"] for c in CASES.values())


def test_report_and_exit_code(tmp_path, capsys):
    out = tmp_path / "results.json"
    assert run_evals.main(["--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"Total: {len(CASES)}/{len(CASES)} passed" in text
    assert out.exists()


def test_scorer_detects_failures():
    """A wrong expectation must fail, so a green suite means something."""
    case = CASES["delete_approved"]
    got = run_evals.run_case(case)
    bad = dict(case["expect"], task_count=1, executed=["list_tasks"])
    problems = run_evals.score(bad, got)
    assert len(problems) == 2
