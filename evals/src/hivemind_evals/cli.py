"""hm-evals: the scenario list, the small set, and the b.6 comparison.

    hm-evals scenarios                    list the small set (rebuilds the evaluation database; no model)
    hm-evals small [--customer scripted]  run the small set once each (real models: sends made-up chat text to
                                          google for hivemind, to groq for the simulated customer, and to
                                          groq when the case lead's model is busy)
    hm-evals b6                           run the b.6 comparison (real models, as above)
    hm-evals hard [--customer scripted]   run the hard set once each (real models, as above)
    hm-evals leads --model P:M [--runs 3] the case-lead test: the hard set with the scripted customer, the case lead
                                          and the several-issue reconciler on one model with no fallback, the
                                          database rebuilt before each run (real models: google, and groq when P
                                          is groq)

every run first rebuilds its own database (hivemind_eval) from seed 42; the main database is never touched. the
report is written as json to --out and summed up on screen.
"""

import argparse
import json
import sys
from pathlib import Path

from hivemind_evals import harness, runner
from hivemind_evals.scenarios import opening as scenarios_opening


# the jobs --same-model puts on one model: the case lead and the baseline, so a b.6 run compares the shape alone
SAME_MODEL_JOBS = ("case_lead", "baseline_agent")
# the jobs the case-lead test puts on the model under test: the planning jobs of record 013
LEAD_JOBS = ("case_lead", "reconciler_complex")


def same_model(models, pick: str, jobs_on_it: tuple[str, ...] = SAME_MODEL_JOBS):
    """models.toml's settings with the given jobs on one model (provider:model), and no fallback, so no call of
    those jobs goes to another model."""
    from dataclasses import replace

    provider, _, model = pick.partition(":")
    if not model or provider not in models.key_envs:
        raise SystemExit(f"the model must be provider:model with a provider in models.toml, not {pick!r}")
    jobs = {**models.jobs}
    for name in jobs_on_it:
        jobs[name] = replace(jobs[name], provider=provider, model=model, fallback=None, fallback_provider=None)
    return replace(models, jobs=jobs)


def _router(pick: str | None = None, jobs_on_it: tuple[str, ...] = SAME_MODEL_JOBS):
    from hivemind_agents import settings, tracing
    from hivemind_agents.llm import ModelRouter

    models = settings.load_models()
    return ModelRouter(same_model(models, pick, jobs_on_it) if pick else models,
                       tracing.Tracer(tracing.make_provider(None)))


def _only(values: list[str] | None) -> list[tuple[str, str]] | None:
    return [tuple(v.split("/", 1)) for v in values] if values else None  # type: ignore[misc]


def _show(summary: dict) -> None:
    print(f"\n{summary['system']}: {summary['passed']}/{summary['scenarios']} scenarios passed every check")
    if summary["not_run"]:
        print(f"  not run (the simulated customer's model failed): {', '.join(summary['not_run'])}")
    for name, score in summary["checks"].items():
        print(f"  {name:16} {score}")
    print(f"  false corroborations {summary['false_corroborations']}, wrong orders {summary['wrong_orders']}, "
          f"questions {summary['questions']}")
    print(f"  model calls {summary['model_calls']}, tokens {summary['tokens']}, seconds {summary['seconds']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="hm-evals")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scenarios", help="list the small set")
    small = sub.add_parser("small", help="run the small set once each")
    small.add_argument("--customer", choices=("simulated", "scripted"), default="simulated")
    small.add_argument("--only", nargs="*", help="fault/variant, for example missing_item/store_miss")
    small.add_argument("--out", default="evals-small.json")
    b6 = sub.add_parser("b6", help="hivemind against minimum plan only and the single-agent baseline")
    b6.add_argument("--only", nargs="*")
    b6.add_argument("--out", default="evals-b6.json")
    b6.add_argument("--pause", type=float, default=0.0, help="seconds between scenarios, for a per-minute limit")
    b6.add_argument("--same-model", metavar="PROVIDER:MODEL",
                    help="the case lead and the baseline on one model, no fallback: google:gemini-3.5-flash-lite")
    hard = sub.add_parser("hard", help="run the hard set once each")
    hard.add_argument("--customer", choices=("simulated", "scripted"), default="simulated")
    hard.add_argument("--out", default="evals-hard.json")
    hard.add_argument("--pause", type=float, default=0.0)
    leads = sub.add_parser("leads", help="the case-lead test on one model")
    leads.add_argument("--model", required=True, metavar="PROVIDER:MODEL")
    leads.add_argument("--runs", type=int, default=3)
    leads.add_argument("--pause", type=float, default=0.0)
    leads.add_argument("--out", default="evals-leads.json")
    sub.add_parser("hard-scenarios", help="list the hard set")
    args = parser.parse_args()

    harness.prepare()
    is_hard = args.command in ("hard", "leads", "hard-scenarios")
    chosen = runner.load_scenarios(harness.EVAL_DB, _only(getattr(args, "only", None)), hard=is_hard)
    if args.command in ("scenarios", "hard-scenarios"):
        for s in chosen:
            for p in s.parts():
                print(f"{p.scenario_id:40} {p.case_type:22} {p.order_id}  names it by {p.names_order_by}"
                      + ("" if p.names_items else ", no item named"))
            print(f"  says: {scenarios_opening(s)}")
        return

    if args.command == "leads":
        llm = _router(args.model, LEAD_JOBS)
        results = []
        for n in range(args.runs):
            if n:
                harness.prepare()  # each run starts from the same data, with no case of the last run
            result = runner.run_chats(chosen, llm, "hivemind", "scripted", harness.EVAL_DB, pause_s=args.pause)
            result.system = f"{args.model} run {n + 1}"
            results.append(result)
    elif args.command == "hard":
        llm = _router()
        results = [runner.run_chats(chosen, llm, "hivemind", args.customer, harness.EVAL_DB, pause_s=args.pause)]
    elif args.command == "small":
        llm = _router()
        results = [runner.run_chats(chosen, llm, "hivemind", args.customer, harness.EVAL_DB)]
    else:
        llm = _router(args.same_model)
        chosen = runner.b6_scenarios(chosen)
        results = []
        for system in runner.SYSTEMS:
            harness.prepare()  # each system starts from the same data, with no case of another
            if system == "baseline":
                results.append(runner.run_baseline(chosen, llm, harness.EVAL_DB, args.pause))
            else:
                results.append(runner.run_chats(chosen, llm, system, "scripted", harness.EVAL_DB,
                                                pause_s=args.pause))
    report = {"command": args.command, "same_model": getattr(args, "same_model", None),
              "lead_model": getattr(args, "model", None),
              "scenarios": [s.to_json() for s in chosen],
              "results": [{"summary": r.summary(), "grades": [g.to_json() for g in r.grades],
                           "transcripts": r.transcripts} for r in results]}
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for r in results:
        _show(r.summary())
    print(f"\nreport: {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
