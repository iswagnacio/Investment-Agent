"""Replay the Week 6 benchmark against the stock analysis agent and a one-call baseline."""

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime
from hashlib import sha256
import json
import logging
from math import isfinite
from pathlib import Path
import re
from statistics import mean, quantiles
from time import perf_counter
from traceback import extract_tb

from langchain_core.callbacks import BaseCallbackHandler
from openai import APIError

from fundamental import FundamentalCalculator
from lead_agent import GeneralAnalystAgent, extract_json_from_text
from llm import create_chat_model
from sentiment import NewsArticle

ROOT = Path(__file__).resolve().parent / "benchmark"
MODEL = "glm-5.3-flash"
ROUNDS = ("baseline", "agent-0", "agent-1", "agent-2", "agent-3")


class Snapshot:
    """Implement the data interfaces consumed by the lead agent and specialists."""

    def __init__(self, records, trace):
        self.records = records
        self.trace = trace

    def get_stock_info(self, ticker):
        return self.records[ticker]["stock_info"]

    def get_all_indicators(self, ticker):
        value = self.records[ticker]["technical"]
        if value is None:
            self.trace.events.append({"stage": "technical data", "status": "missing", "ticker": ticker})
            raise ValueError(f"No price history for {ticker} in snapshot")
        return value

    def fetch_all(self, ticker, *args, **kwargs):
        data = self.records[ticker]["news"]
        articles = {
            kind: [NewsArticle.from_dict(article) for article in collection]
            for kind, collection in data.items()
        }
        company = articles["company"]
        industry = articles["industry"]
        return {
            "company_news": {"articles": company, "total_count": len(company)},
            "industry_news": {"articles": industry, "total_count": len(industry)},
            "summary": {
                "total_items": len(company) + len(industry),
                "sources_used": ["Yahoo Finance snapshot"] if company else [],
                "has_sentiment_scores": any(item.sentiment_hint is not None for item in company),
            },
        }


class SnapshotFundamentals(FundamentalCalculator):
    def __init__(self, records, trace):
        self.records = records
        self.trace = trace

    def get_all_fundamentals(self, ticker, **kwargs):
        data = self.records[ticker]["fundamental"]
        if not data:
            self.trace.events.append({"stage": "fundamental data", "status": "missing", "ticker": ticker})
        return data


class TaskTrace(BaseCallbackHandler):
    def __init__(self):
        self.calls = {}
        self.events = []
        self.provider_errors = []

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, **kwargs):
        self.calls[run_id] = (tags[0] if tags else "model", perf_counter())

    def on_llm_end(self, response, *, run_id, **kwargs):
        stage, start = self.calls.pop(run_id)
        usage = response.generations[0][0].message.usage_metadata or {}
        self.events.append({
            "stage": stage,
            "status": "ok",
            "seconds": round(perf_counter() - start, 3),
            **{key: usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")},
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        stage, start = self.calls.pop(run_id)
        status = getattr(error, "status_code", None)
        self.events.append({
            "stage": stage,
            "status": type(error).__name__,
            "http_status": status,
            "seconds": round(perf_counter() - start, 3),
        })
        if isinstance(error, APIError):
            body = error.body if isinstance(error.body, dict) else {}
            detail = body.get("error", body)
            code = detail.get("code") if isinstance(detail, dict) else None
            self.provider_errors.append({"http_status": status, "code": str(code)})


def task_records(task, snapshots):
    records = {}
    missing = {
        "stock_info": {"name": "Unknown", "sector": "Unknown", "industry": "Unknown", "market_cap": 0},
        "technical": None,
        "fundamental": {},
        "news": {"company": [], "industry": []},
    }
    for symbol in task["tickers"]:
        ticker = symbol.upper()
        records[ticker] = {**snapshots.get(ticker, missing), **task.get("overrides", {}).get(ticker, {})}
    return records


def traced_model(client, trace, stage, effort):
    # Apply the same GLM request settings to the agent and the single-call baseline.
    return client.bind(extra_body={"max_tokens": 8192, "reasoning_effort": effort}).with_config(
        {"callbacks": [trace], "tags": [stage]}
    )


def run_agent(task, records, trace, effort):
    agent = GeneralAnalystAgent(model=MODEL)
    if not all((agent.technical_agent, agent.fundamental_agent, agent.sentiment_agent)):
        raise RuntimeError("Stock analysis specialists did not initialize")

    snapshot = Snapshot(records, trace)
    fundamentals = SnapshotFundamentals(records, trace)
    agent.technical = agent.technical_agent.calculator = agent.sentiment_agent._tech_calc = snapshot
    agent.fundamental = agent.fundamental_agent.calculator = fundamentals
    agent.sentiment = agent.sentiment_agent.fetcher = snapshot
    for stage, analyst in (
        ("lead", agent),
        ("technical", agent.technical_agent),
        ("fundamental", agent.fundamental_agent),
        ("sentiment", agent.sentiment_agent),
    ):
        analyst.llm = traced_model(analyst.llm, trace, stage, effort)

    if task["operation"] == "compare":
        return agent.compare_stocks(task["tickers"])
    ticker = task["tickers"][0]
    if task["operation"] == "quick":
        return json.dumps(agent.get_quick_recommendation(ticker))
    return agent.analyze(ticker)


def run_baseline(task, records, trace, effort):
    client = traced_model(create_chat_model(MODEL), trace, "baseline", effort)
    output_format = {
        "quick": "Return one JSON object with ticker, company and current_price, plus any requested fields.",
        "compare": "Return a compact Markdown table with ticker and snapshot price on the same row.",
        "report": "Return a short report in English with clear labels for the requested facts.",
    }[task["operation"]]
    prompt = (
        "You are a stock analysis assistant in a frozen-data benchmark. Use only the supplied snapshot. "
        "For unavailable information, explicitly say it is unavailable and omit invented figures. "
        f"{output_format}\n\nTask: {task['question']}\n\n"
        "Snapshot as captured (technical.price.current is the reference price; null means missing):\n"
        + json.dumps(records, ensure_ascii=False, allow_nan=False)
    )
    return client.invoke(prompt).content


def grade_answer(task, answer):
    flags = re.IGNORECASE | re.MULTILINE
    outcomes = []
    parsed = extract_json_from_text(answer) if task["operation"] == "quick" else None
    for check in task["checks"]:
        if check.get("rule") == "price_target_order":
            try:
                target = parsed["price_targets"]
                low, mid, high = (float(target[key]) for key in ("low", "mid", "high"))
                passed = 0 < low < mid < high and isfinite(high)
            except (KeyError, TypeError, ValueError):
                passed = False
        elif check.get("rule") == "current_price":
            try:
                raw = str(parsed["current_price"]).strip().lstrip("$").replace(",", "")
                passed = abs(float(raw) - check["expected"]) <= 0.0051
            except (KeyError, TypeError, ValueError):
                passed = False
        elif check.get("rule") == "comparison_price":
            ticker = re.escape(check["ticker"])
            row = re.search(rf"^[^\S\n]*\|?[^\S\n]*\**{ticker}\**\b([^\n]*)", answer, flags)
            numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", row.group(1)) if row else []
            passed = any(abs(float(value.replace(",", "")) - check["expected"]) <= 0.0051
                         for value in numbers)
        else:
            passed = bool(re.search(check["pattern"], answer, flags)) != check.get("forbid", False)
        outcomes.append({"criterion": check["criterion"], "pass": passed})
    return outcomes


def trial(task, round_name, snapshots, effort):
    trace = TaskTrace()
    records = task_records(task, snapshots)
    start = perf_counter()
    answer = ""
    error = None
    try:
        runner = run_baseline if round_name == "baseline" else run_agent
        answer = runner(task, records, trace, effort)
        if not isinstance(answer, str):
            raise TypeError("Expected a text response")
    except Exception as exc:
        frames = extract_tb(exc.__traceback__)
        error = {
            "type": type(exc).__name__,
            "frames": [f"{Path(frame.filename).name}:{frame.lineno} ({frame.name})" for frame in frames[-4:]],
        }
    seconds = round(perf_counter() - start, 3)
    checks = grade_answer(task, answer)
    completed_calls = [event for event in trace.events if event.get("total_tokens") is not None]
    llm_calls = [event for event in trace.events if "seconds" in event]
    return {
        "id": task["id"], "category": task["category"], "round": round_name,
        "status": "provider_error" if trace.provider_errors else "complete",
        "passed": error is None and not trace.provider_errors and all(item["pass"] for item in checks),
        "seconds": seconds, "turns": len(llm_calls),
        "tokens": sum(item["total_tokens"] for item in completed_calls),
        "answer": answer, "checks": checks, "error": error,
        "provider_errors": trace.provider_errors, "trace": trace.events,
    }


def percentile95(values):
    return quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0]


def write_reports(results, tasks, manifest):
    by_id = {task["id"]: task for task in tasks}
    rows = [row for row in results.values() if row["status"] == "complete"]
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as output:
        columns = ("id", "category", "round", "passed", "seconds", "turns", "tokens")
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["round"], row["id"])))

    total = len(tasks) * len(ROUNDS)
    lines = [
        "# Week 6 benchmark results",
        "",
        f"Model: {MODEL}. Reasoning effort: {manifest['reasoning_effort']}. "
        f"Completed {len(rows)}/{total} evaluations.",
        "",
        "Market and news responses were replayed from the same frozen inputs. "
        "Latency was measured during concurrent execution and includes report generation. "
        "Snapshot capture and grading are outside that measurement. "
        "Turns count LLM calls. Token counts use provider-reported totals.",
        "",
        "The baseline receives the same frozen records in a single prompt. "
        "Comparison checks require ticker and price on the same row. "
        "News checks use predetermined event patterns, and price-range checks measure ordering.",
        "",
        "| System | Tasks | Accuracy | Mean turns | Mean latency (s) | P95 latency (s) | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for round_name, label in (("agent-0", "Agent"), ("baseline", "Single-call baseline")):
        group = [row for row in rows if row["round"] == round_name]
        if group:
            durations = [row["seconds"] for row in group]
            lines.append(
                f"| {label} | {len(group)}/{len(tasks)} | {mean(r['passed'] for r in group):.1%} | "
                f"{mean(r['turns'] for r in group):.2f} | {mean(durations):.2f} | "
                f"{percentile95(durations):.2f} | {mean(r['tokens'] for r in group):.1f} |"
            )
    reruns = ("agent-1", "agent-2", "agent-3")
    outcomes = {
        task_id: [results[task_id, round_name]["passed"] for round_name in reruns]
        for task_id in by_id
        if all(results.get((task_id, round_name), {}).get("status") == "complete" for round_name in reruns)
    }
    lines.extend(["", f"Three-run coverage: {len(outcomes)}/{len(tasks)} tasks."])
    if outcomes:
        lines.append(f"pass@3: **{mean(map(any, outcomes.values())):.1%}**. "
                     f"pass^3: **{mean(map(all, outcomes.values())):.1%}**.")
        flips = [
            f"`{task_id}` ({'/'.join('pass' if passed else 'fail' for passed in values)})"
            for task_id, values in outcomes.items() if any(values) and not all(values)
        ]
        lines.append("Flipped tasks: " + (", ".join(flips) if flips else "none") + ".")
    lines.extend([
        "",
        "## Reproduce",
        "",
        "Run from the repository root with a local Pi Z.ai configuration. "
        "Existing results are used as a checkpoint.",
        "",
        "```bash",
        "ZAI_USE_PI_AUTH=1 uv run --no-project --with-requirements requirements.txt python eval.py --workers 5",
        "```",
    ])
    (ROOT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failure = next((row for row in rows if row["round"] == "agent-0" and row["error"]
                    and row["error"]["type"] == "ZeroDivisionError"), None)
    if failure:
        missing = dict.fromkeys(event["stage"] for event in failure["trace"] if event["status"] == "missing")
        frame = failure["error"]["frames"][-1]
        paragraph = (
            f"In task {failure['id']}, the frozen fixture made {' and '.join(missing)} unavailable. "
            "The lead Agent therefore had no usable current price. Its price-target function returned zero "
            "for the current price, but the report formatting step still divided the low, mid, and high "
            f"targets by that value. The recorded run ended with ZeroDivisionError at {frame}, "
            "so the user received no report explaining that the price was unavailable. The expected "
            "behavior is to disclose the missing price and omit percentage returns."
        )
        (ROOT / "failure.md").write_text("# Failure trace\n\n" + paragraph + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--round", choices=ROUNDS, action="append")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="low")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    create_chat_model(MODEL)
    task_file = ROOT / "tasks.jsonl"
    snapshot_file = ROOT / "snapshots.json"
    tasks = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines()]
    snapshots = json.loads(snapshot_file.read_text(encoding="utf-8"))
    manifest = {
        "model": MODEL,
        "reasoning_effort": args.reasoning_effort,
        "inputs_sha256": sha256(task_file.read_bytes() + snapshot_file.read_bytes()).hexdigest(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest_file = ROOT / "run-config.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        parser.error("Existing run has different inputs or model settings. Move its outputs before creating a new run.")
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    journal = ROOT / "runs.jsonl"
    results = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            results[row["id"], row["round"]] = row
    jobs = deque(
        (task, round_name, 0)
        for round_name in (args.round or ROUNDS)
        for task in tasks[:args.limit]
        if results.get((task["id"], round_name), {}).get("status") != "complete"
    )
    planned = len(jobs)
    index = 0
    worker_limit = args.workers
    hard_provider_errors = 0
    stop_scheduling = False
    with journal.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        while pending or (jobs and not stop_scheduling):
            while jobs and len(pending) < worker_limit and not stop_scheduling:
                task, round_name, retries = jobs.popleft()
                future = pool.submit(trial, task, round_name, snapshots, args.reasoning_effort)
                pending[future] = (task, round_name, retries)
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                task, round_name, retries = pending.pop(future)
                row = future.result()
                errors = row["provider_errors"]
                if errors and all(error["code"] == "1302" for error in errors):
                    worker_limit = max(1, worker_limit - 1)
                    if retries < 2:
                        jobs.append((task, round_name, retries + 1))
                    print(f"Provider concurrency limit reached. Continuing with {worker_limit} workers.", flush=True)
                elif errors:
                    hard_provider_errors += 1
                    stop_scheduling = hard_provider_errors >= 5
                else:
                    hard_provider_errors = 0
                index += 1
                results[task["id"], round_name] = row
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                print(f"[{index}/{planned}] {row['round']} {row['id']}: {row['status']} "
                      f"passed={row['passed']} ({row['seconds']}s) {errors}", flush=True)
                if index % 5 == 0:
                    write_reports(results, tasks, manifest)
    write_reports(results, tasks, manifest)
    if jobs or any(row["status"] == "provider_error" for row in results.values()):
        raise SystemExit("Some requests failed at the provider. Re-run the same command to resume.")


if __name__ == "__main__":
    main()
