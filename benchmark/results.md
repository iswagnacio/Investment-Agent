# Week 6 benchmark results

Model: glm-5.3-flash. Reasoning effort: low. Completed 250/250 evaluations.

Market and news responses were replayed from the same frozen inputs. Latency was measured during concurrent execution and includes report generation. Snapshot capture and grading are outside that measurement. Turns count LLM calls. Token counts use provider-reported totals.

The baseline receives the same frozen records in a single prompt. Comparison checks require ticker and price on the same row. News checks use predetermined event patterns, and price-range checks measure ordering.

| System | Tasks | Accuracy | Mean turns | Mean latency (s) | P95 latency (s) | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| Agent | 50/50 | 88.0% | 4.42 | 46.71 | 92.11 | 8907.1 |
| Single-call baseline | 50/50 | 94.0% | 1.00 | 7.06 | 13.88 | 4791.3 |

Three-run coverage: 50/50 tasks.
pass@3: **90.0%**. pass^3: **90.0%**.
Flipped tasks: none.

## Reproduce

Run from the repository root with a local Pi Z.ai configuration. Existing results are used as a checkpoint.

```bash
ZAI_USE_PI_AUTH=1 uv run --no-project --with-requirements requirements.txt python eval.py --workers 5
```
