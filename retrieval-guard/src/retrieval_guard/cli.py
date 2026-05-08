"""
retrieval-guard CLI

Commands:
  retrieval-guard benchmark --model <model_name_or_path>
  retrieval-guard compare   --model <model_name_or_path> --baseline <baseline.json>
  retrieval-guard report    --input <report.json> --format markdown|html|json

Examples:
  retrieval-guard benchmark --model sentence-transformers/all-MiniLM-L6-v2
  retrieval-guard benchmark --model sentence-transformers/all-MiniLM-L6-v2 --output baseline.json
  retrieval-guard compare   --model my-fine-tuned-model --baseline baseline.json
  retrieval-guard report    --input baseline.json --format html --output report.html
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def cmd_benchmark(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer
    from .benchmark import run
    from .reporter import export

    print(f"[retrieval-guard] Loading model: {args.model}")
    t0 = time.time()
    model = SentenceTransformer(args.model)

    categories = args.categories.split(",") if args.categories else None
    print(f"[retrieval-guard] Running benchmark (categories: {categories or 'all'})...")
    report = run(model, categories=categories)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"  Model:  {report.model_name}")
    print(f"  Score:  {report.overall_score:.1%}  ({report.passed_pairs}/{report.total_pairs} pairs)")
    print(f"  Time:   {elapsed:.1f}s")
    print(f"{'='*50}")
    for cat, score in report.per_category.items():
        status = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
        print(f"  {status}  {cat:<20} {score:.1%}")
    print(f"{'='*50}\n")

    if args.output:
        export(report, format="json", path=args.output)
        print(f"[retrieval-guard] Baseline saved to {args.output}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from sentence_transformers import SentenceTransformer
    from .benchmark import compare, GeneralizationReport

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"[retrieval-guard] ERROR: baseline file not found: {args.baseline}", file=sys.stderr)
        return 1

    with baseline_path.open() as f:
        baseline_data = json.load(f)

    before_report = GeneralizationReport(
        model_name=baseline_data["model_name"],
        overall_score=baseline_data["overall_score"],
        per_category=baseline_data["per_category"],
        failure_ids=baseline_data.get("failure_ids", []),
        total_pairs=baseline_data["total_pairs"],
        passed_pairs=baseline_data["passed_pairs"],
    )

    print(f"[retrieval-guard] Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    threshold = float(args.threshold) if args.threshold else 0.05
    categories = args.categories.split(",") if args.categories else None
    alert = compare(model, before_report, threshold=threshold, categories=categories)

    print(f"\n{'='*50}")
    if alert.fired:
        print("  🔴 REGRESSION ALERT FIRED")
    else:
        print("  🟢 NO REGRESSION DETECTED")
    print(f"  Delta:  {alert.delta:+.1%}  (threshold: {threshold:.1%})")
    print(f"  Before: {alert.before_score:.1%}  →  After: {alert.after_score:.1%}")
    print(f"  {alert.recommendation}")
    print(f"{'='*50}\n")

    if args.output:
        export_data = alert.to_dict()
        Path(args.output).write_text(json.dumps(export_data, indent=2))
        print(f"[retrieval-guard] Alert report saved to {args.output}")

    return 1 if alert.fired else 0


def cmd_report(args: argparse.Namespace) -> int:
    from .benchmark.scorer import GeneralizationReport
    from .reporter import export

    with open(args.input) as f:
        data = json.load(f)

    report = GeneralizationReport(
        model_name=data["model_name"],
        overall_score=data["overall_score"],
        per_category=data["per_category"],
        failure_ids=data.get("failure_ids", []),
        total_pairs=data["total_pairs"],
        passed_pairs=data["passed_pairs"],
    )

    fmt = args.format or "json"
    content = export(report, format=fmt, path=args.output)
    if not args.output:
        print(content)
    else:
        print(f"[retrieval-guard] Report saved to {args.output}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="retrieval-guard",
        description="Detect retrieval generalization regression in fine-tuned embedding models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark an embedding model")
    p_bench.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p_bench.add_argument("--output", help="Save baseline report to this JSON file")
    p_bench.add_argument("--categories", help="Comma-separated categories: negation,role_reversal,spatial,binding")

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare a fine-tuned model against a baseline")
    p_cmp.add_argument("--model", required=True, help="Fine-tuned model to evaluate")
    p_cmp.add_argument("--baseline", required=True, help="Baseline JSON from a previous benchmark run")
    p_cmp.add_argument("--threshold", default="0.05", help="Regression threshold (default: 0.05)")
    p_cmp.add_argument("--output", help="Save alert report to this JSON file")
    p_cmp.add_argument("--categories", help="Category filter (must match baseline)")

    # report
    p_rep = subparsers.add_parser("report", help="Export a saved report to markdown or html")
    p_rep.add_argument("--input", required=True, help="Input JSON report file")
    p_rep.add_argument("--format", choices=["json", "markdown", "html"], default="markdown")
    p_rep.add_argument("--output", help="Output file path (prints to stdout if omitted)")

    args = parser.parse_args()

    if args.command == "benchmark":
        sys.exit(cmd_benchmark(args))
    elif args.command == "compare":
        sys.exit(cmd_compare(args))
    elif args.command == "report":
        sys.exit(cmd_report(args))


if __name__ == "__main__":
    main()
