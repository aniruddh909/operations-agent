"""Render eval results as a Markdown table (for the README) and a console view."""

from __future__ import annotations

from .harness import EvalReport, compare

_METRIC_LABEL = {
    "gate": "Confidence-gate accuracy",
    "duplicate": "Duplicate verdict",
    "filed": "File / skip decision",
    "priority": "Priority correctness",
    "component": "Component correctness",
    "overall": "Overall",
}

_METRIC_ORDER = ["gate", "duplicate", "filed", "priority", "component", "overall"]


def to_markdown(report: EvalReport) -> str:
    """A Markdown table of metric -> accuracy (correct/total)."""
    lines = [
        f"### Eval results ({report.label})",
        "",
        "| Metric | Accuracy | Correct / Total |",
        "| --- | --- | --- |",
    ]
    for metric in _METRIC_ORDER:
        b = report.totals.get(metric)
        if not b:
            continue
        acc = report.accuracy(metric)
        label = _METRIC_LABEL.get(metric, metric)
        star = " (non-negotiable)" if metric == "gate" else ""
        lines.append(
            f"| {label}{star} | {acc:.0%} | {b['correct']}/{b['total']} |"
        )
    return "\n".join(lines)


def comparison_to_markdown(a: EvalReport, b: EvalReport) -> str:
    diff = compare(a, b)
    lines = [
        f"### Comparison: {a.label} vs {b.label}",
        "",
        f"| Metric | {a.label} | {b.label} | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for metric in _METRIC_ORDER:
        if metric not in diff:
            continue
        d = diff[metric]
        label = _METRIC_LABEL.get(metric, metric)
        lines.append(
            f"| {label} | {d[a.label]:.0%} | {d[b.label]:.0%} | "
            f"{d['delta']:+.0%} |"
        )
    return "\n".join(lines)
