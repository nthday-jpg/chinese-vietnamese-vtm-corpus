"""Conservative OCR quality flags; retain flagged output but abstain from votes."""
from __future__ import annotations

from collections import Counter
import re
from statistics import median

from .text_alignment import comparison_text


def ocr_quality_issues(text, raw):
    issues = list(raw.get("quality_issues", []))
    if raw.get("hit_max"):
        issues.append("token_limit")
    visible = comparison_text(text)
    # Long runs anywhere, including a damaged final token and repeated prefixes.
    if re.search(r"(.{1,40}?)\1{11,}", visible):
        match = max((m.end() - m.start() for m in re.finditer(r"(.{1,40}?)\1{11,}", visible)), default=0)
        if match >= 80:
            issues.append("repetitive_output")
    if len(visible) >= 80:
        lines = [comparison_text(line) for line in text.splitlines()]
        # Numbered hallucinated rows can repeat while the number changes.
        signatures = [re.sub(r"[0-9０-９]+", "0", line) for line in lines if len(line) >= 12]
        if signatures and max(Counter(signatures).values()) >= 6:
            issues.append("repetitive_output")
        symbols = sum(not c.isalnum() for c in visible)
        if symbols >= 80 and symbols / len(visible) > .5:
            issues.append("symbol_dominated_output")
    return sorted(set(issues))


def page_quality_issues(records):
    issues = {name: list(record.get("quality_issues", [])) for name, record in records.items() if record is not None}
    lengths = {name: record["visible"] for name, record in records.items()
               if record is not None and not issues[name] and record["visible"] > 0}
    # Only flag an outlier when at least two other healthy engines corroborate it.
    for name, length in lengths.items():
        peers = [value for other, value in lengths.items() if other != name]
        if len(peers) < 2 or max(peers) > 1.6 * min(peers):
            continue
        typical = median(peers)
        if typical >= 80 and length < .35 * typical:
            issues[name].append("suspected_incomplete_output")
        elif length > max(3 * typical, typical + 200):
            issues[name].append("length_outlier")
    return {name: sorted(set(reasons)) for name, reasons in issues.items() if reasons}
