"""Exact text alignment; offsets refer to comparison strings, not image coordinates."""
from __future__ import annotations

import unicodedata


class AlignmentLimitError(ValueError):
    """The exact alignment exceeds the configured CPU/memory budget."""


def comparison_text(text: str, mode: str = "visible") -> str:
    if mode not in {"han", "visible"}:
        raise ValueError("mode must be han or visible")
    if mode == "visible":
        return "".join(c for c in text if not c.isspace())
    return "".join(c for c in text if ord(c) >= 0x3400 and unicodedata.name(c, "").startswith(
        ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")))


def align_reference_lines(reference, candidate, lines, *, max_cells=8_000_000):
    """Constrain star alignment with unique text anchors inside reference lines.

    This is a text heuristic, not geometric correspondence. Keep pairwise edit
    distances separate: these forced anchors need not give a global minimum.
    Ambiguous anchors, reordered lines, or weak evidence use the exact fallback.
    """
    filtered = [comparison_text(line) for line in lines]
    if "".join(filtered) != reference:
        return None, {"status": "fallback", "reason": "line_text_mismatch"}
    anchors, offset = [], 0
    for line in filtered:
        match = None
        for width in range(min(8, len(line)), 3, -1):
            for start in range(len(line) - width + 1):
                token = line[start:start + width]
                if not all(c.isalnum() for c in token):
                    continue
                left, right = reference.find(token), candidate.find(token)
                if (left == offset + start and reference.find(token, left + 1) < 0 and
                        right >= 0 and candidate.find(token, right + 1) < 0):
                    match = (left, right, width)
                    break
            if match:
                break
        if match:
            anchors.append(match)
        offset += len(line)
    # Longest monotone chain; never force anchors that cross between models.
    chains = []
    for i, anchor in enumerate(anchors):
        previous = [chains[j] for j in range(i) if anchors[j][1] + anchors[j][2] <= anchor[1]]
        chains.append(max(previous, key=len, default=[]) + [anchor])
    selected = max(chains, key=len, default=[])
    if len(selected) < 2 or len(selected) < .75 * len(anchors):
        return None, {"status": "fallback", "reason": "insufficient_monotone_line_anchors",
                      "anchors": len(selected)}
    aligned, left, right = [], 0, 0
    for x, y, width in selected:
        aligned.extend(align_characters(reference[left:x], candidate[right:y], max_cells=max_cells))
        aligned.extend((c, c) for c in reference[x:x + width])
        left, right = x + width, y + width
    aligned.extend(align_characters(reference[left:], candidate[right:], max_cells=max_cells))
    return aligned, {"status": "anchored", "anchors": len(selected), "segments": len(selected) + 1}


def align_characters(left: str, right: str, *, max_cells: int = 8_000_000) -> list[tuple[str, str]]:
    """Unit-cost Levenshtein alignment; an empty string denotes an alignment gap.

    Trim equal ends, keep only two cost rows and one traceback byte per cell.
    Equally optimal paths prefer diagonal, deletion, then insertion. Repeated
    characters can admit multiple optimal alignments; positions are indicative.
    """
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    end_l, end_r = len(left), len(right)
    while end_l > prefix and end_r > prefix and left[end_l - 1] == right[end_r - 1]:
        end_l -= 1
        end_r -= 1
    a, b = left[prefix:end_l], right[prefix:end_r]
    n, m = len(a), len(b)
    if n * m > max_cells:
        raise AlignmentLimitError(f"{n * m:,} alignment cells exceed max_cells={max_cells:,}")
    if not a or not b:
        middle = [(c, "") for c in a] + [("", c) for c in b]
    else:
        trace = [bytearray(m + 1) for _ in range(n + 1)]
        previous = list(range(m + 1))
        for i, ca in enumerate(a, 1):
            current = [i] + [0] * m
            for j, cb in enumerate(b, 1):
                diagonal = previous[j - 1] + (ca != cb)
                deletion, insertion = previous[j] + 1, current[j - 1] + 1
                best = min(diagonal, deletion, insertion)
                current[j] = best
                trace[i][j] = 0 if best == diagonal else 1 if best == deletion else 2
            previous = current
        middle = []
        i, j = n, m
        while i or j:
            direction = trace[i][j] if i and j else 1 if i else 2
            if direction == 0:
                middle.append((a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
            elif direction == 1:
                middle.append((a[i - 1], ""))
                i -= 1
            else:
                middle.append(("", b[j - 1]))
                j -= 1
        middle.reverse()
    return ([(c, c) for c in left[:prefix]] + middle + [(c, c) for c in left[end_l:]])
