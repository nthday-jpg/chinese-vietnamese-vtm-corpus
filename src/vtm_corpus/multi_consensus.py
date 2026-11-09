"""Conservative, reference-anchored multi-OCR alignment and character votes."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import log2

from .text_alignment import AlignmentLimitError, align_characters, align_reference_lines, comparison_text


def _pair_map(reference, candidate, max_cells, *, alignment=None):
    insertions = [""] * (len(reference) + 1)
    characters = [""] * len(reference)
    index = 0
    for a, b in (alignment if alignment is not None else align_characters(reference, candidate, max_cells=max_cells)):
        if a:
            characters[index] = b
            index += 1
        else:
            insertions[index] += b
    return insertions, characters


def _star_columns(texts, reference, max_cells):
    """Align each sequence to a fixed anchor; merge its insertion slots."""
    maps = {name: _pair_map(texts[reference], text, max_cells) for name, text in texts.items()}
    columns = []
    for slot in range(len(texts[reference]) + 1):
        width = max(len(value[0][slot]) for value in maps.values())
        for index in range(width):
            columns.append({name: value[0][slot][index:index + 1] for name, value in maps.items()})
        if slot < len(texts[reference]):
            columns.append({name: value[1][slot] for name, value in maps.items()})
    return columns


def multi_character_consensus(candidates: dict[str, str | None], *, reference="ppocr",
                              mode="visible", max_cells=8_000_000, excluded_models=None,
                              reference_lines=None) -> dict:
    """Keep missing engines distinct from a real empty output and alignment gaps.

    This is star alignment, not a globally optimal multiple-sequence alignment.
    Insertions are aligned locally to their longest candidate before merging.
    A strict majority requires > half of ALL configured models; a missing model
    never votes. Votes are evidence, not calibrated probabilities or truth.
    """
    if len(candidates) < 2:
        raise ValueError("At least two configured models are required")
    texts = {name: comparison_text(text, mode) if text is not None else None
             for name, text in candidates.items()}
    excluded = dict(excluded_models or {})
    if set(excluded) - candidates.keys():
        raise ValueError("Excluded models must be configured candidates")
    available = {name: text for name, text in texts.items() if text is not None and name not in excluded}
    result = {"models": list(candidates), "texts": texts, "reference": reference,
              "alignment": "star", "status": "missing_output", "columns": [],
              "conflicts": [], "candidate_scores": {}, "pairwise_distances": {},
              "missing_models": [name for name, text in texts.items() if text is None],
              "excluded_models": excluded,
              "line_alignment": {},
              "mean_entropy_bits": None, "disagreement_pct": None}
    align_characters("", "", max_cells=max_cells)
    if not available or (excluded and len(available) < 2):
        if excluded:
            result["status"] = "quality_review"
        return result
    if reference not in available or not available[reference]:
        reference = max(available, key=lambda name: len(available[name]))
    result["reference"] = reference
    if not any(available.values()):
        result["status"] = "both_empty" if len(available) == 2 else "all_empty"
        return result
    distances = {name: [] for name in available}
    reference_alignments = {}
    try:
        for a, b in combinations(available, 2):
            # Distances are symmetric, but tie-breaking alignment paths are not.
            # Use the reference on the left so its map exactly matches the
            # original star-alignment semantics without computing it twice.
            left, right = (b, a) if b == reference else (a, b)
            aligned = align_characters(available[left], available[right], max_cells=max_cells)
            if left == reference:
                reference_alignments[right] = aligned
            denominator = max(len(available[a]), len(available[b]))
            distance = sum(x != y for x, y in aligned) / denominator if denominator else None
            result["pairwise_distances"][f"{a}|{b}"] = distance
            if distance is not None:
                distances[a].append(distance)
                distances[b].append(distance)
        if reference_lines:
            for name in available:
                if name == reference:
                    continue
                anchored, evidence = align_reference_lines(available[reference], available[name], reference_lines,
                                                            max_cells=max_cells)
                result["line_alignment"][name] = evidence
                if anchored is not None:
                    reference_alignments[name] = anchored
                    result["alignment"] = "star_reference_line_anchors"
        maps = {reference: ([""] * (len(available[reference]) + 1), list(available[reference]))}
        maps.update({name: _pair_map(available[reference], text, max_cells,
                                    alignment=reference_alignments[name])
                     for name, text in available.items() if name != reference})
        aligned_columns = []
        for slot in range(len(available[reference]) + 1):
            inserted = {name: value[0][slot] for name, value in maps.items()}
            if any(inserted.values()):
                longest = max(inserted, key=lambda name: len(inserted[name]))
                aligned_columns.extend(_star_columns(inserted, longest, max_cells))
            if slot < len(available[reference]):
                aligned_columns.append({name: value[1][slot] for name, value in maps.items()})
    except AlignmentLimitError as exc:
        result.update(status="alignment_limit", alignment_error=str(exc), pairwise_distances={})
        return result
    result["candidate_scores"] = {name: sum(values) / len(values) if values else None
                                  for name, values in distances.items()}
    offsets = dict.fromkeys(available, 0)
    for index, values in enumerate(aligned_columns):
        votes = Counter(values.values())
        best, support = votes.most_common(1)[0]
        majority = best if support > len(candidates) / 2 else None
        entropy = -sum((n / len(available)) * log2(n / len(available)) for n in votes.values())
        column = {"index": index, "predictions": {name: values.get(name) for name in candidates},
                  "offsets": {name: offsets.get(name) if values.get(name) else None for name in candidates},
                  "votes": dict(votes), "entropy_bits": entropy,
                  "majority": majority, "support": support, "configured_models": len(candidates),
                  "available_models": len(available), "disagreement": len(votes) > 1,
                  "localization": None}
        result["columns"].append(column)
        if column["disagreement"]:
            result["conflicts"].append(column)
        for name, value in values.items():
            offsets[name] += len(value)
    result.update(status="partial" if result["missing_models"] or excluded else "compared",
                  mean_entropy_bits=sum(c["entropy_bits"] for c in result["columns"]) / len(aligned_columns),
                  disagreement_pct=100 * len(result["conflicts"]) / len(aligned_columns))
    return result
