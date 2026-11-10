"""Page structure evidence before character voting; no OCR output is rewritten.

Unique text anchors locate spans regardless of line wrapping or reading order.
Strict majorities count ALL configured engines. Geometry remains model-estimated
context, never proof that a majority is correct or that text-only spans are columns.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from .character_regions import normalized_polygon
from .ocr_worker import md_to_lines, ordered_page_blocks
from .text_alignment import comparison_text


PAGE_CONSENSUS_DEFAULTS = {
    "anchor_chars": 4, "anchors_per_region": 10, "min_region_chars": 12,
    "max_unit_chars": 80, "max_text_chars": 20000, "max_regions": 256,
    "max_anchor_occurrences": 8,
    "max_global_anchors": 64,
}

PAGE_FINDING_LABELS = {
    "suspected_missing_region": "Nghi thiếu vùng/cột",
    "suspected_duplicate_region": "Nghi lặp vùng/cột",
    "suspected_reading_order": "Nghi sai thứ tự đọc",
    "suspected_extra_region": "Văn bản dư so với đa số",
    "region_text_mismatch": "Có khung vùng nhưng chữ không khớp",
    "unmatched_region_text": "Chữ không khớp; chưa phân biệt thay thế hay dư/thiếu",
}


def _page_location(box, raw, method, granularity):
    location = normalized_polygon(box, raw.get("w", 0), raw.get("h", 0))
    return ({**location, "method": method, "granularity": granularity, "estimated": True}
            if location else None)


def _page_units(record, engine, options):
    """Preserve stream offsets and explicit granularity; no guessed glyph boxes."""
    text = comparison_text(record["text"])
    raw = record.get("geometry", record.get("raw", {}))
    lines = raw.get("text_lines", [])
    entries, method, granularity = [], "text_lines", "text_span"
    if lines and "".join(comparison_text(e.get("text", "")) for e in lines) == text:
        for entry in lines:
            location = (_page_location(entry.get("polygon"), raw, "ocr_line", "line_context")
                        if raw.get("coordinate_frame") == "input_image" else None)
            entries.append((comparison_text(entry.get("text", "")), location))
        method, granularity = "ocr_lines", "line_context"
    elif raw.get("blocks"):
        blocks = ordered_page_blocks(raw["blocks"])
        values = [comparison_text("\n".join(md_to_lines(e.get("text", "")))) for e in blocks]
        if "".join(values) == text:
            entries = [(value, _page_location(entry.get("box"), raw, "ocr_block", "block_context")
                        if raw.get("coordinate_frame") == "input_image" else None)
                       for value, entry in zip(values, blocks)]
            method, granularity = "ocr_blocks", "block_context"
    if not entries:
        entries = [(comparison_text(line), None) for line in record["text"].splitlines()]
        method, granularity = "text_lines", "text_span"
    # Fall back to the exact comparison stream if a backend has unusual newlines.
    if "".join(value for value, _ in entries) != text:
        entries = [(text, None)]
    units, offset = [], 0
    for value, location in entries:
        for start in range(0, len(value), options["max_unit_chars"]):
            stop = min(len(value), start+options["max_unit_chars"])
            units.append({"start": offset+start, "stop": offset+stop, "location": location,
                          "method": method, "granularity": granularity})
        offset += len(value)
    return units


def _page_anchor_index(text, width, maximum):
    positions, counts = defaultdict(list), Counter()
    for offset in range(len(text)-width+1):
        token = text[offset:offset+width]
        if not token.isalnum():
            continue
        counts[token] += 1
        if len(positions[token]) < maximum:
            positions[token].append(offset)
    return {"positions": positions, "counts": counts}


def _page_anchors(text, start, stop, indexes, model, voters, options):
    width = options["anchor_chars"]
    own = indexes[model]
    candidates = []
    for offset in range(start, stop-width+1):
        token = text[offset:offset+width]
        if own["counts"].get(token) != 1:
            continue  # Repeated source phrases cannot anchor a unique region.
        support = sum(bool(indexes[name]["counts"].get(token)) for name in voters)
        candidates.append((offset, token, support))
    # Spread anchors over the span, preferring corroborated tokens within each bin.
    bins = defaultdict(list)
    for entry in candidates:
        bins[min(options["anchors_per_region"]-1,
                 (entry[0]-start)*options["anchors_per_region"]//max(1,stop-start))].append(entry)
    return [{"offset": offset, "text": token} for offset, token, _ in
            (max(values, key=lambda e: (e[2], -e[0])) for _, values in sorted(bins.items()))]


def _page_span_matches(anchors, start, stop, index, options):
    """Cluster independently located anchors into copies of a region."""
    width = options["anchor_chars"]
    if any(index["counts"].get(a["text"], 0) > options["max_anchor_occurrences"] for a in anchors):
        return {"status": "ambiguous", "reason": "anchor_occurrence_limit", "matches": []}
    hits = sorted((position-(anchor["offset"]-start), number, position)
                  for number, anchor in enumerate(anchors)
                  for position in index["positions"].get(anchor["text"], []))
    if not hits:
        return {"status": "absent", "copies": 0, "matches": [], "anchor_hits": 0}
    tolerance = max(3, (stop-start)*.15)
    groups = []
    for hit in hits:
        if not groups or hit[0]-groups[-1][0][0] > tolerance:
            groups.append([hit])
        else:
            groups[-1].append(hit)
    matches = []
    for group in groups:
        numbers = sorted({entry[1] for entry in group})
        if len(numbers) < 2:
            continue
        source_span = anchors[numbers[-1]]["offset"]+width-anchors[numbers[0]]["offset"]
        if source_span < min(12, (stop-start)*.65) or source_span < .65*(stop-start):
            continue
        left, right = min(e[2] for e in group), max(e[2] for e in group)+width
        # Overlapping anchor groups are not independent repeated columns.
        if matches and left < matches[-1]["span"][1]:
            continue
        matches.append({"span": [left,right], "anchors": len(numbers),
                        "source_anchor_coverage": round(source_span/(stop-start),4)})
    return ({"status": "present", "copies": len(matches), "matches": matches, "anchor_hits": len(hits)}
            if matches else {"status": "ambiguous", "reason": "insufficient_coherent_anchors",
                             "matches": [], "anchor_hits": len(hits)})


def _page_unclaimed(start, stop, claimed):
    beginning = None
    for index in range(start, stop+1):
        free = index < stop and not claimed[index]
        if free and beginning is None:
            beginning = index
        elif not free and beginning is not None:
            yield beginning, index
            beginning = None


def _page_matching_location(units, span):
    selected = [u["location"] for u in units if u.get("location")
                and max(u["start"],span[0]) < min(u["stop"],span[1])]
    if not selected:
        return None
    box = [min(e["bbox"][0] for e in selected), min(e["bbox"][1] for e in selected),
           max(e["bbox"][2] for e in selected), max(e["bbox"][3] for e in selected)]
    return {"bbox": box, "method": "ocr_region_context", "granularity": "region_context", "estimated": True}


def _page_spatial_counterpart(location, units):
    """Only compare similar-sized detected contexts, never a whole-page block."""
    if not location:
        return None
    a = location["bbox"]
    area = (a[2]-a[0])*(a[3]-a[1])
    for unit in units:
        other = unit.get("location")
        if not other:
            continue
        b = other["bbox"]
        other_area = (b[2]-b[0])*(b[3]-b[1])
        intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        if .5 <= other_area/area <= 2 and intersection/(area+other_area-intersection) >= .5:
            return other
    return None


def _page_order_sign(first, second):
    if len(first.get("matches", [])) != 1 or len(second.get("matches", [])) != 1:
        return None
    a, b = first["matches"][0]["span"], second["matches"][0]["span"]
    return 1 if a[1] <= b[0] else -1 if b[1] <= a[0] else None


def _page_flanking_gap(text, start, stop, own, target, target_length, width):
    """Text insertion/deletion evidence requires unique flanks or page edges."""
    left, right = (0 if start == 0 else None), (target_length if stop == len(text) else None)
    flanks = []
    for side, candidates in (("left", range(start-width,max(-1,start-80-width),-1)),
                             ("right", range(stop,min(len(text)-width+1,stop+80)))):
        if (side == "left" and start == 0) or (side == "right" and stop == len(text)):
            continue
        for offset in candidates:
            if offset < 0 or offset+width > len(text):
                continue
            token = text[offset:offset+width]
            if own['counts'].get(token) == target['counts'].get(token) == 1:
                position = target['positions'][token][0]
                if side == "left":
                    left = position+width
                else:
                    right = position
                flanks.append({"side": side, "text": token, "reference_offset": offset,
                               "target_offset": position})
                break
    return {"status": "anchored" if left is not None and right is not None and left <= right else "unresolved",
            "gap_span": [left,right], "gap_chars": right-left if left is not None and right is not None and left <= right else None,
            "anchors": flanks}


def _page_global_order_events(texts,indexes,voters,quorum,raw_indexes,options):
    """Detect long-range inversions even when every backend emits one paragraph."""
    reference = max(voters,key=lambda name: len(texts[name]))
    width = options['anchor_chars']
    candidates = []
    previous = -width
    for token,positions in indexes[reference]['positions'].items():
        position = positions[0]
        if indexes[reference]['counts'][token] != 1 or position < previous+width:
            continue
        if sum(indexes[name]['counts'].get(token)==1 for name in voters) >= quorum:
            candidates.append((position,token))
            previous = position
    if len(candidates) > options['max_global_anchors']:
        count = options['max_global_anchors']
        candidates = [candidates[round(i*(len(candidates)-1)/max(1,count-1))] for i in range(count)]
    events = defaultdict(list)
    for (x,first),(y,second) in combinations(candidates,2):
        if y-x < options['min_region_chars']:
            continue
        signs,positions = {},{}
        for name,index in indexes.items():
            a,b = index['positions'].get(first,[]),index['positions'].get(second,[])
            if index['counts'].get(first)==index['counts'].get(second)==1:
                signs[name] = 1 if a[0] < b[0] else -1
                positions[name] = [a[0],b[0]]
        counts = Counter(signs[name] for name in voters if name in signs)
        winner = next((sign for sign,n in counts.items() if n >= quorum),None)
        if winner is None:
            continue
        supporters = [name for name in voters if signs.get(name)==winner]
        for name,sign in signs.items():
            if sign != winner and abs(positions[name][0]-positions[name][1]) >= options['min_region_chars']:
                events[(name,'comparison')].append({'regions':[], 'supporters':supporters,
                    'anchors':[first,second], 'positions':positions})
        for name,index in raw_indexes.items():
            if index['counts'].get(first)==index['counts'].get(second)==1:
                a,b = index['positions'][first][0],index['positions'][second][0]
                if (1 if a < b else -1) != winner and abs(a-b) >= options['min_region_chars']:
                    independent = [peer for peer in supporters if peer != name]
                    if len(independent) >= quorum:
                        events[(name,'raw_before_normalization')].append({'regions':[], 'supporters':independent,
                            'anchors':[first,second], 'positions':{**positions,name:[a,b]}})
    # One inverted short phrase is insufficient to diagnose page reading order.
    return {key:values for key,values in events.items() if len(values) >= 2
            and len({token for event in values for token in event['anchors']}) >= 4}


def page_level_consensus(records, *, excluded_models=None, source_verified=False, options=None):
    settings = {**PAGE_CONSENSUS_DEFAULTS, **(options or {})}
    if set(settings) != set(PAGE_CONSENSUS_DEFAULTS):
        raise ValueError("Unknown page consensus options")
    if any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in settings.values()):
        raise ValueError("Page consensus limits must be positive integers")
    if settings["anchors_per_region"] < 2 or settings["min_region_chars"] < 2*settings["anchor_chars"]:
        raise ValueError("Page consensus requires at least two nonoverlapping anchors per region")
    if settings["max_unit_chars"] < settings["min_region_chars"]:
        raise ValueError("max_unit_chars must be >= min_region_chars")
    if len(records) < 2:
        raise ValueError("At least two configured models are required")
    excluded = excluded_models or {}
    texts = {name: comparison_text(rec["text"]) if rec is not None else None for name,rec in records.items()}
    voters = [name for name,text in texts.items() if text and name not in excluded]
    quorum = len(records)//2+1
    result = {"schema": "vtm-page-consensus-v1", "status": "insufficient_evidence", "complete": True,
        "quorum": quorum, "configured_models": len(records), "voters": voters,
        "source_verified": bool(source_verified), "regions": [], "findings": [], "settings": settings,
        "models": {name: {"status": "missing_output" if text is None else "empty_output" if not text else "unassessed",
                          "flags": [], "comparison_chars": len(text) if text is not None else None,
                          "reading_order_normalization": (records[name] or {}).get("reading_order", {}),
                          "excluded_from_region_votes": name in excluded}
                   for name,text in texts.items()},
        "limitations": ["Text-anchor consensus is not OCR accuracy or ground truth.",
            "Text-only spans do not establish a source-image column or a hallucination.",
            "All models may omit the same content; shared omissions are not detectable here."]}
    if any(text is not None and len(text) > settings["max_text_chars"] for text in texts.values()):
        result.update(status="analysis_limit", complete=False,limit_reason="max_text_chars")
        return result
    if len(voters) < quorum:
        return result
    units = {name: _page_units(rec,name,settings) if rec is not None else [] for name,rec in records.items()}
    indexes = {name: _page_anchor_index(text or "",settings["anchor_chars"],settings["max_anchor_occurrences"])
               for name,text in texts.items()}
    claimed = {name: bytearray(len(text or "")) for name,text in texts.items()}
    # Prefer detailed measured lines, while allowing other models to introduce
    # regions omitted by that detector. A reference model is never ground truth.
    proposals = sorted(voters, key=lambda name: (-sum(bool(u["location"]) for u in units[name]),
        -sum(u['stop']-u['start'] >= settings['min_region_chars'] for u in units[name]), name))
    for reference in proposals:
        for unit in units[reference]:
            for start, stop in list(_page_unclaimed(unit["start"],unit["stop"],claimed[reference])):
                if stop-start < settings["min_region_chars"]:
                    continue
                anchors = _page_anchors(texts[reference],start,stop,indexes,reference,voters,settings)
                if len(anchors) < 2:
                    continue
                matches = {name: _page_span_matches(anchors,start,stop,index,settings) if texts[name] is not None
                           else {"status": "missing_output", "matches": []} for name,index in indexes.items()}
                if matches[reference]["status"] != "present":
                    continue
                if len(result["regions"]) >= settings["max_regions"]:
                    result.update(status="analysis_limit", complete=False,limit_reason="max_regions")
                    break
                location = unit["location"]
                region = {"id": len(result["regions"])+1, "reference_model": reference,
                    "reference_span": [start,stop], "text": texts[reference][start:stop],
                    "granularity": unit["granularity"], "location": location,
                    "localization_status": "model_estimated_context" if location else "unresolved",
                    "anchors": anchors, "models": matches}
                counts = Counter(matches[name].get("copies") for name in voters if "copies" in matches[name])
                winners = [(count,n) for count,n in counts.items() if n >= quorum]
                region["majority_copies"] = winners[0][0] if winners else None
                region["supporters"] = ([name for name in voters if matches[name].get("copies") == winners[0][0]]
                                         if winners else [])
                for name, match in matches.items():
                    for found in match["matches"]:
                        found["location"] = _page_matching_location(units[name],found["span"])
                        left,right = found["span"]
                        claimed[name][left:right] = b"\1"*(right-left)
                claimed[reference][start:stop] = b"\1"*(stop-start)
                result["regions"].append(region)
            if not result["complete"]:
                break
        if not result["complete"]:
            break
    def finding(model, kind, region_ids, supporters, **evidence):
        result["findings"].append({"model": model, "kind": kind, "label": PAGE_FINDING_LABELS[kind],
            "suspected": True, "region_ids": region_ids, "supporters": supporters,
            "evidence_level": "text_anchors", "source_verified": bool(source_verified), **evidence})
    for region in result["regions"]:
        expected = region["majority_copies"]
        if expected is None:
            continue
        for name, match in region["models"].items():
            copies = match.get("copies")
            if copies is None or copies == expected:
                continue
            gap = _page_flanking_gap(texts[region['reference_model']],*region['reference_span'],
                indexes[region['reference_model']], indexes[name],len(texts[name] or ''),settings['anchor_chars'])
            # For an extra span the majority-absent peers, not the span's owner,
            # must have near-adjacent flanks. A substitution is not an insertion.
            if expected == 0:
                peer_gaps = {peer: _page_flanking_gap(texts[region['reference_model']],*region['reference_span'],
                    indexes[region['reference_model']],indexes[peer],len(texts[peer] or ''),settings['anchor_chars'])
                    for peer in region['supporters']}
                flanking_support = [peer for peer, evidence in peer_gaps.items() if evidence['gap_chars'] is not None
                                    and evidence['gap_chars'] <= max(2,len(region['text'])*.2)]
            else:
                peer_gaps, flanking_support = {}, []
            if expected == 0 and copies > 0:
                kind = "suspected_extra_region" if len(flanking_support) >= quorum else "unmatched_region_text"
            elif expected > 0 and copies == 0:
                counterpart = (_page_spatial_counterpart(region["location"],units[name]) if source_verified else None)
                deletion = gap['gap_chars'] is not None and gap['gap_chars'] <= max(2,len(region['text'])*.2)
                kind = ("region_text_mismatch" if counterpart else "suspected_missing_region"
                        if deletion or not texts[name] else "unmatched_region_text")
                match["spatial_counterpart"] = counterpart
            elif copies > expected:
                kind = "suspected_duplicate_region"
            else:
                kind = "suspected_missing_region"
            finding(name,kind,[region["id"]],region["supporters"],expected_copies=expected,
                    observed_copies=copies,stream="comparison",flanking_gap=gap,peer_flanking_gaps=peer_gaps)
    # A pairwise order majority is defined only for non-overlapping, unique spans.
    order_events = defaultdict(list)
    supported = [r for r in result["regions"] if r["majority_copies"] == 1]
    raw_indexes = {name: _page_anchor_index(comparison_text(rec["raw"].get("text", "")),
                         settings["anchor_chars"],settings["max_anchor_occurrences"])
                   for name,rec in records.items() if rec is not None
                   and rec.get("reading_order",{}).get("changed")}
    raw_matches = {name: {r["id"]: _page_span_matches(r["anchors"],*r["reference_span"],index,settings)
                          for r in supported} for name,index in raw_indexes.items()}
    for first,second in combinations(supported,2):
        signs = {name: _page_order_sign(first["models"][name],second["models"][name]) for name in voters}
        frequencies = Counter(sign for sign in signs.values() if sign is not None)
        winner = next((sign for sign,n in frequencies.items() if n >= quorum),None)
        if winner is None:
            continue
        supporters = [name for name,sign in signs.items() if sign == winner]
        for name in records:
            sign = _page_order_sign(first["models"][name],second["models"][name])
            if sign is not None and sign != winner:
                order_events[(name,"comparison")].append({"regions": [first["id"],second["id"]], "supporters": supporters})
            if name in raw_matches:
                raw_sign = _page_order_sign(raw_matches[name][first["id"]],raw_matches[name][second["id"]])
                if raw_sign is not None and raw_sign != winner:
                    independent = [peer for peer in supporters if peer != name]
                    if len(independent) >= quorum:
                        order_events[(name,"raw_before_normalization")].append({"regions": [first["id"],second["id"]], "supporters": independent})
    for key,events in _page_global_order_events(texts,indexes,voters,quorum,raw_indexes,settings).items():
        order_events[key].extend(events)
    for (name,stream), events in sorted(order_events.items()):
        finding(name,"suspected_reading_order",sorted({r for e in events for r in e["regions"]}),
                sorted({m for e in events for m in e["supporters"]}),stream=stream,
                inverted_pairs=events[:32],inverted_pair_count=len(events),
                corrected_before_character_consensus=stream=="raw_before_normalization")
    for item in result["findings"]:
        summary = result["models"][item["model"]]
        summary["flags"] = sorted(set([*summary["flags"],item["kind"]]))
    for name,summary in result["models"].items():
        if texts[name]:
            summary["status"] = ("review" if summary["flags"] else "assessed" if any(
                r["majority_copies"] is not None and r["models"][name].get("copies") is not None
                for r in result["regions"]) else "unassessed")
        summary['anchored_char_fraction'] = (round(sum(bool(v) for v in claimed[name])/len(texts[name]),4)
                                              if texts[name] else None)
    result['unresolved_regions'] = [r['id'] for r in result['regions'] if r['majority_copies'] is None
        or any(m['status'] == 'ambiguous' for m in r['models'].values())]
    different = len({text for text in texts.values() if text is not None}) > 1
    if result["complete"]:
        result["status"] = ("structural_review" if result["findings"] else
                            "unresolved_structure" if different and (result['unresolved_regions'] or any(
                                (model.get('anchored_char_fraction') or 0) < .8 for model in result['models'].values())) else
                            "no_structural_disagreement_detected" if any(r["majority_copies"] == 1 for r in result["regions"])
                            else "insufficient_evidence")
    return result


def page_consensus_exclusions(diagnostic):
    """Only uncorrected, majority-backed structural findings abstain from votes."""
    if not diagnostic["complete"]:
        return {}
    excluded = defaultdict(set)
    for finding in diagnostic["findings"]:
        if finding.get("corrected_before_character_consensus"):
            continue
        if finding['kind'] in {'unmatched_region_text','region_text_mismatch'}:
            continue  # Substitutions/uncertain boundaries belong to character review.
        excluded[finding["model"]].add("page_"+finding["kind"])
    return {name: sorted(issues) for name,issues in excluded.items()}


def page_consensus_summary(diagnostic):
    return "; ".join(f"{name}: " + ", ".join(PAGE_FINDING_LABELS[kind] for kind in model["flags"])
                     for name,model in diagnostic.get("models",{}).items() if model["flags"])


def page_structure_overview(image, diagnostic):
    """Draw only measured reference contexts; never create boxes from text spans."""
    from PIL import ImageDraw
    numbers = {number for finding in diagnostic['findings'] for number in finding['region_ids']}
    regions = [r for r in diagnostic['regions'] if r['id'] in numbers and r['location']]
    if not regions:
        return None
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    for region in regions:
        x1,y1,x2,y2 = region['location']['bbox']
        box = [round(x1*image.width),round(y1*image.height),round(x2*image.width),round(y2*image.height)]
        draw.rectangle(box,outline='red',width=max(1,image.width//700))
        draw.text((box[0]+2,box[1]+2),str(region['id']),fill='red')
    return overview
