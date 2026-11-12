"""Read matched four-engine checkpoints and retain per-character visual evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .character_regions import attach_character_regions, attach_conflict_regions
from .ocr_records import read_json, load_run, check_sources, read_checkpoint_campaign
from .multi_consensus import multi_character_consensus
from .ocr_quality import page_quality_issues
from .page_consensus import page_level_consensus, page_consensus_exclusions, page_consensus_summary


def compare_multi_corpus(manifest_path, *, max_alignment_cells=8_000_000, review_threshold=.05,
                         reading_order="auto", page_consensus=True, page_consensus_options=None,
                         source_dir=None) -> dict:
    if not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold must be between 0 and 1")
    if reading_order not in {"auto", "vertical_rtl", "detection"}:
        raise ValueError("reading_order must be auto, vertical_rtl or detection")
    campaign = read_checkpoint_campaign(manifest_path)
    models = campaign["models"]
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("Campaign must name at least two distinct models")
    rows, warnings, configs, seen, digest_cache = [], [], {}, set(), {}
    for pair in campaign["pairs"]:
        runs = {name: load_run(Path(pair[name]), name, warnings, reading_order=reading_order) for name in models}
        if source_dir:
            for run in runs.values():
                for info in run['manifest'].get('books', {}).values():
                    filename = str(info.get('src', '')).replace('\\', '/').rsplit('/', 1)[-1]
                    candidates = list(Path(source_dir).rglob(filename)) if filename else []
                    if len(candidates) > 1:
                        raise ValueError('Ambiguous relocated source PDF: '+filename)
                    if candidates:
                        info['src'] = str(candidates[0].resolve())
        first = runs[models[0]]
        source_checks = [check_sources(first, runs[name], warnings, digest_cache=digest_cache) for name in models[1:]]
        verified = {check["book"]: all(
            any(c["book"] == check["book"] and c["source_verified"] for c in group)
            for group in source_checks) for group in source_checks for check in group}
        keys = set().union(*(run["selected"] for run in runs.values()))
        for book, page in sorted(keys):
            if (book, page) in seen:
                raise ValueError(f"Duplicate book/page: {book}, {page + 1}")
            seen.add((book, page))
            records = {name: run["records"].get((book, page)) if (book, page) in run["selected"] else None
                       for name, run in runs.items()}
            candidates = {name: r["text"] if r is not None else None for name, r in records.items()}
            quality = page_quality_issues(records)
            page_diagnostic = (page_level_consensus(records, excluded_models=quality,
                source_verified=bool(verified.get(book)), options=page_consensus_options)
                if page_consensus else None)
            if page_diagnostic:
                for name,issues in page_consensus_exclusions(page_diagnostic).items():
                    quality[name] = sorted(set(quality.get(name, [])+issues))
            locator = records.get("ppocr")
            reference_lines = ([line["text"] for line in locator["geometry"].get("text_lines", [])]
                               if locator and locator["reading_order"]["mode"] == "vertical_rtl" and "ppocr" not in quality else None)
            consensus = multi_character_consensus(candidates, max_cells=max_alignment_cells, excluded_models=quality,
                                                   reference_lines=reference_lines)
            attach_character_regions(consensus, records)
            attach_conflict_regions(consensus, records)
            source = first["manifest"].get("books", {}).get(book, {}).get("src")
            reasons = []
            if page_diagnostic and page_diagnostic['findings']:
                reasons.append("page_structure_disagreement")
            if page_diagnostic and not page_diagnostic['complete']:
                reasons.append("page_structure_analysis_limit")
            if page_diagnostic and page_diagnostic['status'] == 'unresolved_structure':
                reasons.append("page_structure_unresolved")
            if consensus["missing_models"]:
                reasons.append("missing_models")
            if quality:
                reasons.append("quality_excluded_models")
                reasons.extend(sorted({issue for issues in quality.values() for issue in issues}))
            if not verified.get(book):
                reasons.append("source_unverified")
            if any((r or {}).get("hit_max") for r in records.values()):
                if "token_limit" not in reasons:
                    reasons.append("token_limit")
            if consensus["status"] in {"alignment_limit", "all_empty", "both_empty", "missing_output", "quality_review"}:
                reasons.append(consensus["status"])
            if consensus["conflicts"]:
                reasons.append("character_conflicts")
            if any(c["localization"] is None for c in consensus["conflicts"]):
                reasons.append("unresolved_character_location")
            pair_scores = [d for d in consensus["pairwise_distances"].values() if d is not None]
            max_distance = max(pair_scores) if pair_scores else None
            if max_distance is not None and max_distance > review_threshold + 1e-12:
                reasons.append("above_review_threshold")
            severe = any(r != "character_conflicts" for r in reasons)
            rows.append({"book": book, "page": page + 1, "source": source,
                         "source_verified": bool(verified.get(book)), "consensus": consensus,
                         "page_consensus": page_diagnostic,
                         "route": "review_page" if severe else "review_characters" if reasons else "agreement_only",
                         "reasons": reasons,
                         "quality_issues": quality,
                         "reading_order": {name: r["reading_order"] for name, r in records.items() if r is not None},
                         "counts": {name: {"han": r["han"], "visible": r["visible"]} if r is not None else None
                                    for name, r in records.items()},
                         "hit_max": {name: r["hit_max"] if r is not None else None for name, r in records.items()},
                         "max_pairwise_distance": max_distance})
        configs[pair["book"]] = {name: run["manifest"] for name, run in runs.items()}
    return {"models": models, "rows": rows, "warnings": sorted(set(warnings)),
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus_manifest": str(Path(manifest_path).resolve()), "manifests": configs,
            "settings": {"mode": "visible", "locator": "ppocr", "review_threshold": review_threshold,
                         "max_alignment_cells": max_alignment_cells, "reading_order": reading_order,
                         "page_consensus": bool(page_consensus), "page_consensus_options": page_consensus_options or {},
                         "quality_policy": "flag_and_abstain", "vote_rule": "strict_majority_all_models"}}


def multi_tables(result):
    import pandas as pd
    pages, conflicts = [], []
    for row in result["rows"]:
        c = row["consensus"]
        pages.append({"Sách": row["book"], "Trang": row["page"], "Xử lý": row["route"],
                      **{name: (row["counts"][name]["han"] if row["counts"][name] else None) for name in result["models"]},
                      "Bất đồng": len(c["conflicts"]),
                      "Đã định vị": sum(x["localization"] is not None for x in c["conflicts"]),
                      "Entropy (bit)": c["mean_entropy_bits"], "Lý do": ", ".join(row["reasons"])})
        diagnostic = row.get("page_consensus") or {}
        pages[-1]["Cấu trúc trang"] = diagnostic.get("status", "not_assessed")
        pages[-1]["Dấu hiệu cấp trang"] = page_consensus_summary(diagnostic)
        pages[-1]["Không bỏ phiếu"] = "; ".join(f"{name}: {', '.join(issues)}" for name, issues in c.get("excluded_models", {}).items())
        if row.get('spatial_coverage'):
            spatial = row['spatial_coverage']
            pages[-1]['Coverage'] = spatial['status']
            pages[-1]['Ứng viên nghi chưa phủ'] = spatial.get('summary', {}).get('potential_uncovered')
        if row.get('variant_stability'):
            pages[-1]['Cặp biến thể đổi chữ'] = row['variant_stability']['summary']['changed_pairs']
        for conflict in c["conflicts"]:
            conflicts.append({"Sách": row["book"], "Trang": row["page"], "Cột": conflict["index"],
                              **{name: "không bỏ phiếu (output cần kiểm tra)" if name in c.get("excluded_models", {}) else "chưa có output" if value is None else value or "∅"
                                 for name, value in conflict["predictions"].items()},
                              "Đa số": (conflict["majority"] or "∅") if conflict["majority"] is not None else "hòa/chưa đủ phiếu",
                              "Phiếu": f"{conflict['support']}/{conflict['configured_models']}",
                              "Định vị": conflict["localization_status"]})
    return pd.DataFrame(pages), pd.DataFrame(conflicts)


def show_character_conflicts(result, book, page, *, max_characters=20, dpi=250, renderer=None, view="regions"):
    """Show disputed regions or characters, rendering the selected page only once."""
    from IPython.display import display
    from .character_regions import character_crops
    from .ocr_worker import render
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if view not in {"regions", "characters"}:
        raise ValueError("view must be regions or characters")
    row = next((r for r in result["rows"] if r["book"] == book and r["page"] == page), None)
    if row is None:
        raise ValueError("Book/page not in selected campaign")
    diagnostic = row.get("page_consensus")
    if diagnostic:
        print("Đồng thuận cấp trang:", diagnostic['status'])
        for finding in diagnostic['findings']:
            print(f"{finding['model']}: {finding['label']}; vùng {finding['region_ids']}; "
                  f"đối chứng {finding['supporters']}; stream={finding['stream']}.")
        if diagnostic['findings']:
            print("Đây là dấu hiệu từ các mốc văn bản; vùng ảnh chỉ được xác nhận khi có tọa độ model.")
    _, table = multi_tables({**result, "rows": [row]})
    display(table.head(max_characters))
    if not row["consensus"]["conflicts"]:
        print("Chưa ghi nhận cột bất đồng; kiểm tra trạng thái model thiếu, không bỏ phiếu hoặc chưa căn chỉnh được.")
        return []
    entries = (row["consensus"].get("conflict_regions", row["consensus"]["conflicts"])
               if view == "regions" else row["consensus"]["conflicts"])
    image = None
    shown = []
    for conflict in entries[:max_characters]:
        excluded = row["consensus"].get("excluded_models", {})
        label = " | ".join(f"{name}: {'không bỏ phiếu (output cần kiểm tra)' if name in excluded else 'chưa có output' if value is None else value or '∅'}"
                           for name, value in conflict["predictions"].items())
        title = (f"Vùng {conflict['number']} (cột {conflict['start'] + 1}–{conflict['stop']})"
                 if "number" in conflict else f"Cột {conflict['index'] + 1}")
        print(f"{title} — {label}")
        location = conflict["localization"]
        if location is None:
            print("Chưa xác định được vùng ảnh từ tọa độ và căn chỉnh hiện có.")
            continue
        if not row["source_verified"] or not row["source"] or not Path(row["source"]).is_file():
            print("Chưa xác minh/truy cập được PDF gốc; không hiển thị ảnh có thể nhầm nguồn.")
            continue
        if image is None:
            image = (renderer or render)(row["source"], page - 1, dpi, 100_000)
        glyph, context, rectangle = character_crops(image, location)
        if location.get("granularity") == "line_context":
            print("Ảnh ngữ cảnh dòng theo khung PP-OCR; chưa xác định khung chữ bất đồng.")
        else:
            print("Ảnh cắt ký tự/cụm ký tự theo khung PP-OCR ước lượng.")
        print(f"Pixel trang gốc: {rectangle}. Dự đoán model khác được ghép bằng căn chỉnh chuỗi.")
        display(glyph)
        print("Ngữ cảnh xung quanh; khung đỏ đánh dấu vùng vừa cắt:")
        display(context)
        shown.append({"title": title, "rectangle": rectangle, "granularity": location.get("granularity", "character")})
    print(f"Đã hiển thị {len(shown)}/{min(len(entries), max_characters)} vùng ảnh; "
          f"giới hạn hiển thị {max_characters}, tổng {len(entries)} mục bất đồng.")
    return shown
