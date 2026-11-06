"""Tables, charts and a portable HTML report for page-wise count comparisons."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path


STATUS_LABELS = {
    "within_threshold": "Chênh lệch trong ngưỡng",
    "large_gap": "Chênh lệch vượt ngưỡng",
    "one_zero": "Một bên có 0 chữ",
    "both_zero": "Cả hai có 0 chữ (không tính độ khớp)",
    "missing_hunyuan": "Chưa có output Hunyuan",
    "missing_paddle": "Chưa có output Paddle",
    "missing_both": "Chưa có output cả hai",
    "not_selected_hunyuan": "Hunyuan không chọn trang này",
    "not_selected_paddle": "Paddle không chọn trang này",
}


def make_tables(result: dict):
    import pandas as pd

    if not result["rows"]:
        raise ValueError("Không có trang trong hai run. Kiểm tra tasks.jsonl và đường dẫn run.")
    pages = pd.DataFrame(result["rows"])
    numeric = ["page", "han_hunyuan", "han_paddle", "visible_hunyuan", "visible_paddle",
               "count_hunyuan", "count_paddle", "delta_h_minus_p", "abs_gap",
               "agreement_pct", "gap_pct", "sec_hunyuan", "sec_paddle"]
    pages[numeric] = pages[numeric].apply(pd.to_numeric)
    pages["status_label"] = pages["status"].map(STATUS_LABELS)
    paired = pages[pages.has_hunyuan & pages.has_paddle]
    books = []
    for book, group in pages.groupby("book", sort=True):
        both = group[group.has_hunyuan & group.has_paddle]
        valid = int(both.agreement_pct.notna().sum())
        high = both[["count_hunyuan", "count_paddle"]].max(axis=1).sum()
        low = both[["count_hunyuan", "count_paddle"]].min(axis=1).sum()
        books.append({
            "book": book, "selected_union": len(group), "paired": len(both),
            "unpaired": len(group) - len(both),
            "scored_pages": valid,
            "hunyuan_status": result.get("book_states", {}).get(book, {}).get("hunyuan", "unknown"),
            "paddle_status": result.get("book_states", {}).get(book, {}).get("paddle", "unknown"),
            "hunyuan_done": int(group.has_hunyuan.sum()),
            "paddle_done": int(group.has_paddle.sum()),
            "hunyuan_chars_available": int(group.count_hunyuan.sum()),
            "paddle_chars_available": int(group.count_paddle.sum()),
            "both_zero": int((both.status == "both_zero").sum()),
            "flagged_gap": int(both.status.isin(["large_gap", "one_zero"]).sum()) if valid else float("nan"),
            "hit_max_hunyuan": int(group.hit_max_hunyuan.eq(True).sum()) if group.hit_max_hunyuan.notna().any() else float("nan"),
            "hit_max_paddle": int(group.hit_max_paddle.eq(True).sum()) if group.hit_max_paddle.notna().any() else float("nan"),
            "hunyuan_chars_paired": int(both.count_hunyuan.sum()),
            "paddle_chars_paired": int(both.count_paddle.sum()),
            "weighted_count_match_pct": 100 * low / high if high else float("nan"),
            "mean_page_match_pct": both.agreement_pct.mean(),
        })
    return pages, pd.DataFrame(books), paired


def minimal_summary(pages, books):
    """Only coverage and comparable character counts; no charts or files."""
    import math
    import pandas as pd

    def row(book, expected, h_done, p_done, paired_count, h_chars, p_chars, score):
        return {
            "Sách": book,
            "Trang Hunyuan": f"{h_done}/{expected}",
            "Trang Paddle": f"{p_done}/{expected}",
            "Trang so sánh": paired_count,
            "Chữ Hunyuan": h_chars if paired_count else "N/A",
            "Chữ Paddle": p_chars if paired_count else "N/A",
            "Khớp (%)": round(score, 2) if math.isfinite(score) else "N/A",
        }

    rows = [row(r.book, r.selected_union, r.hunyuan_done, r.paddle_done, r.paired,
                r.hunyuan_chars_paired, r.paddle_chars_paired, r.weighted_count_match_pct)
            for r in books.itertuples()]
    if len(books) > 1:
        paired = pages[pages.has_hunyuan & pages.has_paddle]
        denominator = paired[["count_hunyuan", "count_paddle"]].max(axis=1).sum()
        numerator = paired[["count_hunyuan", "count_paddle"]].min(axis=1).sum()
        score = 100 * numerator / denominator if denominator else float("nan")
        rows.append(row("Tổng", len(pages), int(pages.has_hunyuan.sum()), int(pages.has_paddle.sum()),
                        len(paired), int(paired.count_hunyuan.sum()), int(paired.count_paddle.sum()), score))
    return pd.DataFrame(rows)


def make_figures(result: dict, pages, books):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    paired = pages[pages.has_hunyuan & pages.has_paddle]
    h_color, p_color = "#2563eb", "#e87919"
    unit = "chữ Hán" if result["mode"] == "han" else "ký tự không trắng"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.suptitle(f"Hunyuan ↔ Paddle · Tổng quan số lượng {unit}", fontsize=16)
    ax = axes[0, 0]
    positions = np.arange(len(books))
    bars_h = ax.bar(positions - .19, books.hunyuan_chars_available, .38, label="Hunyuan", color=h_color)
    bars_p = ax.bar(positions + .19, books.paddle_chars_available, .38, label="Paddle", color=p_color)
    ax.bar_label(bars_h, labels=[str(v) if n else "Chưa có" for v, n in zip(books.hunyuan_chars_available, books.hunyuan_done)], padding=3, fontsize=8, rotation=90)
    ax.bar_label(bars_p, labels=[str(v) if n else "Chưa có" for v, n in zip(books.paddle_chars_available, books.paddle_done)], padding=3, fontsize=8, rotation=90)
    ax.set(xticks=positions, xticklabels=books.book, title="Chữ đã OCR (tập trang hai model có thể khác nhau)", ylabel=unit)
    if not (books.hunyuan_done.sum() + books.paddle_done.sum()):
        ax.set_ylim(0, 1)
        ax.text(.5, .5, "Chưa có output OCR", ha="center", transform=ax.transAxes)
    else:
        ax.set_ylim(bottom=0)
        ax.margins(y=.20)
    ax.legend()
    ax = axes[0, 1]
    scores = books.weighted_count_match_pct
    bars = ax.bar(positions, scores.fillna(0), color="#0891b2")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                "N/A" if np.isnan(score) else f"{score:.1f}%", ha="center", fontsize=9)
    ax.set(xticks=positions, xticklabels=books.book, ylim=(0, 110),
           title="Độ khớp số lượng theo sách (chỉ phần đã ghép cặp)", ylabel="Khớp có trọng số (%)")
    ax = axes[1, 0]
    values = paired.gap_pct.dropna()
    if len(values):
        ax.hist(values, bins=[0, 5, 10, 20, 40, 60, 80, 100], color="#7c3aed", edgecolor="white")
    else:
        ax.text(.5, .5, "Không có điểm hợp lệ", ha="center", transform=ax.transAxes)
    ax.axvline(result["gap_threshold"] * 100, color="#dc2626", linestyle="--", label="Ngưỡng cảnh báo")
    ax.set(title="Phân bố chênh lệch (không gồm trang cả hai 0 chữ)", xlabel="Chênh lệch = |H−P| / max(H,P) (%)", ylabel="Số trang", xlim=(0, 101))
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax = axes[1, 1]
    ax.bar(positions - .22, books.selected_union, width=.44, color="#e2e8f0")
    ax.bar(positions + .22, books.selected_union, width=.44, color="#e2e8f0")
    ax.bar(positions - .22, books.hunyuan_done, width=.44, label="Hunyuan đã có", color=h_color)
    ax.bar(positions + .22, books.paddle_done, width=.44, label="Paddle đã có", color=p_color)
    ax.set(xticks=positions, xticklabels=books.book,
           title="Tiến độ OCR theo sách (nền xám: toàn bộ trang)", ylabel="Số trang")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)

    return {"dashboard": fig}


def cards_html(pages, books) -> str:
    paired = pages[pages.has_hunyuan & pages.has_paddle]
    denominator = paired[["count_hunyuan", "count_paddle"]].max(axis=1).sum()
    numerator = paired[["count_hunyuan", "count_paddle"]].min(axis=1).sum()
    match = f"{100 * numerator / denominator:.1f}%" if denominator else "N/A"
    scored = int(paired.agreement_pct.notna().sum())
    flagged = str(int(paired.status.isin(["large_gap", "one_zero"]).sum())) if scored else "N/A"
    items = [
        ("Sách / tổng trang", f"{len(books)} / {len(pages):,}"),
        ("Hunyuan đã có output", f"{int(pages.has_hunyuan.sum()):,} / {len(pages):,}"),
        ("Paddle đã có output", f"{int(pages.has_paddle.sum()):,} / {len(pages):,}"),
        ("Trang có output cả hai", f"{len(paired)} / {len(pages)}"),
        ("Khớp số lượng có trọng số", match),
        (f"Vượt ngưỡng / {scored} trang có điểm", flagged),
    ]
    return '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0">' + "".join(
        f'<div style="flex:1;min-width:180px;padding:18px;border:1px solid #cbd5e1;border-radius:12px;background:#f8fafc;color:#0f172a">'
        f'<div style="font-size:13px">{label}</div><div style="font-size:30px;font-weight:700">{value}</div></div>'
        for label, value in items
    ) + "</div>"


def export_report(result: dict, pages, books, figures: dict, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    books.to_csv(directory / "per_book.csv", index=False, encoding="utf-8-sig")
    metadata = {k: v for k, v in result.items() if k != "rows"}
    (directory / "comparison_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plots = []
    for name, figure in figures.items():
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=140, facecolor="white")
        content = buffer.getvalue()
        (directory / f"{name}.png").write_bytes(content)
        plots.append(f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode(content).decode()}" style="width:100%">')
    warning_list = "".join(f"<li>{html.escape(w)}</li>" for w in result["warnings"])
    header = (
        '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>So sánh số lượng chữ OCR</title><style>body{font-family:Arial,sans-serif;max-width:1250px;margin:24px auto;padding:16px;color:#172033}'
        'table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd;text-align:right}th{background:#edf2f7}'
        '.scroll{overflow:auto;max-height:650px}code{overflow-wrap:anywhere}</style><body><h1>Hunyuan ↔ Paddle: số lượng chữ</h1>'
        '<p>Đây là độ khớp về số lượng, không phải độ chính xác OCR. Cùng số chữ vẫn có thể khác nội dung.</p>'
        f'<p>Đếm: <b>{html.escape(result["mode"])}</b>; ngưỡng chênh lệch: {result["gap_threshold"]:.0%}. '
        'Mỗi trang: khớp = min(H,P)/max(H,P); điểm có trọng số = Σmin(H,P)/Σmax(H,P). '
        'Chỉ tính trang có cả hai output; cả hai bằng 0 không có điểm. Trang thiếu không được tính là 0 chữ.</p>'
        f'<p>Hunyuan: <code>{html.escape(result["hunyuan_run"])}</code><br>Paddle: <code>{html.escape(result["paddle_run"])}</code></p>'
        '<p>Text được so là kết quả sau bộ lọc của pipeline. Khác chính sách bỏ tiêu đề/chú thích có thể làm lệch số lượng.</p>'
        f'<p>Thời điểm đọc dữ liệu (UTC): {html.escape(result.get("captured_at", ""))}. '
        'Biểu đồ số chữ hiển thị mọi output đã có; điểm khớp chỉ tính các trang ghép cặp. '
        'N/A nghĩa là chưa đủ dữ liệu để tính, không phải 0%.</p>'
    )
    report = directory / "report.html"
    report.write_text(header + cards_html(pages, books) + f"<ul>{warning_list}</ul>" + "".join(plots)
                      + '<h2>Theo sách</h2><div class="scroll">' + books.round(2).to_html(index=False)
                      + '</div><h2>Tổng hợp trạng thái</h2>'
                      + pages.groupby("status_label", sort=False).size().rename("Số trang").to_frame().to_html()
                      + "</body></html>", encoding="utf-8")
    return report
