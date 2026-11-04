"""VTM corpus: OCR sách Hán scan và dịch sang tiếng Việt."""

from .config import LAYOUT_PROMPT, OCRConfig, ReportConfig, TranslateConfig
from .pipeline import (
    RunResult,
    export_ocr,
    export_translate,
    find_run,
    list_pdfs,
    list_texts,
    run_ocr,
    run_translate,
    select_entries,
)
from .report import (
    ocr_summary,
    show_compare,
    show_layouts,
    show_ocr_report,
    show_pages,
    show_translation_pairs,
    show_translate_report,
    translate_summary,
)

__all__ = [
    "OCRConfig", "TranslateConfig", "ReportConfig", "LAYOUT_PROMPT",
    "RunResult", "run_ocr", "run_translate", "export_ocr", "export_translate",
    "find_run", "list_pdfs", "list_texts", "select_entries",
    "ocr_summary", "translate_summary", "show_ocr_report", "show_translate_report",
    "show_layouts", "show_pages", "show_translation_pairs", "show_compare",
]
