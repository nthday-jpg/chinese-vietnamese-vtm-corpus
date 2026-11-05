import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vtm_corpus.preprocess import cached_page


class PageCacheTests(unittest.TestCase):
    def test_resume_invalidation_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf, source, weights = [root / name for name in ("book.pdf", "model.py", "weights.pth")]
            for path in (pdf, source, weights):
                path.write_bytes(b"test")
            args = (str(pdf), 0, 200, 1600, str(root / "cache"))
            with patch("vtm_corpus.restoration.resolve_assets", return_value=(source, weights)), \
                 patch("vtm_corpus.ocr_worker.render", return_value=Image.new("RGB", (10, 20), "gray")) as render, \
                 patch("vtm_corpus.restoration.restore_image", side_effect=lambda image, *a: image) as restore:
                first = cached_page(*args)
                self.assertEqual(first.size, (10, 20))
                cached_page(*args)
                self.assertEqual(restore.call_count, 1)
                self.assertEqual(render.call_count, 1)
                png = next((root / "cache").rglob("*.png"))
                png.write_bytes(b"broken")
                cached_page(*args)
                self.assertEqual(restore.call_count, 2)
                weights.write_bytes(b"changed weights")
                cached_page(*args)
                self.assertEqual(restore.call_count, 3)
                self.assertEqual(len(list((root / "cache").rglob("manifest.json"))), 2)

    def test_worker_receives_cache_directory(self):
        from vtm_corpus.config import OCRConfig
        from vtm_corpus.pipeline import _ocr_cmd
        cfg = OCRConfig(swinir=True)
        cmd, _ = _ocr_cmd(cfg, Path("raw"), Path("tasks"), 0, 1, "test", [], 0)
        self.assertEqual(cmd[cmd.index("--swinir-cache") + 1], cfg.swinir_cache)
