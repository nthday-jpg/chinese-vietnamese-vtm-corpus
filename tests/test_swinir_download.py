import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from vtm_corpus.config import OCRConfig
from vtm_corpus.restoration import _download, resolve_assets, auto_device


class AssetTests(unittest.TestCase):
    def test_automatic_device(self):
        for available, expected in ((True, "cuda"), (False, "cpu")):
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
            with patch.dict("sys.modules", {"torch": torch}):
                self.assertEqual(auto_device(), expected)

    def test_flag_needs_no_paths(self):
        self.assertTrue(OCRConfig(swinir=True).swinir)

    def test_download_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"XDG_CACHE_HOME": directory}):
                with patch("vtm_corpus.restoration.urlopen", side_effect=lambda *a, **k: io.BytesIO(b"asset")) as request:
                    source, weights = resolve_assets()
                    self.assertEqual(source.read_bytes(), b"asset")
                    self.assertEqual(weights.read_bytes(), b"asset")
                    resolve_assets()
                    self.assertEqual(request.call_count, 2)

    def test_local_overrides_do_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "models").mkdir()
            (repo / "models/network_swinir.py").write_text("# local")
            weights = repo / "weights.pth"
            weights.write_bytes(b"weights")
            with patch("vtm_corpus.restoration.urlopen") as request:
                resolve_assets(str(repo), str(weights))
                request.assert_not_called()

    def test_failed_download_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights.pth"
            with patch("vtm_corpus.restoration.urlopen", return_value=io.BytesIO(b"")):
                with self.assertRaises(RuntimeError):
                    _download("https://example.org/weights", target)
            self.assertFalse(target.exists())
            self.assertEqual([p.name for p in target.parent.iterdir()], ["weights.pth.lock"])
