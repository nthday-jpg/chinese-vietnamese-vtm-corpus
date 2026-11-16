import json
from pathlib import Path
import tempfile
import unittest

from vtm_corpus.config import TranslateConfig
from vtm_corpus.dich_worker import build_prompt, main


class TranslateTests(unittest.TestCase):
    def test_erax_defaults(self):
        cfg = TranslateConfig()
        self.assertEqual(cfg.model, "erax-ai/EraX-Translator-V1.0")
        self.assertEqual(cfg.gpu_per_model, 1)
        self.assertEqual(cfg.batch, 2)
        self.assertTrue(cfg.user.format(text="漢文").endswith("Dịch sang tiếng Việt."))

    def test_gemma_template_fallback(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if messages[0]["role"] == "system":
                    raise ValueError("System role not supported")
                return messages[0]["content"]

        self.assertEqual(build_prompt(Tokenizer(), "system", "text"), "system\n\ntext")

    def test_two_shards_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = root / "chunks.jsonl"
            chunks.write_text("".join(json.dumps({"id": str(i), "zh": "漢文"}) + "\n"
                                      for i in range(5)), encoding="utf-8")
            for _ in range(2):
                for shard in range(2):
                    main(["--engine", "dummy", "--chunks", str(chunks),
                          "--prompt", str(root / "prompt.json"), "--out", str(root),
                          "--shard", str(shard), "--nshard", "2"])
            rows = [json.loads(line) for f in root.glob("dich_shard*.jsonl")
                    for line in f.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertEqual({r["id"] for r in rows}, {str(i) for i in range(5)})
