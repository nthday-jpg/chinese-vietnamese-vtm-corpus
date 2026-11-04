"""Dịch các đoạn Hán -> Việt bằng 1 LLM (transformers) hoặc dummy; chạy lại thì bỏ qua đoạn đã dịch."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DUMMY_PREFIX = "[VI] "


def load_done(out: Path) -> set[str]:
    done = set()
    for f in out.glob("dich_shard*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def build_prompt(tok, system: str, user: str) -> str:
    """Prompt theo chat template của model (model không nhận vai system thì gộp vào user)."""
    for msgs in ([{"role": "system", "content": system}, {"role": "user", "content": user}],
                 [{"role": "user", "content": system + "\n\n" + user}]):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)  # Qwen3: tắt chế độ suy nghĩ
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("Model không có chat template -- hãy chọn bản *-Instruct / *-Chat")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="bỏ qua khi --engine dummy")
    ap.add_argument("--engine", default="hf", choices=["hf", "dummy"],
                    help="hf = transformers; dummy = không tải model, chỉ test pipeline")
    ap.add_argument("--chunks", required=True, help='jsonl, mỗi dòng {"id", "zh"}')
    ap.add_argument("--prompt", required=True, help='json {"system", "user"}; user chứa {text}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = luôn chọn từ chắc nhất")
    ap.add_argument("--rep-penalty", type=float, default=1.05)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda (hf)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ dịch N đoạn (chạy thử)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(l) for l in open(a.chunks, encoding="utf-8") if l.strip()]
    done = load_done(out)
    todo = [c for i, c in enumerate(chunks) if i % a.nshard == a.shard and c["id"] not in done][:a.limit]
    print(f"[dich#{a.shard}] còn {len(todo)} đoạn", flush=True)
    if not todo:
        return

    if a.engine == "dummy":
        _run_dummy(todo, out, a)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_cuda = a.device != "cpu" and torch.cuda.is_available()
    kw = {"dtype": torch.float16, "device_map": "auto"} if use_cuda else {"dtype": torch.float32}
    if use_cuda and a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    pr = json.loads(Path(a.prompt).read_text(encoding="utf-8"))

    todo.sort(key=lambda c: len(c["zh"]), reverse=True)  # đoạn dài gần nhau -> ít padding
    t_start, n_done = time.time(), 0
    fout = open(out / f"dich_shard{a.shard}.jsonl", "a", encoding="utf-8")
    for b in range(0, len(todo), a.batch):
        batch = todo[b:b + a.batch]
        prompts = [build_prompt(tok, pr["system"], pr["user"].format(text=c["zh"])) for c in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            sample = {"do_sample": True, "temperature": a.temperature} if a.temperature > 0 else {"do_sample": False}
            gen = model.generate(**enc, max_new_tokens=a.max_new, repetition_penalty=a.rep_penalty,
                                 pad_token_id=tok.pad_token_id, **sample)
        gen = gen[:, enc["input_ids"].shape[1]:]
        for c, g in zip(batch, gen):
            n_tok = int((g != tok.pad_token_id).sum())
            vi = tok.decode(g, skip_special_tokens=True).strip()
            fout.write(json.dumps({"id": c["id"], "vi": vi, "hit_max": n_tok >= a.max_new},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        n_done += len(batch)
        eta = (time.time() - t_start) / n_done * (len(todo) - n_done) / 60
        print(f"[dich#{a.shard}] {n_done}/{len(todo)} đoạn | còn ~{eta:.0f} phút\n"
              f"   ZH: {batch[-1]['zh'][:60]}\n   VI: {vi[:120]}", flush=True)
    fout.close()


def _run_dummy(todo: list[dict], out: Path, a) -> None:
    """Ghi bản dịch giả để test chia đoạn, resume và ghép file mà không cần model."""
    fout = open(out / f"dich_shard{a.shard}.jsonl", "a", encoding="utf-8")
    for n, c in enumerate(todo, 1):
        fout.write(json.dumps({"id": c["id"], "vi": DUMMY_PREFIX + c["id"], "hit_max": False},
                              ensure_ascii=False) + "\n")
        fout.flush()
        if n % max(1, a.batch) == 0 or n == len(todo):
            print(f"[dummy#{a.shard}] {n}/{len(todo)} đoạn", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
