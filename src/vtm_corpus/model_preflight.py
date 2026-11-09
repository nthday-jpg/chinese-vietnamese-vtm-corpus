"""Import worker APIs and check native rendering/weight paths without loading OCR models."""
from __future__ import annotations

import json
import os
import subprocess
import sys


PROBE = """
import importlib.util, json, pathlib, sys
engine, repo, custom_config = sys.argv[1:4]
packages = {'hunyuan': ['torch', 'transformers'], 'paddle': ['paddle', 'paddleocr'],
            'ppocr': ['paddle', 'paddleocr'],
            'tonggu': ['torch', 'transformers', 'mmcv', 'mmdet', 'mmrotate', 'yaml', 'timm', 'cv2', 'Polygon']}[engine]
missing = [p for p in [*packages, 'pypdfium2', 'PIL', 'tqdm'] if importlib.util.find_spec(p) is None]
checks = []
if engine == 'tonggu':
    root = pathlib.Path(repo)
    if not (root / 'tongguocr/pipeline.py').is_file(): missing.append('tonggu_repo/tongguocr/pipeline.py')
    if custom_config:
        if not pathlib.Path(custom_config).is_file(): missing.append('tonggu_config')
    else:
        for p in ['weights/layout/finetune_infer.pth', 'weights/ocr/config.json']:
            if not (root / p).is_file(): missing.append(p)
if not missing:
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument.new()
    page = document.new_page(8,8)
    bitmap = page.render(scale=1)
    image = bitmap.to_pil()
    assert image.size == (8,8)
    image.close(); bitmap.close(); page.close(); document.close()
    checks.append('pdfium_render')
    if engine == 'hunyuan':
        from transformers import AutoProcessor
        try:
            from transformers import HunYuanVLForConditionalGeneration as Model
        except ImportError:
            from transformers import AutoModelForImageTextToText as Model
        assert callable(AutoProcessor) and callable(Model)
        checks.append('hunyuan_worker_api')
    elif engine in {'paddle','ppocr'}:
        from paddleocr import PaddleOCR, PaddleOCRVL
        assert callable(PaddleOCR) and callable(PaddleOCRVL)
        checks.append('paddle_worker_api')
    else:
        import cv2
        assert callable(getattr(cv2,'resize',None)) and callable(getattr(cv2,'cvtColor',None)), 'Incomplete cv2 API'
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import mmcv.ops, mmdet, mmrotate, Polygon, timm
        sys.path.insert(0,str(root.resolve()))
        from tongguocr.config import load_config
        from tongguocr.pipeline import run_pipeline_on_image
        assert callable(run_pipeline_on_image) and callable(AutoProcessor) and callable(AutoModelForImageTextToText)
        config = load_config(custom_config or None)
        for label, value, directory in (
            ('tonggu_layout_weights',config['layout_inference']['detector_checkpoint_path'],False),
            ('tonggu_layout_config',config['layout_inference']['detector_config_path'],False),
            ('tonggu_ocr_weights',config['ocr']['transformers']['model_path'],True),
        ):
            path = pathlib.Path(value)
            if not path.is_absolute(): path = root / path
            if directory:
                if not (path/'config.json').is_file(): missing.append(label+': '+str(path))
            elif not path.is_file(): missing.append(label+': '+str(path))
        checks.append('tonggu_worker_api')
print('VTM_PREFLIGHT_RESULT=' + json.dumps({'missing': missing, 'python': sys.executable, 'checks': checks}))
"""


def preflight_models(models):
    rows = []
    for name, cfg in models.items():
        if cfg.worker_wsl_distro:
            if os.name != "nt" or not cfg.worker_python.startswith("/"):
                rows.append({"model": name, "ready": False, "detail": "WSL cần Windows và worker_python là đường dẫn Linux tuyệt đối"})
                continue
            command = ["wsl.exe", "--distribution", cfg.worker_wsl_distro, "--exec", cfg.worker_python]
        else:
            command = [cfg.worker_python or sys.executable]
        command += ["-c", PROBE, name, cfg.tonggu_repo, cfg.tonggu_config]
        try:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join([*cfg.worker_pythonpath, environment.get("PYTHONPATH", "")])
            if name in {'paddle', 'ppocr', 'tonggu'}:
                environment['MPLBACKEND'] = 'Agg'
            environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                               PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True', USE_TF='0', USE_FLAX='0')
            result = subprocess.run(command, capture_output=True, timeout=120, env=environment)
            if result.returncode:
                output = result.stderr or result.stdout
                detail = output.decode("utf-16-le" if b"\x00" in output else "utf-8", errors="replace").strip()
                rows.append({"model": name, "ready": False, "detail": detail[-4000:] or f"exit={result.returncode}"})
                continue
            lines = result.stdout.decode('utf-8').strip().splitlines()
            marker = 'VTM_PREFLIGHT_RESULT='
            payload = next((line.split(marker, 1)[1] for line in reversed(lines) if marker in line), lines[-1])
            probe = json.loads(payload)
            rows.append({"model": name, "ready": not probe["missing"],
                         "detail": ", ".join(probe["missing"]) or probe["python"]})
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            rows.append({"model": name, "ready": False, "detail": str(exc)})
    return rows
