"""Optional SwinIR color denoising at native resolution (no upscaling)."""
from functools import lru_cache
from pathlib import Path
import importlib.util
import os
import shutil
import tempfile
from urllib.request import urlopen

SWINIR_REVISION = "6545850fbf8df298df73d81f3e8cba638787c8bd"
SWINIR_WEIGHTS = "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"


def _download(url: str, destination: Path):
    """Serialize worker downloads and publish only complete files (Linux)."""
    import fcntl

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(str(destination) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file() and destination.stat().st_size:
            return
        temporary = None
        try:
            print(f"[swinir] downloading {url}", flush=True)
            with urlopen(url, timeout=120) as response:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    temporary = Path(output.name)
                    shutil.copyfileobj(response, output)
            if not temporary.stat().st_size:
                raise OSError("empty download")
            temporary.replace(destination)
        except Exception as exc:
            raise RuntimeError(f"SwinIR download failed: {url}. Check internet access or provide "
                               "populate the SwinIR asset cache for offline use.") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def resolve_assets(repo: str = "", checkpoint: str = "") -> tuple[Path, Path]:
    """Resolve explicit local assets or download official defaults to XDG cache.

    The downloaded architecture is Python code from a pinned upstream commit.
    Weights are from the official SwinIR v0.0 release; no full git clone needed.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "vtm_corpus" / "swinir"
    source = (Path(repo).expanduser() / "models" / "network_swinir.py" if repo else
              cache / SWINIR_REVISION / "network_swinir.py")
    weights = Path(checkpoint).expanduser() if checkpoint else cache / SWINIR_WEIGHTS
    # Validate overrides before attempting any network access.
    for provided, path in ((repo, source), (checkpoint, weights)):
        if provided and not path.is_file():
            raise FileNotFoundError(f"SwinIR asset not found: {path}")
    if not repo:
        _download(f"https://raw.githubusercontent.com/JingyunLiang/SwinIR/{SWINIR_REVISION}/models/network_swinir.py", source)
    if not checkpoint:
        _download(f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{SWINIR_WEIGHTS}", weights)
    return source, weights


@lru_cache(maxsize=2)
def _load(repo: str, checkpoint: str, device: str):
    import torch
    source, checkpoint = resolve_assets(repo, checkpoint)
    spec = importlib.util.spec_from_file_location("vtm_swinir_network", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                         img_range=1., depths=[6] * 6, embed_dim=180,
                         num_heads=[6] * 6, mlp_ratio=2, upsampler='', resi_connection='1conv')
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights.get("params", weights), strict=True)
    return model.to(device).eval()


def auto_device():
    """Select CUDA visible to this worker, otherwise CPU."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def restore_image(image):
    """Denoise RGB using overlapping 128px tiles; retain exact dimensions."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    device = auto_device()
    model = _load("", "", device)
    array = np.array(image.convert("RGB"), copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255
    h, w = array.shape[:2]
    tensor = F.pad(tensor, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
    _, _, ph, pw = tensor.shape
    result, counts = torch.zeros_like(tensor), torch.zeros_like(tensor)
    tile = 128
    ys = list(range(0, max(ph - tile, 0), 96)) + [max(ph - tile, 0)]
    xs = list(range(0, max(pw - tile, 0), 96)) + [max(pw - tile, 0)]
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                patch = tensor[..., y:y + tile, x:x + tile].to(device)
                prediction = model(patch).cpu()
                result[..., y:y + tile, x:x + tile] += prediction
                counts[..., y:y + tile, x:x + tile] += 1
    array = (result / counts)[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype("uint8"))
