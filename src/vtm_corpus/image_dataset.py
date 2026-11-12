"""Offline, content-addressed image preparation and portable dataset snapshots.

Only PDFium and Pillow are required. JSONL avoids a Parquet/Arrow dependency in
the independent OCR runtimes. Images are stored once, irrespective of how many
logical page/group/column views refer to them. No OCR predictions are stored.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time

from .pdf_images import acquire_pdf_image
from .image_preparation import projection_gaps, projection_mask, _partition_box
from .restoration_classical import EXTRA_CLASSICAL, classical_parameters, extra_classical_variant


IMAGE_DATASET_FORMAT = "vtm-image-dataset"
IMAGE_DATASET_VERSION = 1
IMAGE_DATASET_CODE_SIGNATURE = ""  # The notebook builder supplies its source hash.


@dataclass
class ImageDatasetConfig:
    dpi: int = 200
    acquisition: str = "auto"
    max_source_pixels: int = 100_000_000
    analysis_max_side: int = 1200
    reading_direction: str = "vertical_rtl"
    columns_per_group: int = 3
    max_columns: int = 64
    column_cuts: list[float] | None = None
    page_column_cuts: dict = field(default_factory=dict)
    ink_threshold: int = 40
    gap_ink_fraction: float = .006
    min_gap_fraction: float = .006
    variants: list[str] = field(default_factory=lambda: ["original"])
    variant_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.dpi <= 0 or self.max_source_pixels < 1 or not 32 <= self.analysis_max_side <= 1600
                or self.columns_per_group < 1 or self.max_columns < 1):
            raise ValueError("Invalid preparation dimensions/group limits")
        if self.acquisition not in {"auto", "render"}:
            raise ValueError("acquisition must be auto or render")
        if self.reading_direction not in {"vertical_rtl", "horizontal_ltr"}:
            raise ValueError("Invalid reading_direction")
        if not 0 < self.ink_threshold < 255 or not 0 <= self.gap_ink_fraction < 1 or not 0 < self.min_gap_fraction < .5:
            raise ValueError("Invalid projection thresholds")
        if (not isinstance(self.variants, list) or "original" not in self.variants
                or len(set(self.variants)) != len(self.variants)
                or set(self.variants)-{"original", "bicubic_x2", "bicubic_x4", "denoised", "clahe", *EXTRA_CLASSICAL}):
            raise ValueError("variants require original and supported classical names")
        if set(self.variant_options)-{"clahe", "denoised", *EXTRA_CLASSICAL}:
            raise ValueError("Unknown variant_options")
        for name, options in self.variant_options.items():
            if name in EXTRA_CLASSICAL:
                classical_parameters(name, options)
                continue
            if name == 'denoised':
                continue  # Conservative NLM limits are checked at transformation time.
            if set(options)-{"clip_limit", "tile_grid"}:
                raise ValueError(f"Unknown {name} parameters")
            if not 0 < options.get("clip_limit", 1.5) <= 4 or not 2 <= options.get("tile_grid", 8) <= 32:
                raise ValueError("CLAHE clip_limit must be (0,4], tile_grid [2,32]")
        dataset_validate_cuts(self.column_cuts)
        for page, cuts in self.page_column_cuts.items():
            if not str(page).isdigit() or int(page) < 1:
                raise ValueError("page_column_cuts uses 1-based page keys")
            dataset_validate_cuts(cuts)


def dataset_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def dataset_hash(value):
    return hashlib.sha256(dataset_json_bytes(value)).hexdigest()


def dataset_file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_code_signature():
    if IMAGE_DATASET_CODE_SIGNATURE:
        return IMAGE_DATASET_CODE_SIGNATURE
    root = Path(__file__).parent
    return dataset_hash({name: dataset_file_hash(root/name) for name in
                         ("image_dataset.py", "image_preparation.py", "pdf_images.py", "restoration_classical.py")})


def dataset_environment():
    import PIL
    import pypdfium2 as pdfium
    result = {"pillow": PIL.__version__, "pypdfium2": str(pdfium.PYPDFIUM_INFO),
              "pdfium": str(pdfium.PDFIUM_INFO), "python": sys.version.split()[0]}
    return result


def dataset_path(root, relative):
    relative = str(relative)
    if "\\" in relative or ":" in relative:
        raise ValueError(f"Dataset requires portable relative paths: {relative}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError(f"Invalid dataset path: {relative}")
    root = Path(root).resolve()
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


def dataset_write(root, relative, data):
    path = dataset_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only generated cache objects/indexes are replaced; arbitrary files in the
    # dataset root are neither deleted nor included in packaged snapshots.
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return {"path": str(relative), "sha256": hashlib.sha256(data).hexdigest()}


def dataset_read(root, reference):
    data = dataset_path(root, reference["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {reference['path']}")
    return data


def dataset_store_image(root, image):
    image = image.convert("RGB")
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=6)
    data = output.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    relative = f"objects/{sha[:2]}/{sha}.png"
    path = dataset_path(root, relative)
    if not path.exists() or dataset_file_hash(path) != sha:
        dataset_write(root, relative, data)
    return {"path": relative, "sha256": sha, "size": list(image.size),
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}


def dataset_load_image(root, reference):
    from PIL import Image
    with Image.open(io.BytesIO(dataset_read(root, reference))) as loaded:
        image = loaded.convert("RGB")
    if (list(image.size) != reference["size"] or
            hashlib.sha256(image.tobytes()).hexdigest() != reference["pixel_sha256"]):
        raise ValueError(f"Dataset image dimensions/pixels changed: {reference['path']}")
    return image


@contextmanager
def dataset_lock(root):
    """OS lock releases after interruption; no stale PID/lockfile cleanup needed."""
    import os
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root)/"prepare.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def dataset_validate_cuts(cuts):
    if cuts is not None and (not isinstance(cuts, list) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 1
        for v in cuts) or cuts != sorted(set(cuts))):
        raise ValueError("column_cuts must be unique increasing fractions in (0,1)")


def dataset_partition_qc(regions, size, direction):
    """Exact geometric partition check, independent of OCR/text completeness."""
    w, h = size
    expected = 0 if direction == "horizontal_ltr" else w
    for index, region in enumerate(regions):
        x1, y1, x2, y2 = region["source_box"]
        if (not all(isinstance(v, int) for v in (x1,y1,x2,y2)) or
                not 0 <= x1 < x2 <= w or (y1,y2) != (0,h) or region["reading_order"] != index):
            raise ValueError("Invalid canonical crop bounds/order")
        start, end = (x1, x2) if direction == "horizontal_ltr" else (x2, x1)
        if start != expected:
            raise ValueError("Canonical partition has a gap, overlap or inconsistent reading order")
        expected = end
    if not regions or expected != (w if direction == "horizontal_ltr" else 0):
        raise ValueError("Canonical partition omits page content")


def dataset_plan_layout(image, config, *, page=0):
    """Projection baseline, no model/pixel budget; all coordinates in source pixels."""
    from PIL import Image, ImageDraw
    analysis = image.copy()
    analysis.thumbnail((config.analysis_max_side, config.analysis_max_side), Image.Resampling.LANCZOS)
    options = {key: getattr(config, key) for key in ("ink_threshold", "gap_ink_fraction", "min_gap_fraction")}
    mask = projection_mask(analysis, options)
    w, h = image.size
    aw, ah = analysis.size
    manual = config.page_column_cuts.get(str(page+1), config.column_cuts)
    if manual is not None:
        cuts = [round(v*w) for v in manual]
        method = "manual_columns"
    elif config.reading_direction == "vertical_rtl":
        cuts = sorted(set(round(v*w/aw) for v in projection_gaps(analysis, "x", options)))
        method = "whitespace_projection"
    else:
        cuts, method = [], "whole_page_horizontal"
    if len(set([0,*cuts,w])) != len(cuts)+2:
        raise ValueError("Column cuts collapse in source space")
    warnings = ["heuristic_layout; geometric_coverage_is_not_text_coverage"]
    if len(cuts)+1 > config.max_columns:
        if manual is not None:
            raise ValueError("Manual layout exceeds max_columns")
        warnings.append("too_many_gutters; retained_whole_page")
        cuts = []
    if not cuts:
        warnings.append("no_partition; whole_page_retained")
    thresholds = sorted(set([max(1,config.ink_threshold-8), config.ink_threshold, min(254,config.ink_threshold+8)]))
    found = [projection_gaps(analysis, "x", {**options, "ink_threshold": t}) for t in thresholds]
    ink = mask.tobytes()
    top, bottom = int(.03*ah), max(1,int(.97*ah))
    projection = [sum(ink[y*aw+x] > 0 for y in range(top,bottom))/(bottom-top) for x in range(aw)]
    evidence = []
    for cut in cuts:
        x = min(aw-1,max(0,round(cut*aw/w)))
        lo, hi = x, x
        while lo > 0 and projection[lo-1] <= config.gap_ink_fraction:
            lo -= 1
        while hi < aw-1 and projection[hi+1] <= config.gap_ink_fraction:
            hi += 1
        tolerance = max(2,round(aw*.006))
        stability = sum(any(abs(c-x) <= tolerance for c in cs) for cs in found)/len(found)
        radius = max(2,round(aw*.035))
        left, right = projection[max(0,x-radius):x], projection[x+1:min(aw,x+radius+1)]
        risks = []
        if projection[x] > .02:
            risks.append("cut_crosses_analysis_ink; rules_or_strokes_need_review")
        if stability < 2/3:
            risks.append("unstable_across_ink_thresholds")
        if not left or not right or min(max(left),max(right)) <= config.gap_ink_fraction*2:
            risks.append("weak_text_support_on_one_side")
        evidence.append({"source_x":cut,"analysis_x":x,"gutter_width_analysis":hi-lo+1,
                         "ink_fraction":projection[x],"threshold_stability":stability,"warnings":risks})
    full = (0,0,w,h)
    columns = _partition_box(full,cuts,"x",config.reading_direction)
    group = config.columns_per_group
    groups = []
    for start in range(0,len(columns),group):
        chunk = columns[start:start+group]
        groups.append((min(b[0] for b in chunk),0,max(b[2] for b in chunk),h))
    regions = []
    for kind, boxes in (("page",[full]),("group",groups),("column",columns)):
        for index, box in enumerate(boxes):
            regions.append({"id":f"{kind}_{index:03d}","crop_type":kind,"reading_order":index,
                            "source_box":list(box),"overlap":False})
        dataset_partition_qc([r for r in regions if r["crop_type"]==kind], image.size, config.reading_direction)
    preview = analysis.copy()
    draw = ImageDraw.Draw(preview)
    for region in [r for r in regions if r["crop_type"]=="column"]:
        box = region["source_box"]
        x1,x2 = round(box[0]*aw/w),round(box[2]*aw/w)
        draw.rectangle((x1,0,max(x1,x2-1),ah-1),outline="red",width=1)
        draw.text((x1+2,3),str(region["reading_order"]+1),fill="red")
    for item in evidence:
        if item["warnings"]:
            draw.line((item["analysis_x"],0,item["analysis_x"],ah-1),fill="orange",width=2)
    draw.text((3,ah-14), "RTL <-" if config.reading_direction=="vertical_rtl" else "LTR ->",fill="red")
    layout = {"method":method,"source_size":[w,h],"analysis_size":[aw,ah],
              "analysis_to_source":[w/aw,0,0,h/ah,0,0],"reading_direction":config.reading_direction,
              "regions":regions,"cuts":evidence,"excluded_regions":[],
              "uncertain_regions":[r["id"] for r in regions if r["crop_type"]=="column"],
              "geometric_coverage":1.0,"text_coverage":"unverified","warnings":warnings,
              "thresholds_tested":thresholds,"review_status":"needs_visual_review"}
    return layout, mask.convert("RGB"), preview


def dataset_variant(image, name, options):
    from PIL import Image
    if name in EXTRA_CLASSICAL:
        return extra_classical_variant(image, name, options)
    if name == "original":
        return image, {"method":"identity","parameters":{},"warnings":[]}
    if name in {"bicubic_x2", "bicubic_x4"}:
        scale = 2 if name == 'bicubic_x2' else 4
        return image.resize((image.width*scale,image.height*scale),Image.Resampling.BICUBIC), {
            "method":"bicubic","parameters":{"scale":scale},
            "warnings":["experimental_interpolation; no_recovered_source_detail"]}
    if name == "clahe":
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("clahe requires an already available OpenCV/NumPy installation; use original offline") from exc
        params = {"clip_limit":1.5,"tile_grid":8,**options}
        lab = cv2.cvtColor(np.asarray(image.convert("RGB")),cv2.COLOR_RGB2LAB)
        grid = int(params["tile_grid"])
        lab[:,:,0] = cv2.createCLAHE(clipLimit=float(params["clip_limit"]),tileGridSize=(grid,grid)).apply(lab[:,:,0])
        return Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)), {
            "method":"clahe_lab_l","parameters":params,"opencv":cv2.__version__,
            "warnings":["experimental_contrast; inspect_weak_strokes"]}
    if name == 'denoised':
        import cv2
        import numpy as np
        defaults = {'h': 2., 'template_window': 7, 'search_window': 21}
        params = {**defaults, **options}
        if (set(options)-set(defaults) or not isinstance(params['h'], (int,float)) or not math.isfinite(params['h']) or not 0 < params['h'] <= 5 or
                any(not isinstance(params[k], int) or params[k] % 2 != 1 for k in ('template_window', 'search_window')) or
                not 3 <= params['template_window'] <= params['search_window'] <= 63):
            raise ValueError('Conservative NLM requires 0<h<=5 and odd windows 3..63')
        lab = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.fastNlMeansDenoising(lab[:, :, 0], None, float(params['h']), params['template_window'], params['search_window'])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)), dict(method='nlm_lab_l', parameters=params,
            warnings=['may_remove_weak_strokes; opt_in_visual_review_required'])
    raise ValueError(f"Unsupported image variant: {name}")


def dataset_validate_page(root, record, *, check_pixels=True):
    source = dataset_load_image(root,record["source"]) if check_pixels else None
    acquisition = json.loads(dataset_read(root,record["acquisition"]))
    layout = json.loads(dataset_read(root,record["layout"]))
    if record["source"]["size"] != layout["source_size"] or acquisition["size"] != layout["source_size"]:
        raise ValueError("Source, acquisition and layout size mismatch")
    dataset_read(root,layout["analysis_mask"])
    dataset_read(root,layout["preview"])
    for kind in ("page","group","column"):
        regions = [r for r in layout["regions"] if r["crop_type"]==kind]
        dataset_partition_qc(regions,layout["source_size"],layout["reading_direction"])
    if [r["id"] for r in record["regions"]] != [r["id"] for r in layout["regions"]]:
        raise ValueError("Image dataset omits canonical regions")
    for region, geometry in zip(record["regions"],layout["regions"]):
        if any(region[k] != geometry[k] for k in ("source_box","crop_type","reading_order")):
            raise ValueError("Region geometry differs from canonical layout")
        if "original" not in region["images"]:
            raise ValueError("Region original variant is missing")
        x1,y1,x2,y2 = region["source_box"]
        for name, entry in region["images"].items():
            expected = [(x2-x1)*(2 if name=="bicubic_x2" else 1),(y2-y1)*(2 if name=="bicubic_x2" else 1)]
            if entry["size"] != expected or entry["image_to_source"] != [
                    (x2-x1)/expected[0],0,0,(y2-y1)/expected[1],x1,y1]:
                raise ValueError("Variant geometry/transform mismatch")
            if check_pixels:
                image = dataset_load_image(root,entry)
                if name=="original" and image.tobytes() != source.crop(region["source_box"]).tobytes():
                    raise ValueError("Original crop is not pixel-identical to source")
    return layout


def dataset_open(directory):
    from .worker_paths import runtime_path
    path = runtime_path(directory)
    descriptor = path/"dataset.json" if path.is_dir() else path
    root = descriptor.parent.parent if descriptor.parent.name=="snapshots" else descriptor.parent
    data = json.loads(descriptor.read_text(encoding="utf-8"))
    if data.get("format") != IMAGE_DATASET_FORMAT or data.get("version") != IMAGE_DATASET_VERSION:
        raise ValueError("Unsupported image dataset format/version")
    body = {k:v for k,v in data.items() if k!="snapshot_id"}
    if dataset_hash(body) != data["snapshot_id"]:
        raise ValueError("Dataset snapshot identity mismatch")
    return root, data


def dataset_page(root, descriptor, source_sha, page):
    entries = [p for p in descriptor["pages"] if p["source_sha256"]==source_sha and p["page"]==page]
    if not entries:
        return None
    records = [json.loads(dataset_read(root,e["record"])) for e in entries]
    if len({r["input_key"] for r in records}) != 1:
        raise ValueError("Ambiguous prepared layouts for the same source/page")
    record = records[0]
    if record["source_sha256"] != source_sha or record["page"] != page:
        raise ValueError("Dataset page identity mismatch")
    return record


def bind_image_dataset(cfg, source_sha, pages, *, allow_unavailable=False):
    """Bind an immutable snapshot before looking up OCR checkpoints."""
    from .worker_paths import runtime_path
    if not cfg.image_dataset:
        return
    root, descriptor = dataset_open(cfg.image_dataset)
    snapshot_path = root/f"snapshots/{descriptor['snapshot_id']}.json"
    # Freeze the snapshot rather than following a mutable dataset.json later.
    if not snapshot_path.is_file():
        raise ValueError("Dataset immutable snapshot file is missing")
    if dataset_open(snapshot_path)[1] != descriptor:
        raise ValueError("Immutable dataset snapshot differs from descriptor")
    cfg.image_dataset = str(runtime_path(snapshot_path).resolve())
    cfg.image_dataset_key = descriptor["snapshot_id"]
    from .prepared_variants import bind_restoration_variant, restoration_index
    if descriptor.get("restoration"):
        restoration_index(root, descriptor)
    needs_extension = cfg.image_variant not in {"original", "clahe", "bicubic_x2"}
    for page in pages:
        rec = dataset_page(root, descriptor, source_sha, page)
        kind = "page" if cfg.image_crop_type == "native" else cfg.image_crop_type
        if rec and any(cfg.image_variant not in r["images"] for r in rec["regions"] if r["crop_type"] == kind):
            needs_extension = True
    if needs_extension:
        return bind_restoration_variant(cfg, root, descriptor, source_sha, pages,
                                        allow_unavailable=allow_unavailable)
    states = {}
    for page in pages:
        record = dataset_page(root,descriptor,source_sha,page)
        if record is None:
            if allow_unavailable:
                states[page] = {"state": "missing_page"}
                continue
            if cfg.image_dataset_fallback:
                continue
            raise ValueError(f"Prepared dataset missing source {source_sha}, page {page+1}")
        layout = dataset_validate_page(root,record)
        if layout["reading_direction"] != cfg.reading_direction:
            raise ValueError("Dataset and OCR reading directions differ; choose a matching preparation profile")
        kind = "page" if cfg.image_crop_type=="native" else cfg.image_crop_type
        for region in record["regions"]:
            if region["crop_type"]==kind and cfg.image_variant not in region["images"]:
                raise ValueError(f"Missing variant {cfg.image_variant} for {region['id']}")
        states[page] = {"state": "ok"}
    # Content, rather than the dataset's mount point or unrelated variants,
    # determines raw OCR reuse. Extension variants already use this contract.
    from .prepared_variants import bind_restoration_variant
    return bind_restoration_variant(cfg, root, descriptor, source_sha, pages,
                                    allow_unavailable=allow_unavailable or cfg.image_dataset_fallback)


def dataset_ocr_page(cfg, source_sha, page):
    """Load prepared source with inference selection attached; never render a PDF."""
    root, descriptor = dataset_open(cfg.image_dataset)
    if cfg.image_dataset_key and descriptor["snapshot_id"] != cfg.image_dataset_key:
        raise ValueError("Prepared input snapshot changed since OCR plan")
    record = dataset_page(root,descriptor,source_sha,page)
    if record is None:
        if cfg.image_dataset_fallback:
            if cfg.image_variant != 'original':
                raise ValueError('Missing prepared variant cannot fall back to original PDF pixels')
            return None
        raise ValueError(f"Prepared dataset missing page {page+1}")
    layout = dataset_validate_page(root,record,check_pixels=False)
    from .prepared_variants import resolve_variant_page
    record = resolve_variant_page(root, descriptor, record, cfg)
    image = dataset_load_image(root,record["source"])
    kind = "page" if cfg.image_crop_type=="native" else cfg.image_crop_type
    regions = [r for r in record["regions"] if r["crop_type"]==kind]
    if not regions:
        raise ValueError(f"Prepared crop type missing: {kind}")
    for region in regions:
        dataset_load_image(root,region["images"][cfg.image_variant])
    image.info["prepared_dataset"] = {"root":str(root),"snapshot_id":descriptor["snapshot_id"],
        "source_sha256":source_sha,"source":record["source"],"layout":layout,"regions":regions,
        "variant":cfg.image_variant,"crop_type":cfg.image_crop_type,
        "acquisition":json.loads(dataset_read(root,record["acquisition"]))}
    return image, dataset_path(root,record["source"]["path"])


def prepare_image_dataset(input_dir, output_dir, *, config=None, books=None, pages="1-2",
                          limit=2, max_books=1, profile_dir=None, publish=True):
    """Prepare a bounded selection; repeat the same command to resume.

    A completed source/layout/variant is independently reusable. Only after QC
    succeeds is a new dataset.json published. Existing snapshots remain usable.
    """
    from .pipeline import list_pdfs, select_entries, parse_pages
    config = config or ImageDatasetConfig()
    config.__post_init__()
    if limit is not None and limit < 1 or max_books is not None and max_books < 1:
        raise ValueError("limit/max_books must be positive or None")
    entries = select_entries(list_pdfs(input_dir),books or [])
    if max_books is not None:
        entries = entries[:max_books]
    selected, remaining = [], limit
    for entry in entries:
        wanted = parse_pages(pages,entry["n_pages"])
        if remaining is not None:
            wanted = wanted[:remaining]; remaining -= len(wanted)
        if wanted:
            selected.append((entry,wanted))
    if not selected:
        raise ValueError("No pages selected")
    root = Path(output_dir).resolve()
    stats = {"pages":0,"source_reused":0,"layout_reused":0,"variants_reused":0}
    started, environment, code = time.monotonic(), dataset_environment(), dataset_code_signature()
    page_entries, manifest_rows, configurations = [], [], {}
    with dataset_lock(root):
        marker = root/"dataset.json"
        if marker.exists():
            dataset_open(root)  # Never replace an unrelated user file.
        for entry, wanted in selected:
            import pypdfium2 as pdfium
            cfg = ImageDatasetConfig(**asdict(config))
            if profile_dir:
                profile = json.loads((Path(profile_dir)/(entry["stem"]+".json")).read_text(encoding="utf-8-sig"))
                # OCR token/resize settings have no place in a canonical dataset.
                for key in ("dpi","reading_direction"):
                    if key in profile:
                        setattr(cfg,key,profile[key])
            cfg.__post_init__()
            configurations[entry["name"]] = asdict(cfg)
            source_sha = dataset_file_hash(entry["path"])
            doc = pdfium.PdfDocument(entry["path"])
            try:
                doc.init_forms()
                for page in wanted:
                    source_key = dataset_hash({"source_sha256":source_sha,"page":page,
                        "acquisition":cfg.acquisition,"dpi":cfg.dpi,"max_source_pixels":cfg.max_source_pixels,
                        "code":code,"environment":environment})
                    source_receipt = f"sources/{source_key}.json"
                    try:
                        cached = json.loads(dataset_path(root,source_receipt).read_bytes())
                        if cached["key"] != source_key:
                            raise ValueError("Source cache key mismatch")
                        image = dataset_load_image(root,cached["source"])
                        acquisition = json.loads(dataset_read(root,cached["acquisition"]))
                        stats["source_reused"] += 1
                    except (OSError,ValueError,KeyError):
                        pdf_page = doc[page]
                        try:
                            image, acquisition = acquire_pdf_image(pdf_page,dpi=cfg.dpi,
                                max_source_pixels=cfg.max_source_pixels,strategy=cfg.acquisition)
                        finally:
                            pdf_page.close()
                        source_asset = dataset_store_image(root,image)
                        acquisition_ref = dataset_write(root,f"sources/{source_key}.acquisition.json",dataset_json_bytes(acquisition))
                        cached = {"key":source_key,"source":source_asset,"acquisition":acquisition_ref}
                        dataset_write(root,source_receipt,dataset_json_bytes(cached))
                    layout_options = {k:v for k,v in asdict(cfg).items() if k not in
                        {"dpi","acquisition","max_source_pixels","variants","variant_options"}}
                    # Only this page's override influences this page's layout cache.
                    layout_options["page_column_cuts"] = {str(page+1):cfg.page_column_cuts.get(str(page+1),cfg.column_cuts)}
                    layout_key = dataset_hash({"source":cached["source"]["sha256"],"options":layout_options,"code":code})
                    layout_relative = f"layouts/{layout_key}.json"
                    try:
                        layout_data = dataset_path(root,layout_relative).read_bytes()
                        layout = json.loads(layout_data)
                        if layout["key"] != layout_key:
                            raise ValueError("Layout key mismatch")
                        dataset_read(root,layout["analysis_mask"]); dataset_read(root,layout["preview"])
                        stats["layout_reused"] += 1
                    except (OSError,ValueError,KeyError):
                        layout, mask, preview = dataset_plan_layout(image,cfg,page=page)
                        layout.update(key=layout_key,analysis_mask=dataset_store_image(root,mask),preview=dataset_store_image(root,preview))
                        layout_data = dataset_json_bytes(layout)
                        dataset_write(root,layout_relative,layout_data)
                    layout_ref = {"path":layout_relative,"sha256":hashlib.sha256(layout_data).hexdigest()}
                    regions = []
                    for geometry in layout["regions"]:
                        crop = image.crop(geometry["source_box"])
                        variants = {}
                        for name in cfg.variants:
                            options = cfg.variant_options.get(name,{})
                            variant_environment = dict(environment)
                            if name in {"clahe", "denoised", *EXTRA_CLASSICAL}:
                                import cv2
                                variant_environment["opencv"] = cv2.__version__
                            key = dataset_hash({"source":cached["source"]["sha256"],"box":geometry["source_box"],
                                "variant":name,"options":options,"code":code,"environment":variant_environment})
                            receipt = f"variants/{key}.json"
                            try:
                                variant = json.loads(dataset_path(root,receipt).read_bytes())
                                if variant["key"] != key:
                                    raise ValueError("Variant cache key mismatch")
                                dataset_load_image(root,variant)
                                stats["variants_reused"] += 1
                            except (OSError,ValueError,KeyError):
                                prepared, operation = dataset_variant(crop,name,options)
                                x1,y1,x2,y2 = geometry["source_box"]
                                variant = {**dataset_store_image(root,prepared),"key":key,"operation":operation,
                                    "image_to_source":[(x2-x1)/prepared.width,0,0,(y2-y1)/prepared.height,x1,y1]}
                                dataset_write(root,receipt,dataset_json_bytes(variant))
                            variants[name] = variant
                            manifest_rows.append({"book":entry["name"],"source_pdf":Path(entry["path"]).relative_to(Path(input_dir)).as_posix(),
                                "source_sha256":source_sha,"page":page+1,"page_index":page,
                                "acquisition_method":acquisition["method"],"source_size":cached["source"]["size"],
                                "source_image":cached["source"],"analysis_size":layout["analysis_size"],
                                "analysis_to_source":layout["analysis_to_source"],"source_to_pdf":acquisition["source_to_pdf"],
                                "region_id":geometry["id"],"crop_type":geometry["crop_type"],
                                "source_box":geometry["source_box"],"reading_order":geometry["reading_order"],
                                "reading_direction":cfg.reading_direction,"variant":name,**variant,
                                "qc_warnings":acquisition["warnings"]+layout["warnings"]+
                                    [w for c in layout["cuts"] for w in c["warnings"]],
                                "layout":layout_ref,"acquisition":cached["acquisition"],"code_sha256":code,
                                "processing_version":IMAGE_DATASET_VERSION})
                        regions.append({**geometry,"images":variants})
                    record = {"input_key":dataset_hash({"source":source_key,"layout":layout_key,"variants":cfg.variants,
                                  "options":cfg.variant_options}),"source_sha256":source_sha,"page":page,
                              "source":cached["source"],"acquisition":cached["acquisition"],"layout":layout_ref,"regions":regions}
                    dataset_validate_page(root,record)
                    record_bytes = dataset_json_bytes(record)
                    record_key = hashlib.sha256(record_bytes).hexdigest()
                    record_ref = dataset_write(root,f"pages/{record_key}.json",record_bytes)
                    page_entries.append({"book":entry["name"],"source_pdf":Path(entry["path"]).relative_to(Path(input_dir)).as_posix(),
                                         "source_sha256":source_sha,"page":page,"record":record_ref})
                    stats["pages"] += 1
            finally:
                doc.close()
        manifest_data = b"".join(dataset_json_bytes(row) for row in manifest_rows)
        manifest_sha = hashlib.sha256(manifest_data).hexdigest()
        manifest_ref = dataset_write(root,f"manifests/{manifest_sha}.jsonl",manifest_data)
        descriptor = {"format":IMAGE_DATASET_FORMAT,"version":IMAGE_DATASET_VERSION,"pages":page_entries,
            "expected_pages":[[e["book"],e["source_sha256"],e["page"]] for e in page_entries],
            "manifest":manifest_ref,"configurations":configurations,"environment":environment,"code_sha256":code,
            "geometry":"top_left_source_pixels; half_open_boxes", "ocr_text_coverage":"unverified"}
        descriptor["snapshot_id"] = dataset_hash(descriptor)
        data = dataset_json_bytes(descriptor)
        snapshot = f"snapshots/{descriptor['snapshot_id']}.json"
        dataset_write(root,snapshot,data)
        # Publish last: a crash/failure before this point leaves the last usable
        # snapshot untouched and all finished stages available for resume.
        if publish:
            dataset_write(root,"dataset.json",data)
    stats.update(snapshot_id=descriptor["snapshot_id"],dataset=str(root/snapshot),
                 seconds=round(time.monotonic()-started,3),manifest_rows=len(manifest_rows))
    return stats


def validate_image_dataset(directory):
    root, descriptor = dataset_open(directory)
    snapshot = root/f"snapshots/{descriptor['snapshot_id']}.json"
    if not snapshot.is_file() or dataset_open(snapshot)[1] != descriptor:
        raise ValueError('Immutable dataset snapshot missing or mismatched')
    dataset_read(root,descriptor["manifest"])
    actual = [[e["book"],e["source_sha256"],e["page"]] for e in descriptor["pages"]]
    if actual != descriptor["expected_pages"] or len(set(map(tuple,actual))) != len(actual):
        raise ValueError("Dataset expected-page coverage mismatch")
    warnings = []
    for entry in descriptor["pages"]:
        record = dataset_page(root,descriptor,entry["source_sha256"],entry["page"])
        layout = dataset_validate_page(root,record)
        warnings.extend(f"{entry['book']}/p{entry['page']+1}: {w}" for w in layout["warnings"])
        warnings.extend(f"{entry['book']}/p{entry['page']+1}/x{c['source_x']}: {w}"
                        for c in layout["cuts"] for w in c["warnings"])
    if descriptor.get('restoration'):
        validate_restoration_dataset(root, descriptor)
    return {"pages":len(actual),"snapshot_id":descriptor["snapshot_id"],"warnings":warnings,
            "partial": bool(descriptor.get('restoration', {}).get('partial')),
            "geometric_coverage":1.0,"ocr_text_coverage":"unverified"}


def validate_restoration_dataset(root, descriptor):
    """Check every extension asset, including rows not selected for OCR."""
    from .prepared_variants import restoration_index, variant_region_entry
    index, report = restoration_index(root, descriptor)
    for page in descriptor['pages']:
        record = dataset_page(root, descriptor, page['source_sha256'], page['page'])
        for region in record['regions']:
            for variant in report['selected_variants']:
                row = index[(page['source_sha256'], page['page'], region['id'], variant)]
                if row['state'] != 'ok':
                    continue
                if variant == 'tadisr_text_mask':
                    dataset_load_image(root, row['image']).close()
                else:
                    variant_region_entry(root, descriptor, record, region, variant,
                                         index=index, report=report, allow_generated=True)
                for extra in row.get('extras', {}).values():
                    dataset_read(root, extra)
    return index, report


def package_image_dataset(directory, archive):
    """ZIP_STORED PNGs; include only this validated snapshot's referenced files."""
    import zipfile
    root, descriptor = dataset_open(directory)
    validate_image_dataset(directory)
    archive = Path(archive).resolve()
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    references = {descriptor["manifest"]["path"]}
    for entry in descriptor["pages"]:
        references.add(entry["record"]["path"])
        record = json.loads(dataset_read(root,entry["record"]))
        references.update(record[k]["path"] for k in ("source","acquisition","layout"))
        layout = json.loads(dataset_read(root,record["layout"]))
        references.update(layout[k]["path"] for k in ("analysis_mask","preview"))
        references.update(v["path"] for r in record["regions"] for v in r["images"].values())
    if descriptor.get('restoration'):
        index, report = validate_restoration_dataset(root, descriptor)
        references.update((descriptor['restoration']['report']['path'], report['manifest']['path']))
        for row in index.values():
            if row.get('image'):
                references.add(row['image']['path'])
            references.update(ref['path'] for ref in row.get('extras', {}).values())
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_STORED,allowZip64=True) as output:
        # Fixed timestamps/order make packages reproducible across machines.
        for name in ["dataset.json",f"snapshots/{descriptor['snapshot_id']}.json",*sorted(references)]:
            info = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.external_attr = 0o100644 << 16
            data = dataset_json_bytes(descriptor) if name=="dataset.json" or name.startswith("snapshots/") else dataset_path(root,name).read_bytes()
            output.writestr(info,data)
    return {"archive":str(archive),"files":len(references)+2,"bytes":archive.stat().st_size,
            "snapshot_id":descriptor["snapshot_id"]}


def add_image_dataset_commands(parser):
    actions = parser.add_subparsers(dest="image_action",required=True)
    prepare = actions.add_parser("prepare",help="prepare once; repeat to resume (default: one PDF/two pages)")
    prepare.add_argument("--input-dir",default="data")
    prepare.add_argument("--output-dir",required=True)
    prepare.add_argument("--config",help="JSON ImageDatasetConfig; no OCR/model settings")
    prepare.add_argument("--books",nargs="+",default=[])
    prepare.add_argument("--pages",default="1-2",help="1-based selection; empty string selects all pages")
    prepare.add_argument("--limit",type=int,default=2,help="total page limit; 0 means unlimited")
    prepare.add_argument("--max-books",type=int,default=1,help="0 means all selected PDFs")
    prepare.add_argument("--profile-dir",help="reuse only dpi/reading_direction from OCR book profiles")
    validate = actions.add_parser("validate",help="verify hashes, pixels, geometry, pages and references")
    validate.add_argument("dataset")
    package = actions.add_parser("package",help="package one validated snapshot for offline Kaggle input")
    package.add_argument("dataset")
    package.add_argument("archive")


def run_image_dataset_command(args):
    if args.image_action=="prepare":
        config = ImageDatasetConfig(**json.loads(Path(args.config).read_text(encoding="utf-8-sig"))) if args.config else ImageDatasetConfig()
        result = prepare_image_dataset(args.input_dir,args.output_dir,config=config,books=args.books,pages=args.pages,
            limit=None if args.limit==0 else args.limit,max_books=None if args.max_books==0 else args.max_books,
            profile_dir=args.profile_dir)
    elif args.image_action=="validate":
        result = validate_image_dataset(args.dataset)
    else:
        result = package_image_dataset(args.dataset,args.archive)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    add_image_dataset_commands(parser)
    run_image_dataset_command(parser.parse_args())
