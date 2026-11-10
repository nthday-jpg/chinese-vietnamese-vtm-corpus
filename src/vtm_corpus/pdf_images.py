"""Acquire visible PDF pages without throwing away native scan pixels.

Native extraction is deliberately conservative. A sole image is insufficient:
placement, page boxes, clips, annotations and a visible-render equality check
must all pass. The original PDF remains the authoritative source representation.
"""
from __future__ import annotations

import math


PDF_IMAGE_VERSION = "pdfium-visible-v1"


def pdf_pixel_to_page(size, bbox, rotation):
    """Six affine coefficients: x'=a*x+c*y+e, y'=b*x+d*y+f.

    Input pixels have a top-left origin; output is unrotated PDF canvas units.
    PDFium does not expose /UserUnit: do not label these physical DPI values.
    """
    w, h = size
    l, b, r, t = bbox
    if rotation == 90:
        return [0, (t-b)/w, (r-l)/h, 0, l, b]
    if rotation == 180:
        return [-(r-l)/w, 0, 0, (t-b)/h, r, b]
    if rotation == 270:
        return [0, -(t-b)/w, -(r-l)/h, 0, r, t]
    return [(r-l)/w, 0, 0, -(t-b)/h, l, t]


def pdf_visible_render(page, scale, max_pixels):
    w, h = page.get_size()
    size = [math.ceil(w*scale), math.ceil(h*scale)]
    if min(size) < 1 or size[0]*size[1] > max_pixels:
        raise ValueError(f"Page render {size} exceeds max_source_pixels={max_pixels}; "
                         "review acquisition settings (source is never silently reduced)")
    bitmap = page.render(scale=scale, draw_annots=True)
    try:
        return bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()


def acquire_pdf_image(page, *, dpi=200, max_source_pixels=100_000_000, strategy="auto"):
    """Return owned RGB pixels and evidence for this page's acquisition decision.

    `dpi` is the rendering fallback in pixels per 72 PDF canvas units. For a
    raster-only full-page scan use its native density, including when raw decode
    fails validation. Mixed/vector pages use at least the requested resolution.
    """
    import pypdfium2 as pdfium
    from PIL import ImageChops

    if strategy not in {"auto", "render"} or dpi <= 0 or max_source_pixels <= 0:
        raise ValueError("Invalid PDF acquisition options")
    width, height = page.get_size()
    bbox, rotation = list(page.get_bbox()), page.get_rotation()
    objects = list(page.get_objects())
    annotations = pdfium.raw.FPDFPage_GetAnnotCount(page)
    images, density = [], 0.0
    for obj in objects:
        if obj.type != pdfium.raw.FPDF_PAGEOBJ_IMAGE:
            continue
        pixels = obj.get_px_size()
        matrix = obj.get_matrix()
        container = obj.container
        while container is not None:
            matrix = matrix.multiply(container.get_matrix())
            container = container.container
        axes = [math.hypot(matrix.a, matrix.b), math.hypot(matrix.c, matrix.d)]
        corners = [matrix.on_point(x,y) for x,y in ((0,0),(1,0),(1,1),(0,1))]
        bounds = [min(c[0] for c in corners),min(c[1] for c in corners),
                  max(c[0] for c in corners),max(c[1] for c in corners)]
        full = (
            max(0, min(bbox[2], bounds[2])-max(bbox[0], bounds[0])) *
            max(0, min(bbox[3], bounds[3])-max(bbox[1], bounds[1])) >=
            .8*(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        native = max(pixels[0]/axes[0], pixels[1]/axes[1]) if min(axes) > 0 else 0
        # Include partial-page/multiple scans as well: a DPI chosen from only a
        # full-page image could silently reduce detail in another visible image.
        density = max(density, native)
        images.append({"native_size": list(pixels), "bounds_pdf": bounds,
                       "matrix_pdf": [matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f],
                       "level": obj.level, "full_page_scan": full,
                       "pixels_per_72_canvas_units": 72*native})
    reasons = []
    simple = len(objects) == 1 and objects[0].type == pdfium.raw.FPDF_PAGEOBJ_IMAGE
    obj = objects[0] if simple else None
    if not simple:
        reasons.append("not_one_top_level_image; preserve_composition")
    if rotation:
        reasons.append("page_rotation")
    if annotations != 0:
        reasons.append("page_annotations_or_unknown")
    if simple:
        m = obj.get_matrix()
        l, b, r, t = bbox
        if (obj.level or abs(m.b) > 1e-6 or abs(m.c) > 1e-6 or m.a <= 0 or m.d <= 0
                or any(abs(a-v) > .001 for a, v in zip(obj.get_bounds(), bbox))
                or any(abs(a-v) > .001 for a, v in zip(
                    [m.e, m.f, m.a, m.d], [l, b, r-l, t-b]))):
            reasons.append("placement_crop_rotation_or_reflection")
        clip = pdfium.raw.FPDFPageObj_GetClipPath(obj)
        # PDFium returns -1 for the non-null empty clip handle on ordinary
        # images. An explicit path is a reason to render; final pixel equality
        # remains mandatory when no clip is reported.
        if clip and pdfium.raw.FPDFClipPath_CountPaths(clip) > 0:
            reasons.append("image_clip_path")

    image, method, validation = None, "pdf_render", None
    if strategy == "auto" and not reasons:
        nw, nh = obj.get_px_size()
        if nw*nh > max_source_pixels:
            raise ValueError("Native image exceeds max_source_pixels")
        # Small fractional PDF dimensions can otherwise add a raster row.
        scale = math.nextafter(min(nw/width, nh/height), 0)
        if abs(nw/width-nh/height) > 1e-4*max(nw/width, nh/height):
            reasons.append("anisotropic_native_geometry")
        else:
            bitmap = obj.get_bitmap(render=False)
            try:
                candidate = bitmap.to_pil().convert("RGB")
            finally:
                # Conversion above owns its data; no PIL view survives closing.
                if hasattr(bitmap, "warn_on_close"):
                    bitmap.warn_on_close = False
                bitmap.close()
            visible = pdf_visible_render(page, scale, max_source_pixels)
            equal = candidate.size == visible.size and ImageChops.difference(candidate, visible).getbbox() is None
            validation = {"method": "full_pixel_equality_to_visible_render", "equal": equal,
                          "render_size": list(visible.size), "scale": scale}
            if equal:
                image, method = candidate, "native_validated"
            else:
                reasons.append("native_pixels_differ_from_visible_page")
    if image is None:
        # Do not inflate tiny scans just because their nominal page size is small.
        raster_only = bool(images) and all(o.type in {pdfium.raw.FPDF_PAGEOBJ_IMAGE,pdfium.raw.FPDF_PAGEOBJ_FORM}
                                          for o in objects) and annotations == 0
        scale = density if raster_only and density else max(dpi/72, density)
        image = pdf_visible_render(page, scale, max_source_pixels)
    else:
        scale = validation["scale"]
    meta = {"version": PDF_IMAGE_VERSION, "method": method, "size": list(image.size),
            "page_size_pdf": [width, height], "page_bbox_pdf": bbox,
            "media_box_pdf": list(page.get_mediabox()), "crop_box_pdf": list(page.get_cropbox()),
            "rotation": rotation, "object_types": [o.type for o in objects],
            "annotations": annotations, "embedded_images": images,
            "requested_dpi": dpi, "render_scale": scale, "native_validation": validation,
            "fallback_reasons": reasons, "source_downsampled": False,
            "source_to_pdf": pdf_pixel_to_page(image.size, bbox, rotation),
            "resolution_unit": "pixels_per_72_pdf_canvas_units; physical_UserUnit_unavailable",
            "warnings": (["low_native_resolution; interpolation_cannot_recover_detail"]
                         if images and max(max(i["native_size"]) for i in images) <= 750 else [])}
    return image, meta
