"""Shared method identities for preparation and read-only OCR diagnostics.

Groups describe mechanisms, never independent evidence or OCR correctness.
"""
from __future__ import annotations


VARIANT_SPECS = {
    'docentr_binarization': (1, 'neural_restoration', 'neural_restoration', 'DocEnTR released base/8 H-DIBCO2018 ViT reconstruction', 'learned_prior_sensitivity'),
    'degan_binarization': (1, 'neural_restoration', 'neural_restoration', 'DE-GAN released U-Net 1024; Keras Torch backend; threshold .95', 'learned_prior_sensitivity'),
    'degan_deblurring': (1, 'neural_restoration', 'neural_restoration', 'DE-GAN released U-Net 1024 deblur; Keras Torch backend', 'learned_prior_sensitivity'),
    'degan_watermark_removal': (1, 'neural_restoration', 'neural_restoration', 'DE-GAN released U-Net 512 watermark removal; explicit region review', 'learned_prior_sensitivity'),
    'original': (1, 'source_observation', 'source', 'canonical RGB pixels', 'reference'),
    'clahe': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'CLAHE Lab L', 'conservative_probe'),
    'bicubic_x2': (2, 'deterministic_transformation', 'geometric_scale_interpolation', 'Pillow bicubic x2', 'conservative_probe'),
    'bicubic_x4': (4, 'deterministic_transformation', 'geometric_scale_interpolation', 'Pillow bicubic x4', 'conservative_probe'),
    'denoised': (1, 'deterministic_transformation', 'denoising_sharpening', 'NLM Lab L', 'glyph_sensitivity'),
    'grayscale_luma': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'RGB luminance projection', 'glyph_sensitivity'),
    'grayscale_red': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'RGB R projection', 'selective_information'),
    'grayscale_green': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'RGB G projection', 'selective_information'),
    'grayscale_blue': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'RGB B projection', 'selective_information'),
    'lab_l': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'CIELab L projection', 'glyph_sensitivity'),
    'lab_a': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'CIELab a chroma diagnostic', 'selective_information'),
    'lab_b': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'CIELab b chroma diagnostic', 'selective_information'),
    'red_ink': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'RGB red dominance mask', 'selective_information'),
    'background_normalized': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'Gaussian background division Lab L', 'glyph_sensitivity'),
    'sauvola': (1, 'deterministic_transformation', 'binarization', 'Sauvola reflected local moments', 'glyph_sensitivity'),
    'niblack': (1, 'deterministic_transformation', 'binarization', 'Niblack mean-minus-k-std; equality is background', 'glyph_sensitivity'),
    'bilateral': (1, 'deterministic_transformation', 'denoising_sharpening', 'bilateral Lab L', 'glyph_sensitivity'),
    'unsharp': (1, 'deterministic_transformation', 'denoising_sharpening', 'thresholded unsharp Lab L', 'glyph_sensitivity'),
    'retinex_msr': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'three-scale log Retinex intensity; percentile mapping', 'glyph_sensitivity'),
    'retinex_msrcp': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'three-scale MSRCP; common RGB gain with saturation bound', 'glyph_sensitivity'),
    'wolf': (1, 'deterministic_transformation', 'binarization', 'Wolf-Jolion global-normalized local threshold', 'glyph_sensitivity'),
    'nick': (1, 'deterministic_transformation', 'binarization', 'NICK mean+k*sqrt(E[I^2]+variance)', 'glyph_sensitivity'),
    'color_pca': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'RGB principal-component projection; luminance sign', 'selective_information'),
    'color_gmm': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'diagonal Lab GMM EM; brightest component is paper', 'selective_information'),
    'mean_shift': (1, 'deterministic_transformation', 'denoising_sharpening', 'OpenCV spatial-color mean shift, no pyramid by default', 'glyph_sensitivity'),
    'anisotropic_diffusion': (1, 'deterministic_transformation', 'denoising_sharpening', 'Perona-Malik exp conductivity; 4 neighbors; no-flux Lab L', 'glyph_sensitivity'),
    'background_morphological': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'grayscale closing surface; divide original Lab L', 'glyph_sensitivity'),
    'background_guided': (1, 'deterministic_transformation', 'deterministic_color_contrast', 'closing surface self-guided filter; divide original Lab L', 'glyph_sensitivity'),
    'background_otsu': (1, 'deterministic_transformation', 'binarization', 'hypothesis: closing surface -> division -> Otsu', 'glyph_sensitivity'),
    'realesrgan_x4': (4, 'neural_restoration', 'neural_restoration', 'Real-ESRGAN RRDBNet x4plus', 'learned_prior_sensitivity'),
    'tadisr_x4': (4, 'neural_restoration', 'neural_restoration', 'TADiSR CogView4 RealCE joint segmentation decoder', 'learned_prior_sensitivity'),
    'difftsr': (None, 'neural_restoration', 'neural_restoration', 'DiffTSR IDM/TDM/MoM and TransOCR text prior', 'learned_prior_sensitivity'),
    'unihir_restoration': (1, 'generative_reconstruction', 'generative_reconstruction', 'UniHIR BAGEL MLLM reconstruction', 'generative_sensitivity'),
    'docres_appearance': (1, 'neural_restoration', 'neural_restoration', 'DocRes Restormer DTSPrompt appearance', 'learned_prior_sensitivity'),
    'docres_deblurring': (1, 'neural_restoration', 'neural_restoration', 'DocRes Restormer DTSPrompt deblurring', 'learned_prior_sensitivity'),
    'docres_deshadowing': (1, 'neural_restoration', 'neural_restoration', 'DocRes Restormer DTSPrompt deshadowing', 'learned_prior_sensitivity'),
    'docres_binarization': (1, 'neural_restoration', 'neural_restoration', 'DocRes Restormer DTSPrompt binarization', 'learned_prior_sensitivity'),
    'docdiff_deblurring': (1, 'neural_restoration', 'neural_restoration', 'DocDiff coarse predictor and linear-100 residual diffusion', 'learned_prior_sensitivity'),
    'restormer_motion_deblurring': (1, 'neural_restoration', 'neural_restoration', 'Restormer MDTA/GDFN motion checkpoint', 'learned_prior_sensitivity'),
    'restormer_defocus_deblurring': (1, 'neural_restoration', 'neural_restoration', 'Restormer MDTA/GDFN single-image defocus checkpoint', 'learned_prior_sensitivity'),
    'restormer_real_denoising': (1, 'neural_restoration', 'neural_restoration', 'Restormer MDTA/GDFN SIDD checkpoint', 'learned_prior_sensitivity'),
    'restormer_gaussian_denoising': (1, 'neural_restoration', 'neural_restoration', 'Restormer MDTA/GDFN blind Gaussian checkpoint', 'learned_prior_sensitivity'),
}

# Observed 2026-11-12; classify by content hash, never filename alone.
SURVEY_SOURCE_GROUPS = {
    '39d1b637eed99a6e1981c326e876537ccf96baa897e16f1dca4f6aec01e32800': 'G1',
    '3e9a2be8a104ad044cd054dee7c0a98422aa57d2c9a4ceaea985b8d9a56a0d0f': 'G1',
    'ba8129a51d43a81a2ab566153596ebe482c72a570a0c9f319a69d79c95dde7b5': 'G1',
    '68b6accd0c037854e41e65ed45fe1ebcd27889999a75ae3df2bfc597275107e7': 'G1',
    '66fd3918224508aa4dffa5cecc5abe741094c563569a61ea9e976f893811b38e': 'G2',
    '4b26a2be1db3c95fbd4d4387fd72c417525a813f5ed2c6a6787e02a08163121a': 'G3',
}

UNAVAILABLE_RESTORATION_METHODS = {
    'lwtformer': {'state':'not_integrated', 'reason':'Official repository has architecture/test code but no verified released pretrained checkpoint',
                  'upstream':'https://github.com/INWLY/LWTformer'},
    'gladunet': {'state':'not_integrated', 'reason':'No accessible official inference implementation and pretrained checkpoint verified',
                 'upstream':'https://www.nature.com/articles/s40494-025-01965-8'},
    'gatos_energy': {'state':'not_integrated', 'reason':'No pinned reliable complete upstream implementation; background_otsu is a separate implemented hypothesis'},
    'single_image_bleed_reduction': {'state':'not_integrated', 'reason':'No verified dedicated implementation and no established bleed-through labels in the surveyed corpus'},
}


def variant_spec(name):
    if name == 'tadisr_text_mask':
        return dict(method_family='neural_restoration', algorithm='TADiSR predicted segmentation mask',
                    perturbation_class='diagnostic_only', provenance='neural_restoration', scale=4)
    scale, provenance, group, algorithm, interpretation = VARIANT_SPECS[name]
    return dict(method_family=group, algorithm=algorithm, perturbation_class=interpretation,
                provenance=provenance, scale=scale)


def preparation_applicability(name, book, page, region, config):
    """Explicit analyst conditions only; never invent a degradation detector."""
    rule = config.get('applicability', {}).get(name, {})
    stem = str(book).removesuffix('.pdf')
    if rule.get('books') is not None and not {str(book), stem}.intersection(rule['books']):
        return 'not_applicable', 'book outside explicit method applicability'
    if rule.get('crop_types') is not None and region.get('crop_type') not in rule['crop_types']:
        return 'not_applicable', 'crop type outside explicit method applicability'
    review = preparation_region_review(book, page, region['id'], config)
    if rule.get('require_review') and name not in review.get('approved_variants', []):
        return 'not_applicable', 'requires explicit region approval; degradation unverified'
    return 'ready', None


def preparation_region_review(book, page, region_id, config):
    reviews = config.get('reviewed_regions', {})
    return reviews.get(f'{book}:{page+1}:{region_id}',
        reviews.get(f'{str(book).removesuffix(".pdf")}:{page+1}:{region_id}', {}))


def validate_applicability(config):
    for name, rule in config.get('applicability', {}).items():
        if name not in VARIANT_SPECS or not isinstance(rule, dict) or set(rule)-{'books', 'crop_types', 'require_review'}:
            raise ValueError('Unknown method/applicability option')
        for key in ('books', 'crop_types'):
            if key in rule and (not isinstance(rule[key], list) or any(not isinstance(v, str) for v in rule[key])):
                raise ValueError('Applicability selections must be lists of strings')
        if set(rule.get('crop_types', []))-{'page', 'group', 'column'}:
            raise ValueError('Invalid applicability crop_types')
        if 'require_review' in rule and type(rule['require_review']) is not bool:
            raise ValueError('require_review must be boolean')
