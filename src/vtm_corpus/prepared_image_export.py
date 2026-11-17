"""Readable image folders shared by the local dataset merge tools."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import uuid
from .image_dataset import dataset_open,dataset_read,dataset_hash,dataset_path,dataset_write,dataset_json_bytes,dataset_file_hash
from .restoration_registry import VARIANT_SPECS
from .dataset_merge import merge_copy_asset
SUPPORTED_IMAGE_EXPORTS = set(VARIANT_SPECS) | {'tadisr_text_mask'}

def open_prepared_export(directory):
    root,base=dataset_open(directory)
    extension=base.get('restoration')
    if not extension or extension['version']!=1:raise ValueError('Missing/unsupported restoration extension')
    report=json.loads(dataset_read(root,extension['report']))
    return root,base,report

def prepared_image_export_plan(directory,*,export_context=None):
    """Name successful snapshot images by method/book/page/crop, with portable paths."""
    import re, unicodedata
    root, descriptor, report = open_prepared_export(directory) if export_context is None else export_context

    def book_slug(book):
        text = unicodedata.normalize('NFC', str(book))
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text).strip(' .')[:64].rstrip(' .') or 'book'
        if re.fullmatch('(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', text.split('.')[0]):
            text = '_' + text
        return text
    books = {(p['book'], p['source_sha256']) for p in descriptor['pages']}
    groups = {}
    for identity in sorted(books):
        groups.setdefault(book_slug(identity[0]).casefold(), []).append(identity)
    names = {}
    for identities in groups.values():
        for identity in identities:
            suffix = '--' + dataset_hash(list(identity))[:8] if len(identities) > 1 else ''
            names[identity] = book_slug(identity[0]) + suffix
    entries = []
    seen = set()
    methods = dict.fromkeys(report.get('selected_variants',[]),0)
    skipped = {}
    export_rows=report.get('export_rows')
    rows=export_rows if export_rows is not None else (json.loads(line) for line in dataset_read(root,report['manifest']).splitlines() if line.strip())
    for row in rows:
        if row['state'] != 'ok':
            skipped[row['state']] = skipped.get(row['state'], 0) + 1
            continue
        method = row['variant']
        if method not in SUPPORTED_IMAGE_EXPORTS:
            raise ValueError(f'Unknown image export method: {method}')
        page = row['page']
        order = row['reading_order']
        kind = row['crop_type']
        if not isinstance(page, int) or page < 1 or (not isinstance(order, int)) or (order < 0):
            raise ValueError('Image export requires 1-based pages and nonnegative crop order')
        if kind not in ('page', 'group', 'column'):
            raise ValueError(f'Unknown export crop type: {kind}')
        suffix = '' if kind == 'page' else f'-{kind}{order + 1:03d}'
        book = names[row['book'], row['source_sha256']]
        relative = f'{method}/{book}-p{page:05d}{suffix}.png'
        if relative.casefold() in seen:
            raise ValueError(f'Duplicate readable image name: {relative}')
        seen.add(relative.casefold())
        source = dataset_path(root, row['image']['path'])
        if source.suffix.lower() != '.png' or not source.is_file():
            raise ValueError(f'Missing PNG: {source}')
        entries.append({'name': relative, 'source': source, 'sha256': row['image']['sha256']})
        methods[method] = methods.get(method, 0) + 1
    if not entries:
        raise ValueError('No successful selected images to export')
    return (root, sorted(entries, key=lambda entry: entry['name']), {'files': len(entries), 'methods': dict(sorted(methods.items())), 'skipped': skipped, 'partial': report['partial'], 'snapshot_id': descriptor['snapshot_id']})

def export_prepared_image_folders(directory, destination,*,export_context=None,verify_image_hashes=False,reserve_bytes=2*1024**3):
    """Refresh only managed images; keep snapshot/cache files under the dataset root."""
    root,entries,summary=prepared_image_export_plan(directory,export_context=export_context)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    receipt_root=destination/'_dataset'
    receipt=f'exports/{dataset_hash(str(destination))}.json'
    try:previous=json.loads(dataset_path(receipt_root,receipt).read_text(encoding='utf8'))
    except FileNotFoundError:previous={}
    if not isinstance(previous,dict):raise ValueError('Invalid image export receipt')
    owned=previous.get('files',{})
    if not isinstance(owned,dict):raise ValueError('Invalid image export file list')
    for relative in owned:
        parts=Path(relative).parts
        if len(parts)!=2 or parts[0] not in SUPPORTED_IMAGE_EXPORTS or Path(relative).suffix.lower()!='.png':
            raise ValueError(f'Invalid managed image path: {relative}')
        dataset_path(destination,relative)
    for method in summary.get('methods',{}):
        if method not in SUPPORTED_IMAGE_EXPORTS:raise ValueError('Unsupported image export method: '+method)
        (destination/method).mkdir(parents=True,exist_ok=True)
    current={entry['name']:entry['sha256'] for entry in entries}
    # Validate every source and every replacement before modifying readable output.
    for entry in entries:
        if verify_image_hashes and dataset_file_hash(entry['source'])!=entry['sha256']:
            raise ValueError(f"Image checksum mismatch: {entry['source']}")
        target=dataset_path(destination,entry['name'])
        if target.exists():
            digest=dataset_file_hash(target)
            if digest not in (entry['sha256'],owned.get(entry['name'])):
                raise FileExistsError(f'Unmanaged or edited image already exists: {target}')
    for entry in entries:
        target=dataset_path(destination,entry['name']);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and dataset_file_hash(target)==entry['sha256']:continue
        temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            try:os.link(entry['source'],temporary) # Immutable objects; avoid duplicate storage where supported.
            except OSError:merge_copy_asset(entry['source'],temporary,reserve_bytes=reserve_bytes)
            temporary.replace(target)
        finally:
            if temporary.exists():temporary.unlink()
    # A new selection removes only unchanged files owned by the previous export.
    for relative,digest in owned.items():
        if relative in current or summary.get('partial'):continue
        target=dataset_path(destination,relative)
        if target.is_file() and dataset_file_hash(target)==digest:
            target.unlink()
            try:target.parent.rmdir()
            except OSError:pass
    dataset_write(receipt_root,receipt,dataset_json_bytes({'destination':str(destination),'files':{**owned,**current} if summary.get('partial') else current,'partial':summary.get('partial',False)}))
    return {**summary,'directory':str(destination)}
