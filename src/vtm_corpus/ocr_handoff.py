"""Portable original-image/OCR checkpoint handoff between the two Kaggle notebooks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import uuid
import zipfile

from .image_dataset import dataset_open, dataset_read, dataset_hash, dataset_json_bytes, dataset_write
from .ocr_records import read_checkpoint_campaign
from .ocr_checkpoint import restore_previous_ocr
from .pipeline import select_entries, parse_pages
from .config import ocr_model_excluded


def discover_original_ocr_handoff(data_dir):
    """Find exactly one marked ZIP or extracted package alongside the PDFs."""
    root = Path(data_dir)
    readable = root / 'original_ocr_readable'
    if (readable / 'manifest.json').is_file():
        return readable
    found = []
    for marker in root.rglob('original_ocr_handoff.json'):
        if json.loads(marker.read_text(encoding='utf8')).get('format') == 'vtm-original-ocr-handoff':
            found.append(marker.parent)
    for path in root.rglob('*.zip'):
        with zipfile.ZipFile(path) as archive:
            if 'original_ocr_handoff.json' in archive.namelist():
                info = json.loads(archive.read('original_ocr_handoff.json'))
                if info.get('format') == 'vtm-original-ocr-handoff':
                    found.append(path)
    if len(found) != 1:
        raise ValueError(f'DATA_DIR needs exactly one original_ocr_handoff.zip or extracted package; found {len(found)}')
    return found[0]


def package_original_ocr_handoff(manifest_path, prepared_dataset, archive_path, *, extras=None):
    """Package lossless canonical assets and the complete original OCR campaign graph.

    Image assets are referenced and streamed once, without recompression. Runtime,
    model weights and duplicate report images are excluded. No inference occurs.
    """
    root, descriptor = dataset_open(prepared_dataset)
    canonical = {k:v for k,v in descriptor.items() if k not in {'restoration','snapshot_id'}}
    canonical['snapshot_id'] = dataset_hash(canonical)
    files, payloads, visited, verified_assets = {}, {}, {}, {}

    def asset_references(value):
        if isinstance(value, dict):
            if {'path','sha256'} <= value.keys():
                relative = value['path']
                name = 'images/'+relative
                if name not in files:
                    # Verify JSON and stream hashes for PNGs, avoiding a corpus-size buffer.
                    path = (root/relative).resolve()
                    if not path.is_relative_to(root.resolve()):
                        raise ValueError('Canonical asset escapes dataset')
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda:stream.read(8*1024*1024), b''):digest.update(block)
                    if (value.get('bytes') is not None and path.stat().st_size != value['bytes']) or digest.hexdigest() != value['sha256']:
                        raise ValueError('Canonical asset hash mismatch: '+relative)
                    files[name] = path
                    verified_assets[name]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
                    if path.suffix == '.json':asset_references(json.loads(path.read_text(encoding='utf8')))
                return
            for item in value.values():asset_references(item)
        elif isinstance(value, list):
            for item in value:asset_references(item)
    asset_references(canonical)
    raw = dataset_json_bytes(canonical)
    payloads['images/dataset.json'] = raw
    payloads[f"images/snapshots/{canonical['snapshot_id']}.json"] = raw

    def campaign(path, top=False):
        path = Path(path).resolve()
        if path in visited:return visited[path]
        name = 'manifest.json' if top else 'campaign_'+hashlib.sha256(str(path).encode()).hexdigest()[:20]+'.json'
        visited[path] = name
        data = read_checkpoint_campaign(path)
        if set(data['models']) != {'ppocr','tonggu','paddle','hunyuan'}:
            raise ValueError('Handoff requires exactly four OCR models')
        if data.get('variants') and data['variants'] != ['original']:
            raise ValueError('Handoff must contain original-only OCR')
        for field in ('crop_manifests','variant_manifests'):
            if data.get(field):data[field] = {k:campaign(v) for k,v in data[field].items()}
        for pair in data.get('pairs', []):
            if pair.get('variant', 'original') != 'original':raise ValueError('Non-original OCR in handoff')
            for engine in data['models']:
                directory = Path(pair[engine]).resolve()
                run = 'runs/'+hashlib.sha256(str(directory).encode()).hexdigest()[:20]
                metadata = json.loads((directory/'run_manifest.json').read_text(encoding='utf8'))
                if metadata['config'].get('image_variant') != 'original':raise ValueError('Non-original run in handoff')
                for filename in ('run_manifest.json','tasks.jsonl'):
                    files[f'ocr/{run}/{filename}'] = directory/filename
                for path in (directory/'ocr_raw').rglob('*.jsonl'):
                    files[f'ocr/{run}/ocr_raw/{path.relative_to(directory/"ocr_raw").as_posix()}'] = path
                pair[engine] = run
        payloads['ocr/'+name] = dataset_json_bytes(data)
        return name
    campaign(manifest_path, top=True)
    for name,path in (extras or {}).items():files['analysis/'+name] = Path(path)
    inventory = {}
    for name,path in files.items():
        if name in verified_assets:
            inventory[name]=verified_assets[name]
            continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024), b''):digest.update(block)
        inventory[name] = dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
    inventory.update({name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in payloads.items()})
    info = dict(format='vtm-original-ocr-handoff',version=1,manifest='ocr/manifest.json',dataset='images/dataset.json',
                snapshot_id=canonical['snapshot_id'],files=inventory,
                preservation='original PNG bytes, canonical geometry, full raw OCR and model settings; no weights/runtime')
    archive_path=Path(archive_path);archive_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=archive_path.with_name(archive_path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for name,path in sorted(files.items()):
                archive.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix.lower()=='.png' else zipfile.ZIP_DEFLATED,compresslevel=1)
            for name,data in sorted(payloads.items()):archive.writestr(name,data)
            archive.writestr('original_ocr_handoff.json',dataset_json_bytes(info))
        temporary.replace(archive_path)
    finally:
        if temporary.exists():temporary.unlink()
    return dict(archive=str(archive_path),files=len(inventory),bytes=archive_path.stat().st_size,snapshot_id=canonical['snapshot_id'])


def restore_original_ocr_handoff(source, destination):
    """Validate every member, then publish a writable copy; never edit DATA_DIR."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Handoff source and writable destination must be separate')
    if source.is_dir() and (source / 'manifest.json').is_file():
        return restore_readable_ocr(source, destination)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        raw = archive.read('original_ocr_handoff.json') if archive else (source/'original_ocr_handoff.json').read_bytes()
        info = json.loads(raw)
        if info.get('format') != 'vtm-original-ocr-handoff' or info.get('version') != 1:
            raise ValueError('Unsupported original OCR handoff')
        fingerprint = hashlib.sha256(raw).hexdigest()
        target=destination/fingerprint[:20]
        for name,expected in info['files'].items():
            pure=PurePosixPath(name)
            if pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe handoff member: '+name)
            path=(target/name).resolve()
            if not path.is_relative_to(target):raise ValueError('Handoff path escapes destination')
            if path.exists():
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024), b''):digest.update(block)
                if path.stat().st_size == expected['bytes'] and digest.hexdigest() == expected['sha256']:continue
                raise ValueError('Writable handoff copy changed; use a fresh WORK_DIR: '+name)
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.tmp')
            digest=hashlib.sha256();count=0
            if not archive and not (source/name).resolve().is_relative_to(source):raise ValueError('Handoff source symlink escapes package')
            stream=archive.open(name) if archive else (source/name).open('rb')
            try:
                with stream,temporary.open('wb') as output:
                    for block in iter(lambda:stream.read(8*1024*1024), b''):
                        output.write(block);digest.update(block);count+=len(block)
                if count != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                    raise ValueError('Handoff member hash mismatch: '+name)
                temporary.replace(path)
            finally:
                if temporary.exists():temporary.unlink()
        for name in (info['manifest'],info['dataset']):
            if name not in info['files']:raise ValueError('Handoff entrypoint missing from inventory')
        dataset_open(target/info['dataset'])
        (target/'original_ocr_handoff.json').write_bytes(raw)
        return dict(directory=str(target),manifest=str(target/info['manifest']),
                    dataset=str(target/info['dataset']),snapshot_id=info['snapshot_id'],fingerprint=fingerprint)
    finally:
        if archive:archive.close()


def _portable_member(root, name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe OCR member: ' + name)
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('OCR member escapes package: ' + name)
    return path


def _file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def convert_original_ocr_readable(source, destination):
    """Make a standalone model/book package, preserving raw history and all assets."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Readable destination must be separate from source')
    info = json.loads((source / 'original_ocr_handoff.json').read_text(encoding='utf8'))
    if info.get('format') != 'vtm-original-ocr-handoff' or info.get('version') != 1:
        raise ValueError('Unsupported original OCR handoff')
    # Verify before conversion; no inference, normalization or record deduplication.
    for name, expected in info['files'].items():
        if _file_identity(_portable_member(source, name)) != expected:
            raise ValueError('Handoff member hash mismatch: ' + name)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'manifest.json'
    if marker.exists():
        raise FileExistsError('Readable package already exists: ' + str(marker))
    inventory, campaigns, books, runs = {}, {}, {}, {}
    folders = {'ppocr': 'ppocr', 'tonggu': 'tonggu', 'paddle': 'paddle_vl', 'hunyuan': 'hunyuan'}

    def visit(name):
        if name in campaigns:
            return
        data = json.loads(_portable_member(source, name).read_text(encoding='utf8'))
        campaigns[name] = data
        for field in ('crop_manifests', 'variant_manifests'):
            for ref in data.get(field, {}).values():
                visit('ocr/' + ref)
        for pair in data.get('pairs', []):
            for model in data['models']:
                if ocr_model_excluded(model,pair['book']):
                    pair[model]=dict(excluded=True,reason='horizontal_book_not_supported')
                    continue
                run = pair[model]
                if run not in runs:
                    book = str(pair['book'])
                    filename = folders[model] + '/' + book + '.json'
                    _portable_member(destination, filename)
                    payload = books.setdefault(filename, dict(model=model, book=book,
                        page_numbering='zero-based', records=[], checkpoints=[]))
                    directory = _portable_member(source, 'ocr/' + run)
                    metadata = json.loads((directory / 'run_manifest.json').read_text(encoding='utf8'))
                    tasks = [json.loads(line) for line in (directory / 'tasks.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
                    shards = {}
                    for shard in sorted((directory / 'ocr_raw').rglob('*.jsonl')):
                        indices = []
                        for line in shard.read_text(encoding='utf8').splitlines():
                            if line.strip():
                                indices.append(len(payload['records']))
                                payload['records'].append(json.loads(line))
                        shards[shard.relative_to(directory / 'ocr_raw').as_posix()] = indices
                    index = len(payload['checkpoints'])
                    payload['checkpoints'].append(dict(metadata=metadata, tasks=tasks, shards=shards))
                    runs[run] = dict(file=filename, checkpoint=index)
                pair[model] = runs[run]
    visit(info['manifest'])
    for filename, payload in books.items():
        # Keep shard order through explicit indices, even with duplicate/retry records.
        order = sorted(range(len(payload['records'])), key=lambda i: payload['records'][i].get('page', -1))
        positions = {old: new for new, old in enumerate(order)}
        payload['records'] = [payload['records'][i] for i in order]
        for checkpoint in payload['checkpoints']:
            checkpoint['shards'] = {name: [positions[i] for i in indices]
                                    for name, indices in checkpoint['shards'].items()}
        path = _portable_member(destination, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        inventory[filename] = _file_identity(path)
    for name, expected in info['files'].items():
        if not name.startswith('images/'):
            continue
        target = _portable_member(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_portable_member(source, name), target)
        if _file_identity(target) != expected:
            raise ValueError('Copied image asset hash mismatch: ' + name)
        inventory[name] = expected
    result = dict(format='vtm-readable-ocr', version=1, dataset=info['dataset'],
        snapshot_id=info['snapshot_id'], manifest=info['manifest'], campaigns=campaigns, files=inventory)
    # Publish the entrypoint last, only when every required file is present.
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return dict(directory=str(destination), books=len(books), files=len(inventory))


def restore_readable_ocr(source, destination):
    """Validate the clean input and adapt checkpoints only inside Kaggle working storage."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Readable source and destination must be separate')
    raw = (source / 'manifest.json').read_bytes()
    info = json.loads(raw)
    if info.get('format') != 'vtm-readable-ocr' or info.get('version') not in (1, 2):
        raise ValueError('Unsupported readable OCR package')
    fingerprint = hashlib.sha256(raw).hexdigest()
    target = destination / fingerprint[:20]
    payloads = {}
    clean_images = info.get('image_storage')
    for name, expected in info['files'].items():
        path = _portable_member(source, name)
        if _file_identity(path) != expected:
            raise ValueError('Readable OCR member hash mismatch: ' + name)
        if name.startswith('images/'):
            if clean_images:
                continue
            out = _portable_member(target, name)
            if out.exists():
                if _file_identity(out) != expected:
                    raise ValueError('Writable image changed; use a fresh WORK_DIR: ' + name)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
        else:
            payloads[name] = json.loads(path.read_text(encoding='utf8'))
    if clean_images:
        _restore_clean_images(source, target, info)
    seen = {}
    for campaign_name, data in info['campaigns'].items():
        # Parse a private copy: the validated input and its provenance stay intact.
        data = json.loads(json.dumps(data))
        for pair in data.get('pairs', []):
            for model in data['models']:
                ref = pair[model]
                if ref.get('excluded'):
                    if not ocr_model_excluded(model,pair['book']):
                        raise ValueError('Unexpected excluded model/book')
                    template=next(p for p in payloads.values() if p.get('model')==model)
                    peer=payloads['ppocr/'+pair['book']+'.json']
                    metadata=json.loads(json.dumps(template['checkpoints'][0]['metadata']))
                    metadata.update(excluded=True,status='excluded',reason=ref.get('reason'),
                        books=json.loads(json.dumps(peer['checkpoints'][0]['metadata']['books'])),
                        tasks_total=0,tasks_selected=0,counts=dict(pages=0,done=0,missing=0))
                    metadata['config']['books']=[pair['book']+'.pdf']
                    for book_info in metadata['books'].values():book_info['pages']=[]
                    run='excluded/'+model+'/'+pair['book']
                    directory=_portable_member(target,'ocr/'+run)
                    directory.mkdir(parents=True,exist_ok=True)
                    (directory/'run_manifest.json').write_bytes(dataset_json_bytes(metadata))
                    (directory/'tasks.jsonl').write_bytes(b'')
                    pair[model]=run
                    continue
                key = (ref['file'], ref['checkpoint'])
                if key not in seen:
                    payload = payloads[ref['file']]
                    if payload['model'] != model or payload['book'] != str(pair['book']):
                        raise ValueError('Readable OCR model/book mismatch')
                    checkpoint = payload['checkpoints'][ref['checkpoint']]
                    run = 'runs/' + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
                    directory = _portable_member(target, 'ocr/' + run)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / 'run_manifest.json').write_bytes(dataset_json_bytes(checkpoint['metadata']))
                    (directory / 'tasks.jsonl').write_bytes(b''.join(dataset_json_bytes(t) for t in checkpoint['tasks']))
                    for name, indices in checkpoint['shards'].items():
                        shard = _portable_member(directory / 'ocr_raw', name)
                        shard.parent.mkdir(parents=True, exist_ok=True)
                        shard.write_bytes(b''.join(dataset_json_bytes(payload['records'][i]) for i in indices))
                    seen[key] = run
                pair[model] = seen[key]
        campaign = _portable_member(target, campaign_name)
        campaign.parent.mkdir(parents=True, exist_ok=True)
        campaign.write_bytes(dataset_json_bytes(data))
    if info['manifest'] not in info['campaigns'] or (not clean_images and info['dataset'] not in info['files']):
        raise ValueError('Readable OCR entrypoint missing')
    dataset_open(target / info['dataset'])
    return dict(directory=str(target), manifest=str(target / info['manifest']),
        dataset=str(target / info['dataset']), snapshot_id=info['snapshot_id'], fingerprint=fingerprint)


def _restore_clean_images(source, target, info):
    storage = info['image_storage']
    if storage.get('format') != 'by-book-v1' or storage['metadata'] not in info['files']:
        raise ValueError('Unsupported readable image storage')
    metadata = json.loads(_portable_member(source, storage['metadata']).read_text(encoding='utf8'))
    if info['dataset'] not in metadata['inventory']:
        raise ValueError('Canonical dataset missing from image metadata')
    for name, expected in metadata['inventory'].items():
        if not name.startswith('images/'):
            raise ValueError('Invalid canonical image member: ' + name)
        out = _portable_member(target, name)
        if out.exists():
            if _file_identity(out) != expected:
                raise ValueError('Writable image changed; use a fresh WORK_DIR: ' + name)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if name in storage['assets']:
            readable = storage['assets'][name]
            if readable not in info['files']:
                raise ValueError('Image asset missing from inventory: ' + readable)
            shutil.copy2(_portable_member(source, readable), out)
        else:
            entry = metadata['files'][name]
            if entry['kind'] == 'json':
                raw = dataset_json_bytes(entry['value'])
            elif entry['kind'] == 'jsonl':
                raw = b''.join(dataset_json_bytes(row) for row in entry['value'])
            else:
                raise ValueError('Unknown image metadata kind')
            out.write_bytes(raw)
        if _file_identity(out) != expected:
            raise ValueError('Restored canonical image hash mismatch: ' + name)


def clean_readable_images(package, backup_directory):
    """Publish human-readable image paths with a byte-exact canonical reconstruction map."""
    package, backup_directory = Path(package).resolve(), Path(backup_directory).resolve()
    if package == backup_directory or package.is_relative_to(backup_directory) or backup_directory.is_relative_to(package):
        raise ValueError('Image backup must be separate from the readable package')
    marker = package / 'manifest.json'
    info = json.loads(marker.read_text(encoding='utf8'))
    if info.get('format') != 'vtm-readable-ocr' or info.get('version') != 1:
        raise ValueError('Expected readable OCR version 1 before image cleanup')
    if backup_directory.exists():
        raise FileExistsError('Image backup already exists: ' + str(backup_directory))
    inventory = {name: value for name, value in info['files'].items() if name.startswith('images/')}
    metadata, assets, indices = {}, {}, {}
    for name, expected in inventory.items():
        path = _portable_member(package, name)
        if _file_identity(path) != expected:
            raise ValueError('Readable image member hash mismatch: ' + name)
        if path.suffix == '.json':
            entry = dict(kind='json', value=json.loads(path.read_bytes()))
            restored = dataset_json_bytes(entry['value'])
        elif path.suffix == '.jsonl':
            entry = dict(kind='jsonl', value=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()])
            restored = b''.join(dataset_json_bytes(row) for row in entry['value'])
        elif path.suffix == '.png':
            continue
        else:
            raise ValueError('Unsupported canonical image file: ' + name)
        if hashlib.sha256(restored).hexdigest() != expected['sha256'] or len(restored) != expected['bytes']:
            raise ValueError('Image metadata is not canonical JSON; refusing lossy cleanup: ' + name)
        metadata[name] = entry
    stage = package / ('.images-clean-' + uuid.uuid4().hex)
    stage.mkdir()

    def assign(ref, readable):
        name = 'images/' + ref['path']
        if name not in inventory:
            raise ValueError('Canonical image reference missing: ' + name)
        if name not in assets:
            _portable_member(stage, readable)
            assets[name] = 'images/' + readable
        return assets[name].removeprefix('images/')

    descriptor = metadata[info['dataset']]['value']
    for page in descriptor['pages']:
        book = str(page['book'])
        prefix = f"{book}/page_{page['page'] + 1:05d}"
        record = metadata['images/' + page['record']['path']]['value']
        main = assign(record['source'], prefix + '.png')
        item = dict(page=page['page'] + 1, image=main, source_pdf=page.get('source_pdf'),
            source_sha256=page.get('source_sha256'), crops=[], diagnostics={})
        for region in record.get('regions', []):
            for variant, ref in region.get('images', {}).items():
                suffix = '' if variant == 'original' else '_' + variant
                name = assign(ref, f"{book}/crops/page_{page['page'] + 1:05d}_{region['id']}{suffix}.png")
                item['crops'].append(dict(region=region['id'], crop_type=region['crop_type'], variant=variant, image=name))
        layout = metadata['images/' + record['layout']['path']]['value']
        for role in ('preview', 'analysis_mask'):
            if layout.get(role):
                item['diagnostics'][role] = assign(layout[role], f"{book}/diagnostics/page_{page['page'] + 1:05d}_{role}.png")
        indices.setdefault(book, []).append(item)
    remaining = set(name for name in inventory if name.endswith('.png')) - assets.keys()
    if remaining:
        raise ValueError('Unclassified canonical images; refusing to drop them: ' + str(sorted(remaining)[:3]))
    new_inventory = {name: value for name, value in info['files'].items() if not name.startswith('images/')}
    for old_name, readable_name in assets.items():
        relative = readable_name.removeprefix('images/')
        out = _portable_member(stage, relative)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_portable_member(package, old_name), out)
        expected = inventory[old_name]
        if _file_identity(out) != expected:
            raise ValueError('Clean image copy hash mismatch: ' + readable_name)
        new_inventory[readable_name] = expected
    for book, pages in indices.items():
        out = _portable_member(stage, book + '/index.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(book=book, page_numbering='one-based',
            paths_relative_to='images/', pages=pages), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        new_inventory['images/' + book + '/index.json'] = _file_identity(out)
    (stage / 'metadata.json').write_text(json.dumps(dict(inventory=inventory, files=metadata),
        ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    new_inventory['images/metadata.json'] = _file_identity(stage / 'metadata.json')
    info.update(version=2, files=new_inventory, image_storage=dict(format='by-book-v1',
        metadata='images/metadata.json', assets=assets))
    pending = package / 'manifest.pending.json'
    pending.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    backup_directory.mkdir(parents=True)
    shutil.copy2(marker, backup_directory / 'manifest.json')
    (package / 'images').replace(backup_directory / 'images')
    try:
        stage.replace(package / 'images')
        pending.replace(marker)
    except BaseException:
        if (package / 'images').exists():
            (package / 'images').replace(stage)
        (backup_directory / 'images').replace(package / 'images')
        raise
    return dict(books=len(indices), png_files=len(assets), metadata_files=len(metadata),
        directory=str(package / 'images'), backup=str(backup_directory))


def materialize_handoff_dataset(restored, destination, config):
    """Reuse canonical pixels; select books/pages from the handoff without rendering."""
    source, descriptor=dataset_open(restored['dataset'])
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    for path in source.rglob('*'):
        if not path.is_file() or path.name=='dataset.json':continue
        target=destination/path.relative_to(source)
        if target.exists():continue
        target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(path,target)
        except OSError:shutil.copy2(path,target)
    books={p['book'] for p in descriptor['pages']}
    entries=[dict(name=Path(book).stem,stem=Path(book).stem,path=book) for book in sorted(books)]
    entries=select_entries(entries,config.get('books',[]))
    if config.get('max_books') is not None:entries=entries[:config['max_books']]
    remaining=config.get('limit');selected=[]
    for entry in entries:
        candidates=sorted((p for p in descriptor['pages'] if p['book']==entry['path']),key=lambda p:p['page'])
        requested=set(parse_pages(config.get('pages',''),max(p['page'] for p in candidates)+1))
        candidates=[p for p in candidates if p['page'] in requested]
        if remaining is not None:
            candidates=candidates[:remaining];remaining-=len(candidates)
        selected.extend(candidates)
    if not selected:raise ValueError('No requested books/pages exist in the original OCR handoff')
    descriptor={k:v for k,v in descriptor.items() if k not in {'snapshot_id','restoration'}}
    descriptor['pages']=selected;descriptor['snapshot_id']=dataset_hash(descriptor)
    data=dataset_json_bytes(descriptor)
    dataset_write(destination,f"snapshots/{descriptor['snapshot_id']}.json",data)
    preserve_restoration=False
    if (destination/'dataset.json').is_file():
        _,active=dataset_open(destination)
        base={k:v for k,v in active.items() if k not in {'restoration','snapshot_id'}}
        preserve_restoration=bool(active.get('restoration')) and dataset_hash(base)==descriptor['snapshot_id']
    if not preserve_restoration:dataset_write(destination,'dataset.json',data)
    return str(destination/f"snapshots/{descriptor['snapshot_id']}.json")


def activate_original_ocr_handoff(restored, work_dir, data_dir, *, dataset=None):
    """Create separate editable run metadata; keep the verified imported copy intact."""
    manifest = restore_previous_ocr(Path(restored['directory'])/'ocr', work_dir, input_dir=data_dir)
    dataset = Path(dataset or restored['dataset'])
    root, descriptor = dataset_open(dataset)
    immutable = root/'snapshots'/f"{descriptor['snapshot_id']}.json"
    selection={}
    for page in descriptor['pages']:selection.setdefault(Path(page['book']).stem,set()).add(page['page'])
    seen = set()
    def visit(path):
        path=Path(path)
        if path in seen:return
        seen.add(path);data=read_checkpoint_campaign(path)
        for field in ('crop_manifests','variant_manifests'):
            for ref in data.get(field,{}).values():visit(ref)
        data['pairs']=[p for p in data.get('pairs',[]) if p['book'] in selection]
        for pair in data.get('pairs',[]):
            for engine in data['models']:
                directory=Path(pair[engine]);meta=directory/'run_manifest.json'
                if meta in seen:continue
                seen.add(meta);run=json.loads(meta.read_text(encoding='utf8'))
                run['config'].update(image_dataset=str(immutable),image_dataset_key=descriptor['snapshot_id'])
                # Source verification must follow this session's mounted PDFs.
                for book,info in run.get('books',{}).items():
                    old=info.get('src','');name=Path(old.replace('\\','/')).name
                    matches=list(Path(data_dir).rglob(name)) if name else []
                    if len(matches)==1:info['src']=str(matches[0].resolve())
                meta.write_bytes(dataset_json_bytes(run))
                task_file=directory/'tasks.jsonl'
                tasks=[json.loads(line) for line in task_file.read_text(encoding='utf8').splitlines() if line]
                tasks=[t for t in tasks if t['page'] in selection[pair['book']]]
                task_file.write_bytes(b''.join(dataset_json_bytes(t) for t in tasks))
        data['total_pages']=sum(len((Path(p[data['models'][0]])/'tasks.jsonl').read_text(encoding='utf8').splitlines()) for p in data['pairs'])
        data['total_books']=len({p['book'] for p in data['pairs']})
        path.write_bytes(dataset_json_bytes(data))
    visit(manifest)
    return manifest


def summarize_original_ocr_disagreement(evidence, *, crop_types=None, thresholds=None):
    """Descriptive threshold sweep only; never recommend or enable restoration."""
    thresholds = thresholds if thresholds is not None else [i/100 for i in range(5, 81, 5)]
    allowed=set(crop_types or ['page','group','column'])
    records=[s for s in evidence['scores'].values() if s.get('region',{}).get('crop_type') in allowed]
    by_crop={}
    for kind in sorted(allowed):
        entries=[s for s in records if s['region']['crop_type']==kind]
        states={}
        for score in entries:states[score['state']]=states.get(score['state'],0)+1
        rates=sorted(s['min_pairwise_rate'] for s in entries if s['state']=='ok')
        quantiles={str(p):rates[round((len(rates)-1)*p)] if rates else None for p in (.1,.25,.5,.75,.9,.95)}
        by_crop[kind]=dict(unique_source_regions=len(entries),states=states,min_pairwise_quantiles=quantiles)
    sweep=[]
    for kind in sorted(allowed):
        entries=[s for s in records if s['region']['crop_type']==kind and s['state']=='ok']
        for threshold in thresholds:
            chosen=[s for s in entries if s['min_pairwise_rate']>=threshold]
            sweep.append(dict(crop_type=kind,threshold=threshold,candidate_regions=len(chosen),
                source_pixels=sum(s['region']['size'][0]*s['region']['size'][1] for s in chosen),
                unique_source_pages=len({(s['region']['source_sha256'],s['region']['page']) for s in chosen})))
    return dict(schema='vtm-original-ocr-threshold-sweep-v1',by_crop=by_crop,threshold_sweep=sweep,
                selected_thresholds=None,restoration_started=False,
                interpretation='counts before contracts/budgets; all eligible original OCR pairs must pass; TongGu excluded for bien; no accuracy claim')


def export_original_ocr_thresholds(evidence, directory, *, crop_types=None):
    import csv
    from html import escape
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    summary=summarize_original_ocr_disagreement(evidence,crop_types=crop_types)
    path=directory/'original_ocr_thresholds.json';path.write_bytes(dataset_json_bytes(summary))
    csv_path=directory/'original_ocr_thresholds.csv'
    columns=['crop_type','threshold','candidate_regions','unique_source_pages','source_pixels']
    with csv_path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(summary['threshold_sweep'])
    html=('<!doctype html><meta charset="utf-8"><title>Ngưỡng bất đồng OCR gốc</title>'
          '<h1>Đếm vùng trước phục dựng</h1><p>Chưa chọn ngưỡng và chưa chạy restoration. '
          'Cả 6 cặp phải đạt ngưỡng; số vùng trước điều kiện đầu vào và quota từng model. '
          'Bất đồng không chứng minh ảnh hỏng hay độ chính xác.</p><table border="1"><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr>')
    for row in summary['threshold_sweep']:
        html+='<tr>'+''.join('<td>'+escape(str(row[c]))+'</td>' for c in columns)+'</tr>'
    html+='</table><h2>Độ đầy đủ OCR theo crop</h2><pre>'+escape(json.dumps(summary['by_crop'],ensure_ascii=False,indent=2))+'</pre>'
    html_path=directory/'original_ocr_thresholds.html';html_path.write_text(html,encoding='utf8')
    return dict(json=str(path),csv=str(csv_path),html=str(html_path),summary=summary)
