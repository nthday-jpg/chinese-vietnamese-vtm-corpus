"""Persist bounded failure evidence under Output, independently of snapshot publication."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def preserve_worker_diagnostics(worker,progress,destination,max_bytes=64*1024**2):
    worker=Path(worker).resolve();progress=Path(progress).resolve()
    destination=Path(destination)/worker.name;destination.mkdir(parents=True,exist_ok=True)
    candidates=[worker/name for name in ('worker.log','worker_config.json','ready.json')]
    if progress.is_relative_to(worker) and progress.is_dir():candidates.extend(sorted(progress.rglob('*')))
    copied=[];skipped=[];used=0
    for source in candidates:
        if not source.is_file() or not source.resolve().is_relative_to(worker):continue
        relative=source.relative_to(worker)
        if used+source.stat().st_size>max_bytes:
            skipped.append(dict(path=str(relative),reason='diagnostic byte budget exceeded'));continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);used+=target.stat().st_size;copied.append(str(relative))
    record=dict(format='vtm-worker-diagnostics-v1',files=copied,skipped=skipped,bytes=used,
                captured_after_worker_failure=True,intermediate_images_are_final=False)
    (destination/'diagnostic_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return str(destination)


def failure_bundle_path(root):
    root=Path(root).resolve();output=root.parent if root.name=='_dataset' else root
    return output.with_name(output.name+'_failure_diagnostics.zip')


def package_failure_diagnostics(root,summary):
    root=Path(root).resolve();archive=failure_bundle_path(root)
    def read(reference):
        path=(root/reference['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Diagnostic reference escapes dataset')
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=reference['sha256']:raise ValueError('Diagnostic reference checksum mismatch')
        return data
    report_bytes=read(summary['report']);report=json.loads(report_bytes)
    temporary=archive.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as output:
            output.writestr('last_attempt.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
            output.writestr('restoration_report.json',report_bytes)
            output.writestr('restoration_manifest.jsonl',read(report['manifest']))
            for name in ('preflight.json','last_plan.json'):
                source=root/name
                if source.is_file():output.write(source,name)
            folder=root/'worker_diagnostics'
            if folder.is_dir():
                for source in sorted(folder.rglob('*')):
                    if source.is_file() and source.resolve().is_relative_to(root):
                        output.write(source,source.relative_to(root).as_posix())
        temporary.replace(archive)
    finally:
        if temporary.exists():temporary.unlink()
    return str(archive)
