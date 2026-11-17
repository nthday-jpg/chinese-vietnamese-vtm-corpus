"""Cache expensive original text comparisons by exact input and algorithm hashes."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import uuid

from .multi_compare import compare_multi_corpus
from .ocr_records import read_checkpoint_campaign


def comparison_file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def comparison_source_signature():
    # The standalone notebook receives the same signature from its builder.
    override=globals().get('COMPARISON_CACHE_CODE_SHA256')
    if override:return override
    names=('comparison_cache.py','multi_compare.py','text_alignment.py','multi_consensus.py',
           'ocr_records.py','reading_order.py','ocr_quality.py','character_regions.py','page_consensus.py')
    return hashlib.sha256(''.join(comparison_file_sha(Path(__file__).with_name(n)) for n in names).encode()).hexdigest()


def cached_original_comparison(manifest_path, cache_dir, **options):
    """Reuse alignment only; image evidence and current manifest metadata stay fresh."""
    path=Path(manifest_path);campaign=read_checkpoint_campaign(path)
    files,pdfs,identities={}, {}, []
    for pair in campaign['pairs']:
        for engine in campaign['models']:
            directory=Path(pair[engine]);metadata=json.loads((directory/'run_manifest.json').read_text(encoding='utf8'))
            identities.append(dict(book=pair['book'],engine=engine,config=metadata['config'],books=metadata['books'],
                                   kind=metadata.get('kind'),allow_incomplete_tail=metadata.get('status')=='running'))
            # Status timestamps/progress are not input to text comparison.
            for source in metadata['books'].values():
                filename=str(source.get('src','')).replace('\\','/').rsplit('/',1)[-1]
                candidates=list(Path(options['source_dir']).rglob(filename)) if options.get('source_dir') and filename else []
                if len(candidates)>1:raise ValueError('Ambiguous relocated source PDF: '+filename)
                pdf=Path(candidates[0] if candidates else source.get('src',''))
                name=str(pdf.resolve())
                if name not in pdfs:pdfs[name]=comparison_file_sha(pdf) if pdf.is_file() else None
            selected=[directory/'tasks.jsonl']
            run_name=metadata['config'].get('run_name') or engine
            selected.extend((directory/'ocr_raw').glob('*/'+run_name+'_shard*.jsonl'))
            for item in selected:
                name=str(item.resolve())
                if name not in files:files[name]=comparison_file_sha(item) if item.is_file() else None
    identity=dict(schema=1,algorithm=comparison_source_signature(),models=campaign['models'],
                  runs=identities,files=files,pdfs=pdfs,options=options)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()
    target=Path(cache_dir)/(key+'.json.gz')
    try:
        with gzip.open(target,'rt',encoding='utf8') as stream:saved=json.load(stream)
        if saved['key']!=key:raise ValueError('Comparison cache identity mismatch')
        result=saved['result'];hit=True
    except (OSError,ValueError,KeyError,EOFError):
        result=compare_multi_corpus(path,**options);hit=False
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with gzip.open(temporary,'wt',encoding='utf8',compresslevel=1) as stream:
                json.dump(dict(key=key,result=result),stream,ensure_ascii=False,separators=(',',':'))
            temporary.replace(target)
        finally:
            if temporary.exists():temporary.unlink()
    # Return current run status/provenance even when alignment came from cache.
    result['manifests']={pair['book']:{engine:json.loads((Path(pair[engine])/'run_manifest.json').read_text(encoding='utf8'))
                        for engine in campaign['models']} for pair in campaign['pairs']}
    result['corpus_manifest']=str(path.resolve())
    result['comparison_cache']=dict(hit=hit,key=key)
    return result
