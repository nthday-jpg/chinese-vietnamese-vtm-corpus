"""Lossless contact sheets for readable image exports; never an OCR input."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import uuid
import zipfile

from .image_dataset import dataset_hash, dataset_file_hash, dataset_json_bytes, dataset_path, dataset_write
from .restoration_storage import PreparationStorageLimit


COMPARISON_EXPORT_VERSION = 1


@lru_cache(maxsize=1)
def comparison_fonts():
    from PIL import ImageFont
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 'C:/Windows/Fonts/arial.ttf'):
        try:return ImageFont.truetype(name,16),ImageFont.truetype(name,20)
        except OSError:pass
    return ImageFont.load_default(),ImageFont.load_default()


def render_comparison_sheet(filename, methods, panels, *, columns, panel_width, panel_height):
    from PIL import Image,ImageDraw,ImageOps
    font,title_font=comparison_fonts()
    columns=min(columns,len(methods));rows=(len(methods)+columns-1)//columns
    padding,header,label,footer=12,44,52,32
    cell_width,cell_height=panel_width+padding*2,panel_height+label+padding*2
    sheet=Image.new('RGB',(columns*cell_width,header+rows*cell_height+footer),'#eeeeee')
    draw=ImageDraw.Draw(sheet)
    draw.text((padding,10),filename,fill='#202020',font=title_font)
    try:
        for index,method in enumerate(methods):
            left,top=(index%columns)*cell_width,header+(index//columns)*cell_height
            draw.rectangle((left+3,top+3,left+cell_width-4,top+cell_height-4),fill='white',outline='#bbbbbb')
            draw.text((left+padding,top+8),method,fill='#202020',font=font)
            entry=panels.get(method)
            if entry is None:
                draw.text((left+padding,top+34),'No output - see dataset manifest',fill='#a03030',font=font)
                continue
            data=Path(entry['source']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError(f"Image checksum mismatch: {entry['source']}")
            with Image.open(io.BytesIO(data)) as loaded:
                image=loaded.convert('RGB')
            try:
                suffix=' / diagnostic mask' if method=='tadisr_text_mask' else ' / reference' if method=='original' else ''
                draw.text((left+padding,top+30),f'{image.width} x {image.height}'+suffix,fill='#555555',font=font)
                # Apply the same display interpolation to every method; source files stay intact.
                preview=ImageOps.contain(image,(panel_width,panel_height),Image.Resampling.LANCZOS)
                try:
                    sheet.paste(preview,(left+padding+(panel_width-preview.width)//2,
                                        top+label+padding+(panel_height-preview.height)//2))
                finally:preview.close()
            finally:image.close()
        draw.text((padding,sheet.height-footer+6),'Comparison preview only - original PNGs remain in the method folders',
                  fill='#555555',font=font)
        return sheet
    except BaseException:
        sheet.close()
        raise


def export_comparison_images(entries, destination, *, methods=None, columns=4, panel_width=384, panel_height=512,
                             reserve_bytes=2*1024**3):
    """One sheet per identical book/page/crop filename, with original first."""
    if any(type(v) is not int or v<=0 for v in (columns,panel_width,panel_height)):
        raise ValueError('Comparison columns/panel dimensions must be positive integers')
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    groups={};observed=[];spellings={}
    for entry in entries:
        name=PurePosixPath(str(entry['name']).replace('\\','/'))
        if len(name.parts)!=2 or name.suffix.lower()!='.png' or '..' in name.parts:
            raise ValueError('Comparison inputs require method/filename.png names')
        method,filename=name.parts
        prior_name=spellings.setdefault(filename.casefold(),filename)
        if prior_name!=filename:raise ValueError('Conflicting comparison filenames: '+filename)
        if method in groups.setdefault(filename,{}):raise ValueError('Duplicate comparison panel: '+entry['name'])
        groups[filename][method]=entry
        if method not in observed:observed.append(method)
    ordered=list(dict.fromkeys(['original',*(methods or []),*observed]))
    if not groups:raise ValueError('No comparison images to export')
    missing_original=[name for name,panels in groups.items() if 'original' not in panels]
    if missing_original:raise ValueError('Comparison requires original for each crop: '+missing_original[0])
    settings=dict(version=COMPARISON_EXPORT_VERSION,methods=ordered,columns=columns,
                  panel_width=panel_width,panel_height=panel_height)
    receipt_root=destination/'_dataset'
    receipt='exports/comparisons_'+dataset_hash(str(destination))+'.json'
    try:previous=json.loads(dataset_path(receipt_root,receipt).read_text(encoding='utf8'))
    except FileNotFoundError:previous={}
    owned=previous.get('files',{})
    for name,record in owned.items():
        if len(PurePosixPath(name).parts)!=2 or PurePosixPath(name).parts[0]!='comparisons' or not name.endswith('.png'):
            raise ValueError('Invalid comparison export receipt')
        dataset_path(destination,name)
    plans=[]
    for filename,panels in sorted(groups.items()):
        relative='comparisons/'+filename;target=dataset_path(destination,relative)
        identity=dataset_hash({'settings':settings,'images':{n:e['sha256'] for n,e in sorted(panels.items())}})
        old=owned.get(relative,{})
        digest=dataset_file_hash(target) if target.is_file() else None
        if target.exists() and (not target.is_file() or digest!=old.get('sha256')):
            raise FileExistsError('Unmanaged or edited comparison image: '+str(target))
        plans.append((filename,panels,relative,target,identity,old,digest))
    current={};generated=reused=0;exported=[]
    try:
        for filename,panels,relative,target,identity,old,digest in plans:
            if digest and old.get('identity')==identity:
                generated_digest=digest;reused+=1
            else:
                sheet=render_comparison_sheet(filename,ordered,panels,columns=columns,
                                              panel_width=panel_width,panel_height=panel_height)
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    stream=io.BytesIO()
                    sheet.save(stream,format='PNG',compress_level=1)
                    data=stream.getvalue()
                    if shutil.disk_usage(destination).free < len(data)+reserve_bytes:
                        raise PreparationStorageLimit('Storage reserve reached while exporting comparisons')
                    temporary.write_bytes(data)
                    temporary.replace(target)
                finally:
                    sheet.close()
                    if temporary.exists():temporary.unlink()
                generated_digest=dataset_file_hash(target);generated+=1
            current[relative]={'identity':identity,'sha256':generated_digest}
            exported.append({'name':relative,'source':str(target),'sha256':generated_digest})
    except PreparationStorageLimit:
        # Retain ownership of finished previews so a later export can reuse
        # them; leave all old sheets in place until the selection completes.
        dataset_write(receipt_root,receipt,dataset_json_bytes(
            {'settings':settings,'files':{**owned,**current},'incomplete_storage':True}))
        raise
    # Retire only unchanged sheets created by this exporter on a previous selection.
    for name,record in owned.items():
        if name not in current:
            target=dataset_path(destination,name)
            if target.is_file() and dataset_file_hash(target)==record['sha256']:target.unlink()
    dataset_write(receipt_root,receipt,dataset_json_bytes({'settings':settings,'files':current}))
    return {'directory':str(destination/'comparisons'),'files':len(current),'generated':generated,
            'reused':reused,'methods':ordered,'entries':exported}


def package_image_export_entries(entries, archive):
    """Package method PNGs and comparison PNGs without recompressing their pixels."""
    archive=Path(archive).absolute()
    if archive.suffix.lower()!='.zip':raise ValueError('Image archive must have a .zip extension')
    archive.parent.mkdir(parents=True,exist_ok=True)
    required=sum(Path(entry['source']).stat().st_size for entry in entries)+len(entries)*512
    if shutil.disk_usage(archive.parent).free < required+2*1024**3:
        raise PreparationStorageLimit('Storage reserve reached before writing the image ZIP')
    temporary=archive.with_name('.'+archive.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        names=set()
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as output:
            for entry in sorted(entries,key=lambda e:e['name']):
                name=PurePosixPath(entry['name'])
                if name.is_absolute() or '..' in name.parts or len(name.parts)!=2 or name.suffix.lower()!='.png':
                    raise ValueError('Unsafe image archive path: '+str(name))
                if str(name).casefold() in names:raise ValueError('Duplicate image archive path: '+str(name))
                names.add(str(name).casefold())
                info=zipfile.ZipInfo(str(name),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
                hasher=hashlib.sha256()
                with Path(entry['source']).open('rb') as source,output.open(info,'w',force_zip64=True) as target:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        hasher.update(block);target.write(block)
                if hasher.hexdigest()!=entry['sha256']:raise ValueError('Image checksum mismatch: '+str(entry['source']))
        temporary.replace(archive)
    finally:
        if temporary.exists():temporary.unlink()
    return {'archive':str(archive),'files':len(entries),'bytes':archive.stat().st_size}
