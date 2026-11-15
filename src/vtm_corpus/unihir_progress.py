"""Observe the unchanged UniHIR pipeline without changing predictions or RNG."""
from __future__ import annotations
import functools
import json
from pathlib import Path
import threading
import time


class UniHIRProgress:
    def __init__(self,net,directory,interval=30,emit=print,input_size=None):
        self.net=net;self.directory=Path(directory);self.interval=interval;self.emit=emit
        self.lock=threading.RLock();self.stop=threading.Event();self.stack=[]
        self.replacements=[];self.hooks=[];self.sequence=0;self.forwards=0;self.tokens=0
        self.artifacts=[];self.started=time.monotonic();self.status='running';self.error_stage=None
        self.input_size=input_size;self.generation_sizes=[]

    def snapshot(self):
        with self.lock:
            return dict(format='vtm-unihir-progress-v1',status=self.status,
                stage=' > '.join(self.stack) or 'between_stages',error_stage=self.error_stage,
                activity=self.sequence,model_forwards=self.forwards,text_tokens=self.tokens,
                seconds=round(time.monotonic()-self.started,3),artifacts=list(self.artifacts),
                input_size=self.input_size,generation_sizes=list(self.generation_sizes),
                observed_processor_sizes=dict(getattr(self.net,'_observed_processor_sizes',{})),
                intermediate_images_are_final=False)

    def publish(self,announce=False):
        with self.lock:
            data=self.snapshot();self.directory.mkdir(parents=True,exist_ok=True)
            target=self.directory/'progress.json';temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            for attempt in range(20):
                try:temporary.replace(target);break
                except PermissionError:
                    if attempt==19:raise
                    time.sleep(.005) # Windows readers may briefly hold the destination open.
        if announce:
            self.emit('[UniHIR progress] '+json.dumps(data,ensure_ascii=False),flush=True)

    def pulse(self,kind):
        # No tensor reads, synchronization, sampling or CUDA allocations here.
        with self.lock:
            self.sequence+=1
            if kind=='forward':self.forwards+=1
            else:self.tokens+=1

    def wrap(self,owner,name,label):
        original=getattr(owner,name,None)
        if not callable(original):return
        owned=name in getattr(owner,'__dict__',{})
        @functools.wraps(original)
        def observed(*args,**kwargs):
            with self.lock:
                self.stack.append(label);self.sequence+=1
                if label=='gen_image' and args and isinstance(args[0],(list,tuple)) and len(args[0])==2:
                    self.generation_sizes.append(list(args[0]))
            self.publish(True)
            try:
                result=original(*args,**kwargs)
                # Preserve completed intermediate candidates even if a later stage times out.
                if label in ('gen_image','gen_text'):
                    self.capture(label,result)
                return result
            except BaseException:
                with self.lock:
                    if self.error_stage is None:self.error_stage=' > '.join(self.stack)
                raise
            finally:
                with self.lock:self.stack.pop();self.sequence+=1
                self.publish(True)
        setattr(owner,name,observed);self.replacements.append((owner,name,original,owned))

    def capture(self,label,result):
        folder=self.directory/'intermediate_not_final';folder.mkdir(exist_ok=True)
        number=len(self.artifacts)+1
        if label=='gen_text' and isinstance(result,str):
            target=folder/f'{number:03d}_text.txt';target.write_text(result,encoding='utf8')
        elif label=='gen_image' and hasattr(result,'save'):
            target=folder/f'{number:03d}_image.png';result.save(target,format='PNG')
        else:return
        with self.lock:self.artifacts.append(str(target.relative_to(self.directory)))

    def heartbeat(self):
        while not self.stop.wait(self.interval):self.publish(True)

    def __enter__(self):
        try:
            for name in ('update_context_text','update_context_image','update_context_image_gen','gen_image','gen_text','decode_image'):
                self.wrap(self.net,name,name)
            vae=getattr(self.net,'vae_model',None)
            if vae is not None:
                for name in ('encode','decode'):self.wrap(vae,name,'vae.'+name)
            model=getattr(self.net,'model',None)
            language=getattr(model,'language_model',None)
            decoder=getattr(language,'model',None)
            layers=getattr(decoder,'layers',None)
            if layers is not None and len(layers):
                self.hooks.append(layers[-1].register_forward_hook(lambda *a:self.pulse('forward')))
            head=getattr(language,'lm_head',None)
            if head is not None:self.hooks.append(head.register_forward_hook(lambda *a:self.pulse('token')))
            self.publish(True)
            self.thread=threading.Thread(target=self.heartbeat,daemon=True);self.thread.start()
            return self
        except BaseException:
            self.__exit__(*__import__('sys').exc_info());raise

    def __exit__(self,kind,value,traceback):
        self.stop.set()
        if hasattr(self,'thread'):self.thread.join(timeout=2)
        for hook in self.hooks:hook.remove()
        for owner,name,original,owned in reversed(self.replacements):
            if owned:setattr(owner,name,original)
            else:delattr(owner,name)
        if kind is None:
            # The successful result already exports upstream draft/refinement/text extras.
            # Keep temporary candidate files only when the job fails or is killed.
            for relative in self.artifacts:
                target=(self.directory/relative).resolve()
                if target.is_relative_to(self.directory.resolve()) and target.is_file():target.unlink()
            folder=self.directory/'intermediate_not_final'
            if folder.is_dir():folder.rmdir()
            self.artifacts.clear()
        with self.lock:self.status='failed' if kind else 'complete'
        self.publish(True)
        return False
