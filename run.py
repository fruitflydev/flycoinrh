"""FLYVUE runner. Imports repo FlyBrain, FlyEye and FlyPilot; no browser automation."""
import os,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from flysim import FlyBrain
from flyeye import FlyEye,FlyPilot
from feed import build_feed,fit_to_fov
from heatmap import make_heatmap_png
ROOT=Path(__file__).resolve().parent; GRAPH=ROOT/'build'/'graph.npz'; ANNOTATIONS=ROOT/'data'/'body-annotations.feather'
STEPS,DT_MS=125,.2; FOV_W,FOV_H=300,210; MEMORY_CEILING=2.5*1024**3
@dataclass
class Simulation:
    heatmap_png:bytes; spike_log:np.ndarray; fired:np.ndarray; soma_xy:np.ndarray; first_seen:dict; spikes_per_sec:float; wall_clock_ms:float; mode:str; neuron_count:int; unique_hex_count:int; fixation:tuple[int,int]
_STATE={'fb':None,'eye':None,'pilot':None,'soma_xy':None,'mode':None}
def _soma_xy(fb):
    import pandas as pd
    a=pd.read_feather(ANNOTATIONS,columns=['bodyId','somaLocation']).drop_duplicates('bodyId').set_index('bodyId'); loc=a['somaLocation'].reindex(fb.bodies).to_numpy(); xy=np.full((fb.n,2),np.nan,np.float32)
    for i,v in enumerate(loc):
        if isinstance(v,(list,tuple,np.ndarray)) and len(v)>=3:
            try:xy[i]=(float(v[0]),float(v[2]))
            except (TypeError,ValueError):pass
    ok=np.isfinite(xy).all(1)
    if ok.sum()<100:raise RuntimeError('insufficient soma coordinates')
    lo,hi=np.nanmin(xy[ok],0),np.nanmax(xy[ok],0); xy[ok]=(xy[ok]-lo)/np.maximum(hi-lo,1e-6); return xy
def _load():
    if _STATE['fb'] is None:
        fb=FlyBrain(GRAPH); eye=FlyEye(fb,str(ANNOTATIONS)); _STATE.update(fb=fb,eye=eye,pilot=FlyPilot(fb,eye=eye,sim_steps=STEPS),soma_xy=_soma_xy(fb),mode='full')
    return _STATE['fb'],_STATE['eye']
def _dense_log(raw,steps,n):
    log=np.zeros((steps,n),np.bool_)
    for step,fired in enumerate(raw[:steps]):
        if len(fired):log[step,np.asarray(fired,np.int64)]=True
    return log
def run_fixation(img,cx,cy,seed=0):
    fb,eye=_load(); image=img if isinstance(img,Image.Image) else Image.fromarray(np.asarray(img)); feed=build_feed(image,fb,eye=eye,fixation=(cx,cy),annotations_path=ANNOTATIONS); t=time.perf_counter(); r=fb.run(feed.drive,steps=STEPS,seed=seed,spike_log=True); wall=(time.perf_counter()-t)*1000; log=_dense_log(r['_spikes'],STEPS,fb.n)
    from order import first_seen
    heat=make_heatmap_png(image,r['_fired'],_STATE['soma_xy']); return Simulation(heat,log,np.asarray(r['_fired'],np.int32),_STATE['soma_xy'],first_seen(log,DT_MS),float(r['_spikes_per_sec']),wall,_STATE['mode'],int(fb.n),feed.unique_hex_count,(int(cx),int(cy)))
def run_image(image,seed=0):return run_fixation(fit_to_fov(image),150,105,seed)
def health():
    fb,eye=_load(); return {'mode':_STATE['mode'],'neuron_count':int(fb.n),'unique_hex_count':int(len(set(zip(eye.on_h1.tolist(),eye.on_h2.tolist()))))}
