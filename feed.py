"""FLYVUE retina feed. Imports repo FlyBrain/FlyEye and preserves FlyEye.look sampling."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
from flysim import FlyBrain
from flyeye import FlyEye
FOV_W,FOV_H,MAX_HZ=300,210,180.0
L2_MAX_HZ=MAX_HZ*0.6
@dataclass(frozen=True)
class FeedResult:
    column_values:np.ndarray; column_hex:np.ndarray; column_uv:np.ndarray; drive:dict; unique_hex_count:int; fixation:tuple[int,int]
def pil_to_luminance(image):
    if not isinstance(image,Image.Image): raise TypeError('image must be PIL.Image')
    return np.asarray(image.convert('L'),dtype=np.float32)/255.0
def fit_to_fov(image,width=FOV_W,height=FOV_H):
    if width<=0 or height<=0: raise ValueError('invalid FOV')
    src=image.convert('RGB'); scale=min(width/src.width,height/src.height)
    nw=max(1,round(src.width*scale)); nh=max(1,round(src.height*scale))
    src=src.resize((nw,nh),Image.Resampling.LANCZOS); out=Image.new('RGB',(width,height),(0,0,0))
    out.paste(src,((width-nw)//2,(height-nh)//2)); return out
def _lookup(path):
    import pandas as pd
    a=pd.read_feather(path,columns=['bodyId','assignedOlHex1','assignedOlHex2']).drop_duplicates('bodyId'); r={}
    for x in a.itertuples(index=False):
        try:
            h1,h2=float(x.assignedOlHex1),float(x.assignedOlHex2)
            if np.isfinite(h1) and np.isfinite(h2): r[int(x.bodyId)]=(int(h1),int(h2))
        except (TypeError,ValueError): pass
    return r
def _hexes(fb,eye,path):
    r=_lookup(path); on=[r[int(fb.bodies[i])] for i in eye.on_idx]; off=[r[int(fb.bodies[i])] for i in eye.off_idx]
    if len(on)!=len(eye.on_idx) or len(off)!=len(eye.off_idx): raise RuntimeError('missing optic-lobe hex annotation')
    return np.asarray(on,np.int32),np.asarray(off,np.int32)
def _aggregate(fb,eye,drive,path):
    on,off=_hexes(fb,eye,path); vals=list(drive.values())
    if len(vals)!=2: raise RuntimeError('FlyEye.look must return ON/OFF drives')
    a,b=np.asarray(vals[0],np.float32),np.asarray(vals[1],np.float32); sums={}; counts={}
    for k,v in zip(map(tuple,on),np.clip(a/MAX_HZ,0,1)): sums[k]=sums.get(k,0)+float(v); counts[k]=counts.get(k,0)+1
    for k,v in zip(map(tuple,off),np.clip(1-b/L2_MAX_HZ,0,1)): sums[k]=sums.get(k,0)+float(v); counts[k]=counts.get(k,0)+1
    keys=sorted(sums); return np.asarray([sums[k]/counts[k] for k in keys],np.float32),np.asarray(keys,np.int32)
def sample(image,fb,eye,cx,cy,annotations_path='data/body-annotations.feather'):
    drive=eye.look(pil_to_luminance(image),int(cx),int(cy),fov_w=FOV_W,fov_h=FOV_H,max_hz=MAX_HZ)
    v,h=_aggregate(fb,eye,drive,Path(annotations_path)); u,w=eye._to_uv(h[:,0],h[:,1])
    return FeedResult(v,h,np.column_stack((u,w)).astype(np.float32),drive,len(h),(int(cx),int(cy)))
def build_feed(image,fb,eye=None,fixation=None,annotations_path='data/body-annotations.feather'):
    eye=eye or FlyEye(fb,str(annotations_path)); image=fit_to_fov(image); cx,cy=fixation or (150,105)
    return sample(image,fb,eye,cx,cy,annotations_path)
