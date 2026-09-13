"""FLYVUE heatmap renderer. Imports no repo module; consumes fired IDs + soma coordinates."""
import io
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter
from matplotlib import colormaps
DEFAULT_SIGMA=7.0; DEFAULT_ALPHA=.58
ATTRIBUTION='Connectome © HHMI Janelia / FlyEM / Google Research, CC-BY'
def _xy(a):
    a=np.asarray(a,np.float32)
    if a.ndim!=2 or a.shape[1]!=2: raise ValueError('soma_xy must have shape (n,2)')
    return a
def fired_density(fired,soma_xy,width,height,sigma=DEFAULT_SIGMA):
    xy=_xy(soma_xy); ids=np.asarray(fired,np.int64).ravel(); ids=ids[(ids>=0)&(ids<len(xy))]
    ids=ids[np.isfinite(xy[ids]).all(axis=1)]; g=np.zeros((height,width),np.float32)
    if len(ids):
        px=np.clip(np.rint(xy[ids,0]*(width-1)).astype(np.int32),0,width-1); py=np.clip(np.rint(xy[ids,1]*(height-1)).astype(np.int32),0,height-1); np.add.at(g,(py,px),1)
    if sigma>0:g=gaussian_filter(g,float(sigma),mode='nearest')
    m=float(g.max()); return g/m if m else g
def density_rgba(density,alpha=DEFAULT_ALPHA):
    d=np.clip(np.asarray(density,np.float32),0,1); rgba=(colormaps['inferno'](d)*255).astype(np.uint8); rgba[...,3]=np.clip(d*alpha*255,0,255).astype(np.uint8); return rgba
def overlay_heatmap(image,density,alpha=DEFAULT_ALPHA):
    base=image.convert('RGBA'); ov=Image.fromarray(density_rgba(density,alpha),'RGBA').resize(base.size,Image.Resampling.BILINEAR); return Image.alpha_composite(base,ov)
def encode_png(image):
    b=io.BytesIO(); image.save(b,'PNG',optimize=True); return b.getvalue()
def make_heatmap_png(image,fired,soma_xy,sigma=DEFAULT_SIGMA,alpha=DEFAULT_ALPHA):
    out=overlay_heatmap(image.convert('RGB'),fired_density(fired,soma_xy,image.width,image.height,sigma),alpha); d=ImageDraw.Draw(out); y=out.height-29; d.rectangle((8,y,min(out.width-8,520),out.height-7),fill=(0,0,0,185)); d.text((14,y+4),ATTRIBUTION,fill='white'); return encode_png(out)
