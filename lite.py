"""FLYVUE lightweight graph builder. Imports graph metadata from flysim's graph format."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from flysim import FlyBrain,Params
from flyeye import FlyEye,FlyPilot

def load_lite(graph_path,annotations_path,depth=4):
    z=np.load(graph_path,allow_pickle=False)
    W=sp.csr_matrix((z['data'].astype(np.float32),z['indices'],z['indptr']),shape=tuple(z['shape']))
    bodies=z['bodies']; types=z['types'].astype(str)
    seed=np.flatnonzero((types=='L1')|(types=='L2')); chosen=set(seed.tolist()); frontier=seed
    for _ in range(depth):
        if len(frontier)==0:break
        target=np.unique(W[:,frontier].nonzero()[0]); new=np.asarray([i for i in target if int(i) not in chosen],dtype=np.int64); chosen.update(map(int,new)); frontier=new
    sel=np.asarray(sorted(chosen),dtype=np.int64); sub=W[sel][:,sel].tocsc()
    fb=FlyBrain.__new__(FlyBrain); fb.W=sub; fb.indptr=sub.indptr; fb.indices=sub.indices; fb.wdata=sub.data.astype(np.float32); fb.n=len(sel); fb.bodies=bodies[sel]; fb.types=types[sel]
    for key in ('superclass','subclass','receptor','fru','nt'):setattr(fb,key,z[key].astype(str)[sel])
    fb.p=Params(); fb.type_names,fb.type_code=np.unique(fb.types,return_inverse=True); fb.n_types=len(fb.type_names); fb.body_to_i={int(b):i for i,b in enumerate(fb.bodies)}; fb.decay=np.float32(np.exp(-fb.p.dt/fb.p.tau_m)); fb.refr_steps=int(np.ceil(fb.p.refractory/fb.p.dt))
    eye=FlyEye(fb,str(annotations_path)); pilot=FlyPilot(fb,eye=eye,sim_steps=125); return fb,eye,pilot
