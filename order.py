"""FLYVUE first-seen ranking. Imports flysim.Params for the repo dt=0.2 ms."""
import numpy as np
from flysim import Params
def first_seen(spike_log,dt_ms=Params.dt):
    log=np.asarray(spike_log,bool)
    if log.ndim!=2: raise ValueError('spike_log must have shape (steps,neurons)')
    steps,_=log.shape; out={}
    for ms in (1,5,10,25):
        k=min(steps,max(1,int(np.ceil(ms/dt_ms)))); seen=np.flatnonzero(log[:k].any(0))
        if len(seen):
            t=np.argmax(log[:k,seen],0).astype(np.float32)*dt_ms; ranked=seen[np.argsort(t,kind='stable')]
        else: ranked=np.empty(0,np.int32)
        out[f'{ms}ms']={'count':int(len(seen)),'neurons':ranked.astype(int).tolist()}
    return out
def rank_vector(spike_log,dt_ms=Params.dt):
    log=np.asarray(spike_log,bool)
    if log.ndim!=2: raise ValueError('spike_log must be 2-D')
    first=np.full(log.shape[1],np.inf,np.float32)
    for step in range(log.shape[0]): first[log[step]&np.isinf(first)]=step*dt_ms
    return first
