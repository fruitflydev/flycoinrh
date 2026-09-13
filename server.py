"""Local FLYVUE server. Imports run.run_image and fly_music; no browser automation."""
import asyncio,base64,io,os,re
from pathlib import Path
from aiohttp import web
from PIL import Image,UnidentifiedImageError
from run import run_image
from fly_music import compose, manifest, lyrics_text, voice_studio_synthesize
ROOT=Path(__file__).resolve().parent; PORT=int(os.environ.get('PORT','4660')); LIMIT=10*1024*1024
ATTRIBUTION='Connectome © HHMI Janelia / FlyEM / Google Research, CC-BY'; SEM=asyncio.Semaphore(1); SAFE=re.compile(r'[^A-Za-z0-9._-]+')
def safe_name(name): return SAFE.sub('_',Path(name or 'upload').name)[:120] or 'upload'
async def index(request): return web.Response(text=(ROOT/'ui.html').read_text(encoding='utf-8'),content_type='text/html')
async def _read_image(request):
    reader=await request.multipart(); field=await reader.next()
    if field is None: raise web.HTTPBadRequest(text='multipart image required')
    _=safe_name(field.filename); data=await field.read()
    if len(data)>LIMIT: raise web.HTTPRequestEntityTooLarge(max_size=LIMIT,actual_size=len(data))
    try:
        image=Image.open(io.BytesIO(data)); image.load()
        if image.format not in {'JPEG','PNG','WEBP'}: raise ValueError('JPEG/PNG/WebP only')
        return image
    except (UnidentifiedImageError,OSError,ValueError) as e: raise web.HTTPBadRequest(text=str(e))
async def test(request):
    image=await _read_image(request)
    async with SEM: result=await asyncio.to_thread(run_image,image)
    return web.json_response({'heatmap_png':'data:image/png;base64,'+base64.b64encode(result.heatmap_png).decode(),'first_seen':result.first_seen,'spikes_per_sec':result.spikes_per_sec,'mode':result.mode,'neuron_count':result.neuron_count,'unique_hex_count':result.unique_hex_count,'sim_wall_clock_ms':result.wall_clock_ms,'soma_xy':result.soma_xy.tolist(),'fired':result.fired.tolist(),'attribution':ATTRIBUTION})
async def sing(request):
    image=await _read_image(request)
    variation=int(request.query.get('variation','0') or 0)
    use_voice=request.query.get('voice','1') != '0'
    async with SEM: result=await asyncio.to_thread(run_image,image,variation)
    song=compose(result.spike_log,result.soma_xy,variation=variation)
    song=manifest(song,result.mode,result.neuron_count,result.unique_hex_count)
    audio_b64=None; audio_error=None
    if use_voice:
        try:
            audio=await asyncio.to_thread(voice_studio_synthesize,song)
            audio_b64='data:audio/wav;base64,'+base64.b64encode(audio).decode()
        except Exception as exc:
            audio_error=f'VoiceStudio unavailable: {type(exc).__name__}: {exc}'
    return web.json_response({'song':song,'lyrics':lyrics_text(song),'audio_wav':audio_b64,'voice_error':audio_error,'neural':{'first_seen':result.first_seen,'spikes_per_sec':result.spikes_per_sec,'sim_wall_clock_ms':result.wall_clock_ms},'attribution':ATTRIBUTION})
def make_app():
    app=web.Application(client_max_size=LIMIT+1024*1024); app.router.add_get('/',index); app.router.add_post('/test',test); app.router.add_post('/sing',sing); return app
if __name__=='__main__': web.run_app(make_app(),host='127.0.0.1',port=PORT)
