"""Local FLYVUE HTTP server. Imports run.run_image; browser automation is not used."""
import asyncio, os
from pathlib import Path
from aiohttp import web
from run import run_image
ROOT=Path(__file__).resolve().parent
PORT=int(os.environ.get('PORT','4660'))
async def index(request):
    return web.Response(text=(ROOT/'ui.html').read_text(),content_type='text/html')
async def test(request):
    reader=await request.multipart()
    field=await reader.next()
    if field is None: raise web.HTTPBadRequest(text='image required')
    data=await field.read()
    from PIL import Image
    import io
    image=Image.open(io.BytesIO(data)); image.load()
    async with asyncio.Semaphore(1):
        result=await asyncio.to_thread(run_image,image)
    return web.json_response({'mode':result.mode,'neuron_count':result.neuron_count,'unique_hex_count':result.unique_hex_count,'sim_wall_clock_ms':result.wall_clock_ms,'spikes_per_sec':result.spikes_per_sec,'first_seen':result.first_seen,'attribution':'Connectome © HHMI Janelia / FlyEM / Google Research, CC-BY'})
def make_app():
    app=web.Application(); app.router.add_get('/',index); app.router.add_post('/test',test); return app
if __name__=='__main__': web.run_app(make_app(),host='127.0.0.1',port=PORT)
