"""Batch FLYVUE demo. Imports run.run_image from the repository."""
import argparse,json
from pathlib import Path
from PIL import Image
from run import run_image
EXT={'.jpg','.jpeg','.png','.webp'}
def main():
    p=argparse.ArgumentParser(); p.add_argument('folder'); p.add_argument('--out',default='build/demo_out'); a=p.parse_args(); src,out=Path(a.folder),Path(a.out); out.mkdir(parents=True,exist_ok=True)
    for f in sorted(x for x in src.iterdir() if x.suffix.lower() in EXT):
        s=run_image(Image.open(f)); (out/(f.stem+'.png')).write_bytes(s.heatmap_png); (out/(f.stem+'.json')).write_text(json.dumps({'mode':s.mode,'neuron_count':s.neuron_count,'unique_hex_count':s.unique_hex_count,'sim_wall_clock_ms':s.wall_clock_ms,'spikes_per_sec':s.spikes_per_sec,'first_seen':s.first_seen,'attribution':'Connectome © HHMI Janelia / FlyEM / Google Research, CC-BY'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
