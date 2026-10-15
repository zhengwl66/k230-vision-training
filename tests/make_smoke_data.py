"""Generated fixtures only: verifies commands, never measures real accuracy."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    names=json.loads((REPO/'configs/classes.json').read_text(encoding='utf-8'))
    for cid,name in enumerate(names):
        folder=args.output/name;folder.mkdir(parents=True)
        for i in range(20):
            image=Image.new('RGB',(320,240),(220+i,230,230));d=ImageDraw.Draw(image);x=50+i;d.rectangle((x,55,x+100,185),fill=(20+cid*3,45,180-i*2))
            path=folder/(str(i).zfill(4)+'.jpg');image.save(path)
            shape=dict(label=name,shape_type='rectangle',points=[[x,55],[x+100,185]],description=('UPRIGHT' if i%2==0 else 'LYING') if cid in (2,3,5) else '')
            path.with_suffix('.json').write_text(json.dumps(dict(imageWidth=320,imageHeight=240,shapes=[shape])),encoding='utf-8')
    print('GENERATED_SMOKE_FIXTURES',args.output)
if __name__=='__main__':main()
