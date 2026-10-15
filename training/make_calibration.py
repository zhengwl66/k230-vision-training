import argparse,json,random,shutil
from pathlib import Path
from PIL import Image
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--kind',choices=('shape','pose'),default='shape');p.add_argument('--count',type=int,default=112);args=p.parse_args();assert not args.output.exists()
    if args.kind=='shape':rows=[r for r in json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8')) if r['split']=='train']
    else:rows=json.loads((args.dataset/'pose_manifest.json').read_text(encoding='utf-8'))['train']
    random.Random(2026101506).shuffle(rows);rows=rows[:args.count];assert len(rows)>=32
    args.output.mkdir(parents=True)
    for i,r in enumerate(rows):
        source=args.dataset/r['image']
        if args.kind=='shape':shutil.copy2(source,args.output/(str(i).zfill(4)+source.suffix.lower()))
        else:Image.open(source).convert('RGB').crop(tuple(r['box'])).resize((96,96),Image.Resampling.BILINEAR).save(args.output/(str(i).zfill(4)+'.png'))
    (args.output/'calibration_manifest.json').write_text(json.dumps(dict(split='train',kind=args.kind,rows=rows),ensure_ascii=False,indent=2),encoding='utf-8');print('CALIBRATION_READY',len(rows))
if __name__=='__main__':main()
