"""Import reviewed LabelMe rectangles, preserve historical splits and deduplicate."""
import argparse,hashlib,json,math,re,shutil
from pathlib import Path
from PIL import Image
REPO=Path(__file__).resolve().parents[1]
NAMES=json.loads((REPO/'configs/classes.json').read_text(encoding='utf-8'))
ZH=json.loads((REPO/'configs/classes_zh.json').read_text(encoding='utf-8'))
ALIASES={n:i for i,n in enumerate(NAMES)}|{n:i for i,n in enumerate(ZH)}
def natural(p):return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',str(p))]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def import_labelme(source,allow_empty=False):
    rows=[];unlabelled=[]
    for image in sorted((p for p in source.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')),key=natural):
        sidecar=image.with_suffix('.json')
        if not sidecar.is_file():unlabelled.append(str(image.relative_to(source)));continue
        data=json.loads(sidecar.read_text(encoding='utf-8-sig'))
        with Image.open(image) as im:im.load();w,h=im.size
        if (data.get('imageWidth'),data.get('imageHeight'))!=(w,h):raise ValueError('Label/image dimensions differ: '+str(sidecar))
        if not data.get('shapes') and not allow_empty:raise ValueError('Empty annotation needs --allow-empty after manual review: '+str(sidecar))
        objects=[]
        for shape in data['shapes']:
            if shape.get('shape_type')!='rectangle' or len(shape['points'])!=2:raise ValueError('Use reviewed rectangle annotations: '+str(sidecar))
            cid=ALIASES.get(shape['label'])
            if cid is None:raise ValueError('Unknown class '+shape['label'])
            (xa,ya),(xb,yb)=shape['points'];box=[min(xa,xb),min(ya,yb),max(xa,xb),max(ya,yb)]
            if not all(math.isfinite(v) for v in box) or not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):raise ValueError('Invalid box: '+str(sidecar))
            pose=shape.get('description','').strip().upper()
            if pose not in ('','UPRIGHT','LYING'):raise ValueError('description must be blank, UPRIGHT or LYING')
            if pose and cid not in (2,3,5):raise ValueError('Pose is only defined for prisms/cylinder')
            objects.append(dict(class_id=cid,box=box,pose=pose))
        rows.append(dict(source=image,sha256=digest(image),size=[w,h],objects=objects,source_relative=image.relative_to(source).as_posix(),group=image.parent.relative_to(source).as_posix()))
    return rows,unlabelled
def label_text(row):
    w,h=row['size'];lines=[]
    for obj in row['objects']:
        x0,y0,x1,y1=obj['box'];lines.append('%d %.8f %.8f %.8f %.8f'%(obj['class_id'],(x0+x1)/2/w,(y0+y1)/2/h,(x1-x0)/w,(y1-y0)/h))
    return '\n'.join(lines)+('\n' if lines else '')
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path);p.add_argument('--allow-empty',action='store_true');args=p.parse_args()
    source=args.source.resolve();out=args.output.resolve();assert source.is_dir() and not out.exists()
    rows,unlabelled=import_labelme(source,args.allow_empty);seen={};allrows=[];reused=0
    if args.base:
        base=args.base.resolve();prior=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
        assert json.loads((base/'classes.json').read_text(encoding='utf-8'))==NAMES
        for row in prior:
            r={**row,'source':(base/row['image']).resolve(),'history':True};assert digest(r['source'])==r['sha256']
            if r['sha256'] in seen:
                assert seen[r['sha256']]['split']==r['split'];continue
            seen[r['sha256']]=r;allrows.append(r)
    new=[]
    for row in rows:
        if row['sha256'] in seen:
            old=seen[row['sha256']]
            if label_text(old)!=label_text(row):raise ValueError('Duplicate image has conflicting annotations: '+row['source_relative'])
            reused+=1;continue
        seen[row['sha256']]=row;new.append(row)
    groups={}
    for row in new:groups.setdefault(row['group'],[]).append(row)
    for group,rs in groups.items():
        if len(rs)<15:raise ValueError('At least 15 unique photos per capture batch needed for grouped train/val/test: '+group)
        # Last two COMPLETE groups are hold-outs. Any remainder stays in train.
        full=len(rs)//5
        for i,row in enumerate(rs):
            block=i//5;row['split']='val' if block==full-2 else 'test' if block==full-1 else 'train'
    allrows+=new;out.mkdir(parents=True);manifest=[];pose={s:[] for s in ('train','val','test')}
    for row in allrows:
        split=row['split'];name=row['sha256'][:24]+row['source'].suffix.lower()
        image=Path('images')/split/name;label=Path('labels')/split/(Path(name).stem+'.txt')
        (out/image).parent.mkdir(parents=True,exist_ok=True);(out/label).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(row['source'],out/image);(out/label).write_text(label_text(row),encoding='utf-8')
        portable={k:v for k,v in row.items() if k!='source'};portable['image']=image.as_posix();manifest.append(portable)
        for obj in row['objects']:
            if obj.get('pose'):pose[split].append(dict(image=image.as_posix(),sha256=row['sha256'],box=obj['box'],class_id=obj['class_id'],label=-1 if obj['pose']=='UPRIGHT' else 1,kind='prior_pose' if row.get('history') else 'supplement_pose_1015'))
    for name,value in [('manifest.json',manifest),('pose_manifest.json',pose),('classes.json',NAMES),('data.yaml',dict(path='.',train='images/train',val='images/val',test='images/test',names=NAMES)),('import_report.json',dict(images=len(manifest),new=len(new),reused=reused,unlabelled_excluded=unlabelled,split_method='Last two complete consecutive 5-image blocks per capture batch; remainder train; historical preserved'))]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('DATASET_READY',out,'images',len(manifest),'new',len(new),'unlabelled excluded',len(unlabelled))
if __name__=='__main__':main()
