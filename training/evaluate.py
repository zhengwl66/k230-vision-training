"""Fixed deployment decoding, ONNX/K230 simulator, no model fitting."""
import argparse,json,sys,math
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
from PIL import Image
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'deployment/board_files/sdcard/vision_update_20261013'))
from shape_k230 import decode_raw,_iou
from common import read_data
from geometry import target
import state_sample as state
def tensor(pixels):
    h,w=pixels.shape[:2];scale=min(320/w,320/h);nw,nh=round(w*scale),round(h*scale)
    result=np.full((320,320,3),114,np.uint8);left,top=(320-nw)//2,(320-nh)//2
    result[top:top+nh,left:left+nw]=cv2.resize(pixels,(nw,nh),interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(result.transpose(2,0,1)[None]),scale,left,top
class Detector:
    def __init__(self,path):
        self.quantized=path.suffix=='.kmodel'
        if self.quantized:
            from compile_kmodel import setup_runtime
            setup_runtime();import nncase
            self.nncase=nncase;self.engine=nncase.Simulator();self.engine.load_model(path.read_bytes())
        else:
            opts=ort.SessionOptions();opts.intra_op_num_threads=4;self.engine=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
    def predict(self,pixels):
        x,scale,left,top=tensor(pixels)
        if self.quantized:
            self.engine.set_input_tensor(0,self.nncase.RuntimeTensor.from_numpy(x));self.engine.run();raw=self.engine.get_output_tensor(0).to_numpy().copy()
        else:raw=self.engine.run(None,{self.engine.get_inputs()[0].name:x.astype('float32')/255.})[0]
        assert raw.shape==(1,11,2100) and np.isfinite(raw).all()
        result=decode_raw(raw,.10,.45);h,w=pixels.shape[:2]
        for d in result:
            a,b,c,e=d['xyxy'];d['xyxy']=[max(0,min(w,(a-left)/scale)),max(0,min(h,(b-top)/scale)),max(0,min(w,(c-left)/scale)),max(0,min(h,(e-top)/scale))]
        return result
def rows_for(data,split):
    _,resolved,_=read_data(data,needed=(split,));rows=[]
    for image in sorted(Path(resolved[split]).rglob('*')):
        if image.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.webp'):continue
        with Image.open(image) as im:w,h=im.size
        parts=list(image.parts);parts[len(parts)-1-parts[::-1].index('images')]='labels';label=Path(*parts).with_suffix('.txt');gt=[]
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            cid,cx,cy,bw,bh=map(float,line.split());gt.append((int(cid),[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        rows.append(dict(image=image,gt=gt))
    return rows
def summarize(matrix):
    tp,fp,fn=map(int,matrix.sum(0));p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(1e-12,p+r),per_class_tp_fp_fn=matrix.tolist())
def evaluate(engine,rows):
    totals={str(c):np.zeros((7,3),int) for c in (.5,.55,.7)}
    for row in rows:
        pred=engine.predict(np.array(Image.open(row['image']).convert('RGB')))
        for conf,matrix in totals.items():
            used=set()
            for d in sorted(pred,key=lambda d:-d['confidence']):
                if d['confidence']<float(conf):continue
                matches=[(_iou(d['xyxy'],b),i) for i,(cid,b) in enumerate(row['gt']) if cid==d['class_id'] and i not in used]
                overlap,index=max(matches,default=(0,None))
                if overlap>=.5:used.add(index);matrix[d['class_id'],0]+=1
                else:matrix[d['class_id'],1]+=1
            for i,(cid,b) in enumerate(row['gt']):
                if i not in used:matrix[cid,2]+=1
    return {c:summarize(m) for c,m in totals.items()}
class PoseEngine:
    def __init__(self,path):
        self.quantized=path.suffix=='.kmodel'
        if self.quantized:
            from compile_kmodel import setup_runtime
            setup_runtime();import nncase
            self.nncase=nncase;self.engine=nncase.Simulator();self.engine.load_model(path.read_bytes())
        else:
            opts=ort.SessionOptions();opts.intra_op_num_threads=4;self.engine=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
    def predict(self,pixels,t,pair):
        h,w=pixels.shape[:2];x0=max(0,int(t['x']));y0=max(0,int(t['y']));x1=min(w,int(t['x']+t['w']));y1=min(h,int(t['y']+t['h']))
        x=np.ascontiguousarray(cv2.resize(pixels[y0:y1,x0:x1],(96,96)).transpose(2,0,1)[None],dtype=np.uint8)
        if self.quantized:
            self.engine.set_input_tensor(0,self.nncase.RuntimeTensor.from_numpy(x));self.engine.run();out=self.engine.get_output_tensor(0).to_numpy().copy()[0]
        else:out=self.engine.run(None,{'images':x.astype('float32')/255.})[0][0]
        return math.tanh(float(out[pair*2+1]-out[pair*2])/2)
def evaluate_pose(detector,pose_path,state_path,manifest,split):
    rows=json.loads(manifest.read_text(encoding='utf-8'))[split];state.set_model(json.loads(state_path.read_text(encoding='utf-8')));state._image_engine=PoseEngine(pose_path)
    result=dict(images=len(rows),joint_correct=0,wrong_pose=0,unknown=0,shape_missed=0)
    for r in rows:
        pixels=np.array(Image.open((manifest.parent/r['image']).resolve()).convert('RGB'));pred=detector.predict(pixels)
        matches=[d for d in pred if d['confidence']>=.55 and d['class_id']==r['class_id'] and _iou(d['xyxy'],r['box'])>=.5]
        if not matches:result['shape_missed']+=1;continue
        t=target(max(matches,key=lambda d:d['confidence'])['xyxy'],r['class_id']);state.attach_state(pixels,t)
        truth='UPRIGHT' if r['label']<0 else 'LYING';hint=t['state_hint']
        result['joint_correct']+=int(hint==truth);result['unknown']+=int(hint=='UNKNOWN');result['wrong_pose']+=int(hint in ('UPRIGHT','LYING') and hint!=truth)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--split',choices=('val','test'),default='val');p.add_argument('--output',type=Path,required=True);p.add_argument('--pose-model',type=Path);p.add_argument('--state-json',type=Path);p.add_argument('--pose-manifest',type=Path);args=p.parse_args();assert not args.output.exists()
    engine=Detector(args.model);rows=rows_for(args.data,args.split);report=dict(split=args.split,images=len(rows),shape=evaluate(engine,rows),physical_board_verified=False)
    if args.pose_model:
        assert args.state_json and args.pose_manifest;report['pose']=evaluate_pose(engine,args.pose_model,args.state_json,args.pose_manifest,args.split)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
