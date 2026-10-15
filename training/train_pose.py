"""Fine-tune pose CNN on visually reviewed crops and prior labelled examples."""
import argparse,copy,json,sys,hashlib,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
REPO=Path(__file__).resolve().parents[1]
from pose_network import PoseNet,Captures,evaluate,MAP,SIDE
from common import load_engine

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--shape-weights',type=Path,default=REPO/'models/current/shape_yolo11n.pt');parser.add_argument('--init',type=Path,default=REPO/'models/current/pose_classifier.pt');parser.add_argument('--scratch',action='store_true');parser.add_argument('--epochs',type=int,default=90);args=parser.parse_args();assert args.epochs>0
    ROOT=args.run.resolve();out=ROOT/'pose_cnn';assert not out.exists(),'pose_cnn already exists';out.mkdir(parents=True)
    torch.set_num_threads(4);seed=2026101504;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    original=json.loads(args.manifest.read_text(encoding='utf-8'))
    for rs in original.values():
        for r in rs:r['image']=str((args.manifest.parent/r['image']).resolve())
    rows={s:[dict(r) for r in original[s]] for s in ('train','val','test')}
    for split,rs in rows.items():
        assert rs,split+' has no pose annotations'
        for cid in (2,3,5):
            assert {-1,1}<={r['label'] for r in rs if r['class_id']==cid},split+' requires both states for class '+str(cid)
    hashes={}
    for split,rs in rows.items():
        for r in rs:hashes.setdefault(hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),set()).add(split)
    assert all(len(s)==1 for s in hashes.values()),'Pose cross-split leakage'
    out=ROOT/'pose_cnn';out.mkdir(exist_ok=True);dump(out/'data_manifest.json',rows)
    _,YOLO=load_engine(out);detector=YOLO(str(args.shape_weights)).model
    model=PoseNet(nn.Sequential(*[copy.deepcopy(detector.model[i]) for i in range(5)]))
    if not args.scratch:
        checkpoint=torch.load(args.init,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict']);del checkpoint
    del detector
    device='cuda:0' if torch.cuda.is_available() else 'cpu';model.to(device)
    loaders={s:DataLoader(Captures(rs,s=='train'),batch_size=32,shuffle=s=='train',num_workers=0) for s,rs in rows.items()}
    optimizer=torch.optim.AdamW([{'params':model.stem.parameters(),'lr':.00008},{'params':model.head.parameters(),'lr':.0004}],weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00001)
    criterion=nn.CrossEntropyLoss(label_smoothing=.03);best=None;rank_best=None;patience=0;history=[];begin=time.monotonic()
    for epoch in range(args.epochs):
        model.train();loss_sum=0.
        for x,y in loaders['train']:
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();loss_sum+=float(loss.detach().cpu())*len(y)
        scheduler.step();logits,labels=evaluate(model,loaders['val'],device)
        pairs=logits.reshape(-1,3,2)[np.arange(len(labels)),labels//2];correct=(pairs.argmax(1)==labels%2)
        val_loss=float(criterion(torch.from_numpy(logits),torch.from_numpy(labels)));rank=(int(correct.sum()),-val_loss)
        if rank_best is None or rank>rank_best:
            rank_best=rank;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1;patience=0
        else:patience+=1
        history.append(dict(epoch=epoch+1,train_loss=loss_sum/len(rows['train']),val_pose_correct=int(correct.sum()),val_images=len(labels),val_loss=val_loss))
        if (epoch+1)%10==0:print('POSE_SUPPLEMENT',epoch+1,'val',int(correct.sum()),'/',len(labels),'loss',round(val_loss,4),flush=True)
        if epoch>=35 and patience>=24:break
    model.load_state_dict(best);model.cpu().eval()
    torch.save(dict(state_dict={k:v.cpu() for k,v in best.items()},side=96,class_ids=[2,3,5],labels=['UPRIGHT','LYING'],best_epoch=best_epoch),out/'pose_classifier.pt')
    torch.onnx.export(model,torch.zeros(1,3,96,96),str(out/'state.onnx'),opset_version=12,input_names=['images'],output_names=['state_logits'],dynamo=False)
    import onnx,onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(out/'state.onnx')))
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(out/'state.onnx'),opts,providers=['CPUExecutionProvider']);x=np.random.default_rng(1).random((1,3,96,96),dtype=np.float32)
    with torch.inference_mode():expected=model(torch.from_numpy(x)).numpy()
    assert np.allclose(expected,session.run(None,{'images':x})[0],atol=1e-4,rtol=1e-4)
    calibration=out/'calibration';calibration.mkdir(exist_ok=True)
    for i,r in enumerate(rows['train']):Image.open(r['image']).convert('RGB').crop(tuple(r['box'])).resize((96,96),Image.Resampling.BILINEAR).save(calibration/(str(i).zfill(4)+'.png'))
    report=dict(best_epoch=best_epoch,initial_weights=str(args.init) if not args.scratch else None,split_counts={s:len(rs) for s,rs in rows.items()},
                history=history,seconds=time.monotonic()-begin,onnx_cpu_equivalence=True,test_used_for_selection=False,
                new_pose_ground_truth='Explicit per-object UPRIGHT/LYING annotations; never inferred from shape or color',
                onnx_sha256=hashlib.sha256((out/'state.onnx').read_bytes()).hexdigest())
    dump(out/'training_report.json',report);print('POSE_SUPPLEMENT_DONE',best_epoch,report['split_counts'],flush=True)
if __name__=='__main__':main()
