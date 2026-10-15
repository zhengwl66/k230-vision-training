"""Run the current 65-epoch YOLO11n recipe with local, portable paths."""
import argparse,json,hashlib,time
from pathlib import Path
from common import read_data,load_engine,resolve_device,assert_yolo11n,dump,versions
REPO=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--weights',type=Path,default=REPO/'models/current/shape_yolo11n.pt');p.add_argument('--scratch',action='store_true');p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int);p.add_argument('--batch',type=int);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    _,data,counts=read_data(args.data);names=json.loads((REPO/'configs/classes.json').read_text(encoding='utf-8'));assert data['names']==names
    out=args.run.resolve()/'shape_training';assert not out.exists(),'Output already exists'
    options=json.loads((REPO/'configs/shape_train.json').read_text(encoding='utf-8'))
    if args.epochs:options['epochs']=args.epochs
    if args.batch:options['batch']=args.batch
    options['close_mosaic']=min(options['close_mosaic'],options['epochs']);assert options['epochs']>0 and options['batch']>0
    if not args.scratch:assert args.weights.is_file()
    plan=dict(options=options,counts=counts,initial_weights='yolo11n.yaml random initialization' if args.scratch else str(args.weights.resolve()),initial_sha256=None if args.scratch else hashlib.sha256(args.weights.read_bytes()).hexdigest(),test_used_for_selection=False)
    if args.dry_run:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    torch,YOLO=load_engine(out);torch.set_num_threads(6)
    dump(out/'data_resolved.yaml',data);dump(out/'training_plan.json',{**plan,'versions':versions()})
    model=YOLO('yolo11n.yaml' if args.scratch else str(args.weights));assert_yolo11n(model,None if args.scratch else names)
    options.update(data=str(out/'data_resolved.yaml'),project=str(out),name='fit',device=resolve_device(torch,args.device))
    start=time.monotonic();model.train(**options)
    dump(out/'training_result.json',dict(best=str(model.trainer.best),last=str(model.trainer.last),seconds=time.monotonic()-start,versions=versions()))
    print('TRAINING_DONE',model.trainer.best)
if __name__=='__main__':main()
