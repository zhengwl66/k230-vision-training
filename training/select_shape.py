"""Select previous/best/last on validation only; test is never opened here."""
import argparse,json,shutil
from pathlib import Path
from common import load_engine
from evaluate import Detector,evaluate,rows_for
REPO=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--baseline',type=Path,default=REPO/'models/current/shape_yolo11n.onnx');p.add_argument('--baseline-pt',type=Path,default=REPO/'models/current/shape_yolo11n.pt');args=p.parse_args();root=args.run.resolve()
    _,YOLO=load_engine(root/'exports');paths={'previous':args.baseline};pts={'previous':args.baseline_pt}
    for name in ('best','last'):
        pt=root/'shape_training/fit/weights'/(name+'.pt');dest=root/'exports'/name;dest.mkdir(parents=True,exist_ok=True)
        generated=YOLO(str(pt)).export(format='onnx',imgsz=320,opset=12,dynamic=False,simplify=False,half=False,batch=1,device='cpu')
        shutil.copy2(generated,dest/'shape_yolo11n.onnx');paths[name]=dest/'shape_yolo11n.onnx';pts[name]=pt
    rows=rows_for(args.data,'val');reports={};ranks={}
    for name,path in paths.items():
        reports[name]=evaluate(Detector(path),rows)
        if name=='previous':baseline=reports[name]
        eligible=all(reports[name][c]['precision']>=baseline[c]['precision']-.005 and reports[name][c]['f1']>=baseline[c]['f1']-.025 for c in ('0.55','0.7'))
        ranks[name]=(eligible,(reports[name]['0.55']['f1']+reports[name]['0.7']['f1'])/2,name!='previous')
    chosen=max(ranks,key=ranks.get);assert ranks[chosen][0];out=root/'selected_onnx';assert not out.exists();out.mkdir()
    shutil.copy2(paths[chosen],out/'shape_yolo11n.onnx');shutil.copy2(pts[chosen],out/'shape_yolo11n.pt')
    (root/'shape_selection.json').write_text(json.dumps(dict(selected=chosen,metrics=reports,ranks=ranks,selection_split='val',test_used_for_selection=False),ensure_ascii=False,indent=2),encoding='utf-8');print('SHAPE_SELECTED',chosen,ranks)
if __name__=='__main__':main()
