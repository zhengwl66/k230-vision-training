"""Generate nominal/augmented features from training only; keep holdouts intact."""
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2
from PIL import Image
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'deployment/board_files/sdcard/vision_update_20261013'))
import state_sample as state
from geometry import target,kernel,fit_rbf

def vector(pixels,box,cid):
    result,diag=state.extract_features(pixels,target(box,cid))
    if result is None:raise ValueError(str(diag))
    return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--baseline',type=Path,default=REPO/'deployment/board_files/sdcard/vision_update_20261013/state_model.json');parser.add_argument('--baseline-cnn',type=Path,default=REPO/'models/current/state.onnx');parser.add_argument('--baseline-pose-pt',type=Path,default=REPO/'models/current/pose_classifier.pt');parser.add_argument('--version',default='pose-custom-v1');args=parser.parse_args();ROOT=args.run.resolve()
    rows=json.loads((ROOT/'pose_cnn/data_manifest.json').read_text(encoding='utf-8'));rng=np.random.default_rng(2026101305)
    allrows=[];features=[];aug=[];aug_labels=[];aug_cids=[]
    for split,rs in rows.items():
        for r in rs:
            pixels=np.array(Image.open(r['image']).convert('RGB'));cid=r['class_id'];box=r['box']
            feature=vector(pixels,box,cid);allrows.append({**r,'split':split});features.append(feature)
            if split!='train':continue
            aug.append(feature);aug_labels.append(r['label']);aug_cids.append(cid)
            h,w=pixels.shape[:2]
            aug.append(vector(cv2.resize(pixels,(w//2,h//2),interpolation=cv2.INTER_AREA),[v/2 for v in box],cid));aug_labels.append(r['label']);aug_cids.append(cid)
            for _ in range(2):
                modified=np.clip(pixels.astype('float32')*rng.uniform(.9,1.1)+rng.uniform(-5,5),0,255).astype('uint8')
                bw,bh=box[2]-box[0],box[3]-box[1]
                jitter=[box[0]+rng.uniform(-.025,.025)*bw,box[1]+rng.uniform(-.025,.025)*bh,box[2]+rng.uniform(-.025,.025)*bw,box[3]+rng.uniform(-.025,.025)*bh]
                aug.append(vector(modified,jitter,cid));aug_labels.append(r['label']);aug_cids.append(cid)
    (ROOT/'pose_manifest.json').write_text(json.dumps(allrows,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez(ROOT/'pose_features.npz',features=np.array(features),augmented=np.array(aug),aug_labels=np.array(aug_labels),aug_class_ids=np.array(aug_cids))
    print('POSE_FEATURES_READY',len(allrows),'augmented_train',len(aug),flush=True)
if __name__=='__main__':main()
