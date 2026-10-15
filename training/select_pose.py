"""Fit geometry and select CNN/fusion using validation only, never test."""
import argparse,json,sys,math,copy,hashlib,shutil
from pathlib import Path
import numpy as np,onnxruntime as ort
from PIL import Image
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'deployment/board_files/sdcard/vision_update_20261013'))
import state_sample as state
from geometry import target,kernel,fit_rbf

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def scores(profile,x):
    basis=np.array(profile['support_vectors']);score=kernel(x,basis,profile['gamma'])@np.array(profile['coefficients'])
    nearest=np.maximum(0,((x[:,None]-basis[None])**2).sum(2)).min(1)
    return score,nearest
def metrics(values,labels,margin,eligible):
    accepted=(np.abs(values)>=margin)&eligible;correct=accepted&((values>0)==(labels>0));wrong=accepted&~correct
    return dict(images=len(labels),correct=int(correct.sum()),wrong=int(wrong.sum()),unknown=int((~accepted).sum()))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--baseline',type=Path,default=REPO/'deployment/board_files/sdcard/vision_update_20261013/state_model.json');parser.add_argument('--baseline-cnn',type=Path,default=REPO/'models/current/state.onnx');parser.add_argument('--baseline-pose-pt',type=Path,default=REPO/'models/current/pose_classifier.pt');parser.add_argument('--version',default='pose-custom-v1');args=parser.parse_args();ROOT=args.run.resolve()
    rows=json.loads((ROOT/'pose_manifest.json').read_text(encoding='utf-8'));data=np.load(ROOT/'pose_features.npz');features=data['features']
    original=json.loads((args.baseline).read_text(encoding='utf-8'));profiles={};fitting={}
    for cid in (2,3,5):
        train=np.array([i for i,r in enumerate(rows) if r['class_id']==cid and r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['class_id']==cid and r['split']=='val'])
        basis=features[train];aug_mask=data['aug_class_ids']==cid;aug=data['augmented'][aug_mask];labels=data['aug_labels'][aug_mask]
        truth=np.array([rows[i]['label'] for i in val]);trials=[]
        for gamma in (.01,.025,.05,.1,.2,.4):
            for reg in (.03,.1,.3,1.):
                coef=fit_rbf(aug,labels,basis,gamma,reg);pred=kernel(features[val],basis,gamma)@coef
                rank=(int(np.sum((pred>0)==(truth>0))),-float(np.mean((pred-truth)**2)))
                trials.append((rank,gamma,reg,coef))
        rank,gamma,reg,coef=max(trials,key=lambda item:item[0]);dist=((basis[:,None]-basis[None])**2).sum(2);np.fill_diagonal(dist,np.inf)
        max_distance=max(12.,float(np.percentile(dist.min(1),95)*3.))
        profile={**original['profiles'][str(cid)],'gamma':gamma,'regularization':reg,'margin':.15,'max_distance':max_distance,
                 'support_vectors':np.round(basis,7).tolist(),'coefficients':np.round(coef,8).tolist(),'training_images':len(train),
                 'validation_images':len(val),'augmented_training_examples':len(aug),'enabled':True}
        profiles[str(cid)]={'previous':original['profiles'][str(cid)],'supplement':profile};fitting[str(cid)]=dict(validation_correct_without_rejection=rank[0],gamma=gamma,regularization=reg)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;valrows=[(i,r) for i,r in enumerate(rows) if r['split']=='val'];cnn_values={}
    cnn_paths={'previous':args.baseline_cnn,'supplement':ROOT/'pose_cnn/state.onnx'}
    for name,path in cnn_paths.items():
        session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider']);values={}
        for i,r in valrows:
            image=Image.open(r['image']).convert('RGB').crop(tuple(r['box'])).resize((96,96),Image.Resampling.BILINEAR)
            x=np.ascontiguousarray(np.array(image).transpose(2,0,1)[None],dtype=np.float32)/255.
            out=session.run(None,{'images':x})[0][0];pair={2:0,3:1,5:2}[r['class_id']]*2;values[i]=math.tanh(float(out[pair+1]-out[pair])/2)
        cnn_values[name]=values
    choices={}
    for cnn_name in cnn_paths:
        chosen={};all_trials={}
        for cid in (2,3,5):
            ids=np.array([i for i,r in valrows if r['class_id']==cid]);truth=np.array([rows[i]['label'] for i in ids]);prior=np.array([rows[i]['kind']!='supplement_pose_1015' for i in ids]);cn=np.array([cnn_values[cnn_name][i] for i in ids]);trials=[]
            oldscores,olddist=scores(original['profiles'][str(cid)],features[ids]);oldcn=np.array([cnn_values['previous'][i] for i in ids]);oldfusion=original['fusion'][str(cid)]
            oldvalues=oldfusion['geometry_weight']*oldscores+(1-oldfusion['geometry_weight'])*oldcn
            oldmetrics=metrics(oldvalues[prior],truth[prior],oldfusion['margin'],olddist[prior]<=original['profiles'][str(cid)]['max_distance'])
            for profile_name,profile in profiles[str(cid)].items():
                gs,dist=scores(profile,features[ids]);eligible=dist<=profile['max_distance']
                for alpha in (0.,.25,.5,.75,1.):
                    values=alpha*gs+(1-alpha)*cn;wrongs=np.abs(values[((values>0)!=(truth>0))&eligible]);floor=.25 if cid==3 else .15
                    margins=sorted(set(max(floor,m) for m in ([floor,.25,.4,.55,.7,.85]+([float(wrongs.max()+.04)] if len(wrongs) else []))))
                    for margin in margins:
                        m=metrics(values,truth,margin,eligible);pm=metrics(values[prior],truth[prior],margin,eligible[prior]);sm=metrics(values[~prior],truth[~prior],margin,eligible[~prior])
                        preserve=(pm['correct']>=oldmetrics['correct'] and pm['wrong']<=oldmetrics['wrong'])
                        # For equal validation outcomes, keep geometry-only on
                        # the simpler shapes to avoid extra per-object KPU work.
                        efficiency=-abs(alpha-.5) if cid==3 else alpha
                        rank=(preserve,m['wrong']==0,m['correct']-3*m['wrong'],-m['wrong'],efficiency,-margin)
                        trials.append(dict(profile=profile_name,geometry_weight=alpha,margin=margin,metrics=m,prior_metrics=pm,supplement_metrics=sm,preserve_prior_validation=preserve,rank=rank))
            best=max(trials,key=lambda r:r['rank']);chosen[str(cid)]=best;all_trials[str(cid)]=trials
            print('POSE_VAL_CHOICE',cnn_name,cid,best,flush=True)
        overall=dict(correct=sum(r['metrics']['correct'] for r in chosen.values()),wrong=sum(r['metrics']['wrong'] for r in chosen.values()),unknown=sum(r['metrics']['unknown'] for r in chosen.values()))
        rank=(all(r['preserve_prior_validation'] for r in chosen.values()),overall['wrong']==0,overall['correct']-3*overall['wrong'],-overall['wrong'],cnn_name=='supplement')
        choices[cnn_name]=dict(chosen=chosen,overall=overall,rank=rank,trials=all_trials)
    selected=max(choices,key=lambda n:choices[n]['rank']);config=copy.deepcopy(original)
    for cid,r in choices[selected]['chosen'].items():
        config['profiles'][cid]=profiles[cid][r['profile']];config['fusion'][cid]=dict(geometry_weight=r['geometry_weight'],margin=r['margin'])
    config.update(version=args.version,training_scope='Explicit labelled train split only; previous/new model and fusion selected on validation',
                  selection_split='validation only',test_used_for_selection=False)
    (ROOT/'state_model.json').write_text(json.dumps(config,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    out=ROOT/'selected_pose';out.mkdir(exist_ok=True)
    shutil.copy2(cnn_paths[selected],out/'state.onnx')
    shutil.copy2(args.baseline_pose_pt if selected=='previous' else ROOT/'pose_cnn/pose_classifier.pt',out/'pose_classifier.pt')
    dump(ROOT/'pose_selection.json',dict(selected_cnn=selected,cnn_path=str(cnn_paths[selected]),fitting=fitting,choices=choices,selection_split='validation only',test_used_for_selection=False,
                                      version=config['version'],fusion=config['fusion'],state_model_sha256=hashlib.sha256((ROOT/'state_model.json').read_bytes()).hexdigest()))
    print('POSE_SELECTED',selected,choices[selected]['overall'],config['fusion'],flush=True)
if __name__=='__main__':main()
