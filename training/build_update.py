"""Build a versioned board update after conversion and held-out evaluation."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ACTIVE='sdcard/vision_update_20261013'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--state-json',type=Path,required=True);p.add_argument('--test-report',type=Path,required=True);p.add_argument('--version',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    report=json.loads(args.test_report.read_text(encoding='utf-8'));assert report['split']=='test','Use a held-out test report'
    for k in ('shape','state','state_json'):assert getattr(args,k).is_file()
    shutil.copytree(REPO/'deployment/board_files',args.output/'board_files');runtime=args.output/'board_files'/ACTIVE
    for path,name in ((args.shape,'shape.kmodel'),(args.state,'state.kmodel'),(args.state_json,'state_model.json')):shutil.copy2(path,runtime/name)
    cfg=json.loads((runtime/'inference_config.json').read_text(encoding='utf-8'));pose=json.loads(args.state_json.read_text(encoding='utf-8'))
    for key in list(cfg):
        if key.startswith('board_'):cfg.pop(key)
    cfg.update(version=args.version,model_sha256=sha(runtime/'shape.kmodel'),state_kmodel_sha256=sha(runtime/'state.kmodel'),state_model_sha256=sha(runtime/'state_model.json'),state_model_version=pose['version'],board_verified=False,verification_scope='New custom weights; computer evaluation only. Requires own K230 check.')
    (runtime/'inference_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(args.test_report,args.output/'test_report.json')
    manifest={p.relative_to(args.output).as_posix():sha(p) for p in args.output.rglob('*') if p.is_file()};(args.output/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=args.output.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in args.output.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(args.output).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    print('UPDATE_PACKAGE_READY',archive)
if __name__=='__main__':main()
