import argparse,hashlib,json,shutil
from pathlib import Path
from common import load_engine,assert_yolo11n
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    _,YOLO=load_engine(args.output);model=YOLO(str(args.weights));assert_yolo11n(model)
    generated=model.export(format='onnx',imgsz=320,opset=12,dynamic=False,simplify=False,half=False,batch=1,device='cpu')
    dest=args.output/'shape_yolo11n.onnx';shutil.copy2(generated,dest);shutil.copy2(args.weights,args.output/'shape_yolo11n.pt')
    (args.output/'export.json').write_text(json.dumps(dict(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),input_shape=[1,3,320,320],output_shape=[1,11,2100],opset=12,nms=False),indent=2),encoding='utf-8')
    print('ONNX_READY',dest)
if __name__=='__main__':main()
