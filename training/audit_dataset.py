import argparse,json,hashlib
from pathlib import Path
from common import read_data
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);args=p.parse_args()
    _,resolved,counts=read_data(args.data,needed=('train','val','test'));seen={}
    for split in ('train','val','test'):
        for image in Path(resolved[split]).rglob('*'):
            if image.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp','.webp'):continue
            h=hashlib.sha256(image.read_bytes()).hexdigest()
            if h in seen and seen[h]!=split:raise ValueError('Exact-image leakage: '+str(image))
            seen[h]=split
    print(json.dumps(dict(passed=True,counts=counts,unique_images=len(seen),same_session_similarity_not_excluded=True),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
