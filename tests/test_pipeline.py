import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'training'));sys.path.insert(0,str(REPO/'deployment/board_files/sdcard/vision_update_20261013'))
from prepare_dataset import import_labelme,label_text
from common import read_data
from audit_dataset import main as audit_main
from color_sample import delta_e00
from box_adapter import centered_letterbox_geometry
class PipelineTests(unittest.TestCase):
    def test_ciede2000_reference_vectors(self):
        vectors=json.loads((REPO/'tests/ciede2000_vectors.json').read_text(encoding='utf-8'))
        self.assertEqual(len(vectors),34)
        for v in vectors:self.assertAlmostEqual(delta_e00(v['lab1'],v['lab2'],1.),v['expected'],delta=.001)
    def make_label(self,root,label='cylinder',pose='UPRIGHT',empty=False):
        Image.new('RGB',(100,80),(90,70,40)).save(root/'sample.jpg')
        data=dict(imageWidth=100,imageHeight=80,shapes=[] if empty else [dict(label=label,shape_type='rectangle',points=[[10,20],[50,60]],description=pose)])
        (root/'sample.json').write_text(json.dumps(data),encoding='utf-8')
    def test_annotation_class_order_and_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_label(root);rows,missing=import_labelme(root)
            self.assertEqual(missing,[]);self.assertEqual(rows[0]['objects'][0]['class_id'],5)
            self.assertEqual(rows[0]['objects'][0]['pose'],'UPRIGHT');self.assertEqual(label_text(rows[0]),'5 0.30000000 0.50000000 0.40000000 0.50000000\n')
    def test_unknown_class_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_label(root,label='filename_guess')
            with self.assertRaises(ValueError):import_labelme(root)
    def test_empty_must_be_explicitly_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_label(root,empty=True)
            with self.assertRaises(ValueError):import_labelme(root)
            self.assertEqual(import_labelme(root,True)[0][0]['objects'],[])
    def test_missing_annotations_not_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(100,80)).save(root/'unlabelled.jpg')
            rows,missing=import_labelme(root,True);self.assertEqual(rows,[]);self.assertEqual(missing,['unlabelled.jpg'])
    def test_same_image_across_splits_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);names=json.loads((REPO/'configs/classes.json').read_text(encoding='utf-8'))
            for si,split in enumerate(('train','val','test')):
                (root/'images'/split).mkdir(parents=True);(root/'labels'/split).mkdir(parents=True)
                for cid in range(7):
                    color=(cid*20,si*40,50)
                    if split=='test' and cid==0:color=(0,0,50)
                    Image.new('RGB',(20,20),color).save(root/'images'/split/(str(cid)+'.png'))
                    (root/'labels'/split/(str(cid)+'.txt')).write_text(str(cid)+' .5 .5 .5 .5\n',encoding='utf-8')
            path=root/'data.yaml';path.write_text(json.dumps(dict(path='.',names=names,train='images/train',val='images/val',test='images/test')),encoding='utf-8')
            with patch.object(sys,'argv',['audit_dataset.py','--data',str(path)]):
                with self.assertRaisesRegex(ValueError,'leakage'):audit_main()
    def test_deployment_hashes(self):
        import hashlib
        base=REPO/'deployment/board_files/sdcard/vision_update_20261013';cfg=json.loads((base/'inference_config.json').read_text(encoding='utf-8'))
        for name,key in [('shape.kmodel','model_sha256'),('state.kmodel','state_kmodel_sha256'),('state_model.json','state_model_sha256')]:self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(),cfg[key])
if __name__=='__main__':unittest.main()
