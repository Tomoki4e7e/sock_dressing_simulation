import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from process_head_reaching_shareset import calibrate, PROFILE


def test_full_training_calibration_keeps_unsampled_peaks_and_excludes_test(tmp_path):
    checkpoint=tmp_path/'weights';checkpoint.write_bytes(b'test')
    train=tmp_path/'data/data_sock_reaching_v1/train'
    for number in (10,13,25,38):
        for variant in (0,1):
            ep=train/f'pose-{number}-v{variant}';(ep/'camera_right').mkdir(parents=True)
            (ep/'metadata.json').write_text(json.dumps(dict(raw_validation_passed=True,camera_profile=PROFILE,
                frames=2,workbook_pattern=dict(number=number))))
            for index in range(2):
                value=100 if number==38 and variant==1 and index==1 else 1
                Image.fromarray(np.full((4,4,3),value,dtype='uint8')).save(ep/'camera_right'/f'{index}.png')
    held=tmp_path/'data/data_sock_reaching_v1/test/pose-50/camera_right';held.mkdir(parents=True)
    Image.fromarray(np.full((4,4,3),255,dtype='uint8')).save(held/'0.png')
    class Model:
        calls=0
        def infer_image(self,image,input_size):
            self.calls+=1
            return image[:,:,0].astype('float32')
    model=Model();cal=calibrate(SimpleNamespace(output=tmp_path,depth_checkpoint=checkpoint),model)
    assert model.calls==16 and cal['training_frames']==16
    assert cal['depth_min']==0 and cal['depth_max']==pytest.approx(100*256/255)
    assert {s['number'] for s in cal['samples']}=={10,13,25,38}
