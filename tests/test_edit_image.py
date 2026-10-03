import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from experiment.edit_image import preprocess_reference

def test_preprocess_reference_rgb_nchw_minus1_plus1():
    arr=np.array([[[0,127,255],[255,127,0]],[[64,128,192],[32,160,224]]],dtype=np.uint8)
    im=Image.fromarray(arr,'RGB')
    out=preprocess_reference(im,2,2)
    assert out.shape==(1,3,2,2)
    assert out.dtype==np.float32
    assert np.isclose(out[0,0,0,0],-1.0)
    assert np.isclose(out[0,2,0,0],1.0)
    assert np.isclose(out[0,0,0,1],1.0)
    assert np.isclose(out[0,2,0,1],-1.0)
    assert out.min()>=-1.0 and out.max()<=1.0

if __name__=='__main__':
    test_preprocess_reference_rgb_nchw_minus1_plus1()
    print('EDIT_IMAGE_TEST=PASS')
