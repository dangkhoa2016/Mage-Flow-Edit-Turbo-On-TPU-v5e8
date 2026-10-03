import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

from experiment import transformer_driver


def test_edit_sequence_helper_exists_and_preserves_replica_layout():
    assert hasattr(transformer_driver, "pack_edit_image_sequence")
    target = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    reference = target + 1000
    packed, cu = transformer_driver.pack_edit_image_sequence(target, reference)

    assert packed.shape == (1, 16, 3)
    assert cu.tolist() == [0, 8, 16]

    t0 = target[0].transpose(1, 2, 0).reshape(4, 3)
    r0 = reference[0].transpose(1, 2, 0).reshape(4, 3)
    t1 = target[1].transpose(1, 2, 0).reshape(4, 3)
    r1 = reference[1].transpose(1, 2, 0).reshape(4, 3)
    np.testing.assert_array_equal(packed[0], np.concatenate([t0, r0, t1, r1], axis=0))


def test_extract_target_prediction_skips_reference_blocks():
    assert hasattr(transformer_driver, "extract_edit_target_prediction")
    packed = np.arange(16 * 3, dtype=np.float32).reshape(1, 16, 3)
    got = transformer_driver.extract_edit_target_prediction(packed, replicas=2, target_tokens=4)
    assert got.shape == (2, 4, 3)
    np.testing.assert_array_equal(got[0], packed[0, 0:4])
    np.testing.assert_array_equal(got[1], packed[0, 8:12])
