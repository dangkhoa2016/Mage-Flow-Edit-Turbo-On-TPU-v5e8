import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from experiment.rope_provider import build_image_rope_sequence

B16 = ROOT / 'runtime' / 'rope' / 'captured-basis-dim16.npy'
B56 = ROOT / 'runtime' / 'rope' / 'captured-basis-dim56.npy'


def test_edit_rope_two_frames_preserves_spatial_and_advances_frame_axis():
    rope = build_image_rope_sequence(
        'control', [(1, 2, 2), (1, 2, 2)], B16, B56
    )
    assert rope.shape == (8, 64, 2)
    tgt = rope[:4]
    ref = rope[4:]
    # Spatial axes (8:64 complex bins) are identical for equally-sized frames.
    assert np.array_equal(tgt[:, 8:, :], ref[:, 8:, :])
    # Frame 0 is phase zero; frame 1 must be distinct on the 8 frame bins.
    assert np.array_equal(tgt[:, :8, 0], np.ones((4, 8), dtype=np.float32))
    assert np.array_equal(tgt[:, :8, 1], np.zeros((4, 8), dtype=np.float32))
    assert not np.array_equal(tgt[:, :8, :], ref[:, :8, :])


if __name__ == '__main__':
    test_edit_rope_two_frames_preserves_spatial_and_advances_frame_axis()
    print('EDIT_ROPE_TEST=PASS')
