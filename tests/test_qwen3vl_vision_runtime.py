import numpy as np


def test_position_ids_merge2_order():
    from experiment import qwen3vl_vision_runtime as vr
    got = vr.vision_position_ids(np.array([[1, 4, 4]], np.int32), 2)
    expected = np.array([
        [0,0],[0,1],[1,0],[1,1],
        [0,2],[0,3],[1,2],[1,3],
        [2,0],[2,1],[3,0],[3,1],
        [2,2],[2,3],[3,2],[3,3],
    ], np.int32)
    assert np.array_equal(got, expected)


def test_cu_seqlens():
    from experiment import qwen3vl_vision_runtime as vr
    got = vr.vision_cu_seqlens(np.array([[1,4,4],[2,2,2]], np.int32))
    assert np.array_equal(got, np.array([0,16,20,24], np.int32))


def _reference_bilinear_indices_and_weights(grid_thw, side, merge_size):
    idx_parts = [[] for _ in range(4)]
    weight_parts = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        h_grid = np.linspace(0, side - 1, int(h), dtype=np.float32)
        w_grid = np.linspace(0, side - 1, int(w), dtype=np.float32)
        h_floor = h_grid.astype(np.int32)
        w_floor = w_grid.astype(np.int32)
        h_ceil = np.minimum(h_floor + 1, side - 1)
        w_ceil = np.minimum(w_floor + 1, side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor
        hfo, hco = h_floor * side, h_ceil * side
        corners = [
            (hfo[:, None] + w_floor[None, :]).ravel(),
            (hfo[:, None] + w_ceil[None, :]).ravel(),
            (hco[:, None] + w_floor[None, :]).ravel(),
            (hco[:, None] + w_ceil[None, :]).ravel(),
        ]
        weights = [
            ((1-h_frac)[:,None]*(1-w_frac)[None,:]).ravel(),
            ((1-h_frac)[:,None]*w_frac[None,:]).ravel(),
            (h_frac[:,None]*(1-w_frac)[None,:]).ravel(),
            (h_frac[:,None]*w_frac[None,:]).ravel(),
        ]
        hi = np.arange(h).reshape(h // merge_size, merge_size)
        wi = np.arange(w).reshape(w // merge_size, merge_size)
        reorder = (hi[:, :, None, None] * w + wi[None, None, :, :]).transpose(0,2,1,3).ravel()
        reorder = np.tile(reorder, int(t))
        for i in range(4):
            idx_parts[i].append(corners[i][reorder])
            weight_parts[i].append(weights[i][reorder])
    return np.stack([np.concatenate(x) for x in idx_parts]), np.stack([np.concatenate(x) for x in weight_parts])


def test_bilinear_matches_transformers_reference_formula():
    from experiment import qwen3vl_vision_runtime as vr
    for grid in (
        np.array([[1,4,4]], np.int32),
        np.array([[1,6,8]], np.int32),
        np.array([[1,32,32]], np.int32),
    ):
        ri, rw = _reference_bilinear_indices_and_weights(grid, 48, 2)
        gi, gw = vr.vision_bilinear_indices_and_weights(grid, 48, 2)
        assert np.array_equal(gi, ri)
        assert np.allclose(gw, rw, atol=1e-5, rtol=0)
