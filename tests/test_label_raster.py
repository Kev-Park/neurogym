"""CSS label samples survive capture downsampling without interpolating IDs."""
import numpy as np

from ngllib.simulator import em, pane2d


def test_thin_label_contributes_partial_pixel_coverage(monkeypatch):
    monkeypatch.setattr(pane2d, "draw_crosshair", lambda rgb: None)
    gray = np.zeros((433, 450), dtype=np.uint8)
    ids = np.zeros((867, 900), dtype=np.uint64)
    root = 720575940603464672
    ids[:, 100] = root  # one CSS pixel wide, half a capture pixel
    selected = pane2d.compose_left_parts(gray, ids, [str(root)])
    all_segments = pane2d.compose_left_parts(gray, ids, [])
    np.testing.assert_array_equal(selected, all_segments)
    assert selected[100, 50].max() > 0
    assert selected[100, 50].max() < 100  # averaged coverage, not opaque nearest-neighbour
    assert not selected[100, 49].any() and not selected[100, 51].any()
    assert ids[100, 100] == root


def test_both_worker_paths_request_css_labels(monkeypatch):
    class Tiles:
        def label_ids(self, pos, ex, ey, out_px):
            assert out_px == (900, 867)
            return np.full(out_px[::-1], 720575940603464672, dtype=np.uint64)

        def tile(self, *args):
            return np.zeros((8, 8), dtype=np.uint8)

    monkeypatch.setattr(em, "_worker_em", lambda source: Tiles())
    source = em.Source.calibrated()
    separate = em.unpack_ids(em.worker_ids(source, [0, 0, 0], 4))
    _, packed, _ = em.worker_pane_parts(source, [0, 0, 0], 4)
    np.testing.assert_array_equal(separate, em.unpack_ids(packed))
    assert separate.shape == (867, 900)
    assert separate[0, 0] == 720575940603464672
