"""CloudVolume access for the native renderer: EM z-slice tiles + label masks.

Promoted from the parity spike harness (neurogym-agent native/render_pairs.py)
after the 2D-pane SSIM 0.898 / pixel-exact-registration result — the tile and
filter chains here are calibrated against browser captures; change them only
with a re-run of the parity harness.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

EM_URL = "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"
SEG_URL = "precomputed://gs://flywire_v141_m783"


class EMTiles:
    """z-slice tiles around a position from the public FlyWire EM volume.

    Real data starts at mip1 (8x8x40nm; mip0 is a placeholder). The mip is
    chosen per request so the tile stays <= max_px, mirroring NG's use of
    coarse mips for the 3D section plane.
    """

    RES_XY = [8, 16, 32, 64, 128]  # nm, mips 1..5

    # In-RAM chunk LRU per volume handle. Chrome keeps the episode's chunk
    # working set hot in memory; without this every revisit pays disk-cache
    # decode (or worse, network), and the synchronous step blocks on it.
    LRU_BYTES = 512 << 20

    def __init__(self, cache_dir: str):
        from cloudvolume import CloudVolume

        self._vols: dict = {}
        self._CloudVolume = CloudVolume
        self._cache = cache_dir

    def _open(self, url: str, **kw):
        base = dict(use_https=True, cache=self._cache, progress=False,
                    fill_missing=True, bounded=False)
        try:
            return self._CloudVolume(url, lru_bytes=self.LRU_BYTES,
                                     **base, **kw)
        except TypeError:  # older cloud-volume without lru_bytes
            return self._CloudVolume(url, **base, **kw)

    def _vol(self, mip: int):
        if mip not in self._vols:
            self._vols[mip] = self._open(EM_URL, mip=mip)
        return self._vols[mip]

    def _seg_vol(self, target_res_nm: float):
        """Seg volume at the coarsest mip still at/below target_res_nm per
        px — fetching labels at native resolution over-fetches ~64x in area
        for typical pane extents."""
        if "seg_scales" not in self._vols:
            try:
                base = self._open(SEG_URL, agglomerate=False)
                self._vols["seg_scales"] = [
                    s["resolution"][0] for s in base.info["scales"]]
                self._vols[("seg", 0)] = base
            except Exception:
                self._vols["seg_scales"] = None
        scales = self._vols["seg_scales"]
        if scales is None:
            return None
        mip = 0
        for i, r in enumerate(scales):
            if r <= target_res_nm:
                mip = i
            else:
                break
        if ("seg", mip) not in self._vols:
            self._vols[("seg", mip)] = self._open(
                SEG_URL, agglomerate=False, mip=mip)
        return self._vols[("seg", mip)]

    def label_tile(self, pos_nm, extent_x_nm, extent_y_nm, root_id,
                   out_px=(450, 433)):
        """Boolean mask of the root segment on the z-slice, or None if the
        static label chunks aren't readable from the m783 bucket."""
        try:
            vol = self._seg_vol(extent_x_nm / out_px[0])
            if vol is None:
                return None
            res = vol.resolution  # nm per voxel
            cx = int(pos_nm[0] / res[0]); cy = int(pos_nm[1] / res[1])
            z = int(pos_nm[2] / res[2])
            hx = int(extent_x_nm / res[0] / 2); hy = int(extent_y_nm / res[1] / 2)
            if hx < 1 or hy < 1:
                return None
            cut = vol[cx - hx:cx + hx, cy - hy:cy + hy, z:z + 1]
            lab = np.asarray(cut)[:, :, 0, 0].T
            mask = (lab == int(root_id)).astype(np.uint8) * 255
            m = Image.fromarray(mask).resize(out_px, Image.NEAREST)
            return np.asarray(m) > 127
        except Exception:
            return None

    def tile(self, pos_nm, extent_x_nm, extent_y_nm=None, max_px=1024,
             subpixel: bool = False):
        """EM z-slice tile. With subpixel=True, the fractional texel phase
        of the requested center is preserved via an affine resample (integer
        snapping costs up to half a texel — visible at membrane scale)."""
        if extent_y_nm is None:
            extent_y_nm = extent_x_nm
        mip = 1
        for i, r in enumerate(self.RES_XY):
            mip = i + 1
            if max(extent_x_nm, extent_y_nm) / r <= max_px:
                break
        res = self.RES_XY[mip - 1]
        vol = self._vol(mip)
        fx, fy = pos_nm[0] / res, pos_nm[1] / res
        cx, cy = int(fx), int(fy)
        z = int(pos_nm[2] / 40.0)
        hx = int(extent_x_nm / res / 2)
        hy = int(extent_y_nm / res / 2)
        pad = 2 if subpixel else 0
        cut = vol[cx - hx - pad:cx + hx + pad,
                  cy - hy - pad:cy + hy + pad, z:z + 1]
        img = np.asarray(cut)[:, :, 0, 0].T.astype(np.uint8)  # row=y, col=x
        if subpixel:
            dx, dy = fx - cx, fy - cy
            im = Image.fromarray(img).transform(
                (img.shape[1], img.shape[0]), Image.AFFINE,
                (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)
            img = np.asarray(im)[pad:-pad or None, pad:-pad or None]
        return img


class MeshStore:
    """Sharded-Draco mesh fetch (CloudVolume) with a decoded-mesh cache."""

    def __init__(self, cache_dir: str):
        from cloudvolume import CloudVolume

        self._vol = CloudVolume(SEG_URL, use_https=True, cache=cache_dir,
                                progress=False)
        self._meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, root_id: str) -> tuple[np.ndarray, np.ndarray]:
        """(vertices_nm float32 [N,3], faces int32 [M,3])."""
        if root_id not in self._meshes:
            m = self._vol.mesh.get(int(root_id))
            mesh = m[int(root_id)] if hasattr(m, "get") or isinstance(m, dict) else m
            self._meshes[root_id] = (
                np.asarray(mesh.vertices, dtype="f4"),
                np.asarray(mesh.faces, dtype="i4"),
            )
        return self._meshes[root_id]

    def drop(self, root_id: str) -> None:
        self._meshes.pop(root_id, None)
