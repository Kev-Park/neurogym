"""Per-stage CPU breakdown of the simulator's fetch/decode, on the CALIBRATED
FlyWire dataset (public S3 EM + GCS seg). Pure CPU + network -- NO GPU, so it
runs on the login node or a CPU-only SLURM job, sidestepping the GPU queue.

Answers what the ~65%-of-node fetch/decode CPU (job 958888) is actually spent
on, and specifically whether there is a *resample* worth moving to the GPU for
the 3D-pane-only path (where worker_tile uses em.tile(subpixel=False) -- a raw
cutout, no PIL resample -- and worker_mesh does CloudVolume Draco + normals).

    cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
    uv run --no-sync python /scratch/kp0374/wt/neurogym-throughput-scaling/scripts/fetch_cpu_breakdown.py
"""
from __future__ import annotations

import time
import numpy as np

from ngllib.simulator.em import Source, EMTiles, MeshStore
from ngllib.simulator import pane2d

# From config.json's calibrated start URL (public FlyWire fafbv14 EM + v141_m783 seg).
POS_VOXEL = np.array([143944.703, 61076.594, 192.58])   # x,y,z voxels
XS = 2.0339912586467497                                  # crossSectionScale
ROOT_ID = "720575940603464672"                           # a visible segment there
MAX_PX = 1024


def _t(fn, n=5):
    """Median + min ms over n calls (min ~= warm/cache-hit, median ~= typical)."""
    ts = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts)), out


def main():
    src = Source.calibrated(cache_dir=None)
    em = EMTiles(src)
    store = MeshStore(src)
    vx = src.voxel_nm
    ext = pane2d.pane_extents_nm(XS, src.canonical_nm)
    print(f"voxel_nm={tuple(vx)} canonical={src.canonical_nm} ext_nm={ext}", flush=True)

    def tile_plane(shift=0.0):
        p = (POS_VOXEL + np.array([shift, shift, 0.0])) * vx
        return em.tile(p, ext[0], ext[1], MAX_PX, False)      # 3D plane: raw cutout

    def tile_sub(shift=0.0):
        p = (POS_VOXEL + np.array([shift, shift, 0.0])) * vx
        return em.tile(p, ext[0], ext[1], MAX_PX, True)       # subpixel affine (left/preview)

    def label(shift=0.0):
        p = (POS_VOXEL + np.array([shift, shift, 0.0])) * vx
        return em.label_ids(p, ext[0], ext[1], (pane2d.PANE, pane2d.PANE_H))

    # --- COLD (first touch, each a fresh position so no chunk reuse) ---
    print("\n=== COLD (fresh position each call: network + decompress) ===", flush=True)
    for name, fn in [("em.tile plane (subpixel=False)", tile_plane),
                     ("em.tile subpixel=True", tile_sub),
                     ("em.label_ids (seg id map)", label)]:
        ts = []
        out = None
        for i in range(4):
            t0 = time.perf_counter()
            out = fn(shift=100.0 * (i + 1))   # move ~100 vox each time -> new chunks
            ts.append((time.perf_counter() - t0) * 1e3)
        nb = getattr(out, "nbytes", None)
        print(f"  {name:34s} cold_med={np.median(ts):8.1f} ms  min={np.min(ts):8.1f}  "
              f"shape={getattr(out,'shape',None)} bytes={nb}", flush=True)

    # --- WARM (same position repeated: chunk LRU hit -> pure CPU decode/resample) ---
    print("\n=== WARM (same position: chunk-cache hit -> the CPU-only cost) ===", flush=True)
    med, mn, tile = _t(lambda: tile_plane(0.0))
    print(f"  em.tile plane warm            med={med:8.2f} ms  min={mn:8.2f}  "
          f"shape={tile.shape} bytes={tile.nbytes}", flush=True)
    med, mn, tsub = _t(lambda: tile_sub(0.0))
    print(f"  em.tile subpixel warm         med={med:8.2f} ms  min={mn:8.2f}  "
          f"shape={tsub.shape} bytes={tsub.nbytes}", flush=True)
    med, mn, _ = _t(lambda: pane2d.resample_em(tsub))
    print(f"  pane2d.resample_em (PIL)      med={med:8.2f} ms  min={mn:8.2f}  "
          f"<- the CPU RESAMPLE (GPU-offload candidate)", flush=True)

    # --- MESH (Draco decode + normal recompute) ---
    print("\n=== MESH worker_mesh stages (per new selection, not per step) ===", flush=True)
    t0 = time.perf_counter()
    v, f = store.get(ROOT_ID, 0)
    t_get = (time.perf_counter() - t0) * 1e3
    print(f"  store.get lod0 (download+Draco)  {t_get:8.1f} ms  verts={len(v)} faces={len(f)} "
          f"bytes={v.nbytes + f.nbytes}", flush=True)
    # normals: the cross + bincount scatter-add (worker_mesh tail), timed warm.
    def normals():
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        fn = np.cross(e1, e2)
        vn = np.zeros_like(v)
        for k in range(3):
            for c in range(3):
                vn[:, c] += np.bincount(f[:, k], fn[:, c], minlength=len(v))
        return vn
    med, mn, _ = _t(normals, n=3)
    print(f"  normal recompute (cross+bincount) med={med:8.1f} ms  min={mn:8.1f}", flush=True)

    print("\n[breakdown] done", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
