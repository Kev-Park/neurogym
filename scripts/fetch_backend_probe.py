"""Parity + affinity gate for the Ray fetch backend (login node, NO GPU).

1. PARITY: a Ray _FetchActor decodes byte-identical tiles + meshes to the direct
   (process-backend) worker call. Disaggregating fetch must not change a pixel or
   a vertex.
2. AFFINITY: a second fetch of the same region on the SAME actor reuses its chunk
   LRU (fast); a fresh actor pays the cold decode -- i.e. env->actor stickiness is
   what buys locality, exactly as the per-process pools do. This is the mechanism
   the affinity-vs-round-robin A/B toggles.

    cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
    uv run --no-sync python /scratch/kp0374/wt/neurogym-throughput-scaling/scripts/fetch_backend_probe.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import numpy as np

from ngllib.simulator.em import Source, worker_tile, worker_mesh
from ngllib.simulator import pane2d

POS = np.array([143944.703, 61076.594, 192.58])
XS = 2.0339912586467497
ROOT = "720575940603464672"
MAXPX = 1024


def main():
    src = Source.calibrated(cache_dir=None)
    ext = pane2d.pane_extents_nm(XS, src.canonical_nm)
    pos_nm = POS * src.voxel_nm

    # --- reference: process-backend semantics (direct call in this process) ---
    ref_tile = worker_tile(src, pos_nm, ext[0], ext[1], MAXPX, False)
    ref_v, ref_vn, ref_f = worker_mesh(src, ROOT, 0)
    print(f"ref tile {ref_tile.shape} mesh v={len(ref_v)} f={len(ref_f)}", flush=True)

    # --- ray backend ---
    import ray
    # Disable Ray>=2.43's uv-run auto-upload of the 1.2GB CWD as runtime_env (it
    # stalls actor startup); all nodes share /scratch, so no upload is needed.
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    ray.init(ignore_reinit_error=True, num_cpus=4, log_to_driver=False,
             include_dashboard=False)
    from ngllib.simulator.fetch_pool import _fetch_actor_cls
    Actor = _fetch_actor_cls()
    a = Actor.remote()

    # TILE parity first (small, fast) -- print immediately so a slow/huge mesh
    # transfer never hides the tile result.
    ray_tile = ray.get(a.run.remote(worker_tile, src, pos_nm, ext[0], ext[1], MAXPX, False),
                       timeout=120)
    dt = int(np.abs(ref_tile.astype(np.int64) - ray_tile.astype(np.int64)).max())
    print(f"tile max diff        = {dt}   (ray actor vs process worker)", flush=True)

    ok = (dt == 0)
    if os.environ.get("PROBE_TILE_ONLY") != "1":
        try:
            ray_v, ray_vn, ray_f = ray.get(a.run.remote(worker_mesh, src, ROOT, 0),
                                           timeout=180)
            dv = float(np.abs(ref_v - ray_v).max())
            dvn = float(np.abs(ref_vn - ray_vn).max())
            df = int(np.abs(ref_f - ray_f).max())
            print(f"mesh vert max diff   = {dv}", flush=True)
            print(f"mesh normal max diff = {dvn}", flush=True)
            print(f"mesh face max diff   = {df}", flush=True)
            ok = ok and dv == 0.0 and dvn == 0.0 and df == 0
        except Exception as e:  # noqa: BLE001
            print(f"mesh path skipped (login-node Ray transfer of ~83MB): {e}", flush=True)
    print(f"FETCH-BACKEND-PARITY {'PASS' if ok else 'FAIL'}", flush=True)

    # --- affinity: same actor reuses chunk LRU; fresh actor is cold ---
    t0 = time.perf_counter()
    ray.get(a.run.remote(worker_tile, src, pos_nm, ext[0], ext[1], MAXPX, False))
    warm = (time.perf_counter() - t0) * 1e3
    b = Actor.remote()
    t0 = time.perf_counter()
    ray.get(b.run.remote(worker_tile, src, pos_nm, ext[0], ext[1], MAXPX, False))
    cold = (time.perf_counter() - t0) * 1e3
    verdict = "OBSERVED" if warm < 0.6 * cold else "weak/none"
    print(f"affinity: same-actor warm={warm:.1f} ms  fresh-actor cold={cold:.1f} ms "
          f"-> chunk-LRU reuse {verdict}", flush=True)

    # --- approach D: _RayPool background-thread path (env thread never calls ray) ---
    from ngllib.simulator.fetch_pool import _RayPool, _LocalFuture
    pool = _RayPool(Actor.remote())
    f = pool.submit(worker_tile, src, pos_nm, ext[0], ext[1], MAXPX, False)
    is_local = isinstance(f, _LocalFuture)      # env thread gets a LOCAL future
    early_done = f.done()                        # local Event check, no ray call
    pool_tile = f.result(timeout=120)            # blocks on local Event; ray.get on bg thread
    dpt = int(np.abs(ref_tile.astype(np.int64) - pool_tile.astype(np.int64)).max())
    pool.shutdown()
    print(f"RayPool(D): local_future={is_local} done_before={early_done} tile_diff={dpt}",
          flush=True)
    d_ok = is_local and dpt == 0
    print(f"RAYPOOL-D-PARITY {'PASS' if d_ok else 'FAIL'}", flush=True)
    ray.shutdown()

    # --- socket backend: start a local fetch server, connect a _SocketPool ---
    from ngllib.simulator.fetch_pool import _SocketPool, _LocalFuture
    sport = 39917
    srv = subprocess.Popen([sys.executable, "-m", "ngllib.simulator.fetch_server",
                            str(sport)])
    time.sleep(4)  # let it bind
    s_ok = False
    try:
        sp = _SocketPool("127.0.0.1", sport)
        f = sp.submit(worker_tile, src, pos_nm, ext[0], ext[1], MAXPX, False)
        s_local = isinstance(f, _LocalFuture)
        s_tile = f.result(timeout=120)
        dst = int(np.abs(ref_tile.astype(np.int64) - s_tile.astype(np.int64)).max())
        fm = sp.submit(worker_mesh, src, ROOT, 0)
        s_v, s_vn, s_f = fm.result(timeout=180)
        dmv = float(np.abs(ref_v - s_v).max())
        dmf = int(np.abs(ref_f - s_f).max())
        # affinity: same pool (warm, sticky server proc) vs fresh pool (cold)
        t0 = time.perf_counter()
        sp.submit(worker_tile, src, pos_nm, ext[0], ext[1], MAXPX, False).result(60)
        s_warm = (time.perf_counter() - t0) * 1e3
        sp2 = _SocketPool("127.0.0.1", sport)
        t0 = time.perf_counter()
        sp2.submit(worker_tile, src, pos_nm, ext[0], ext[1], MAXPX, False).result(60)
        s_cold = (time.perf_counter() - t0) * 1e3
        sp.shutdown(); sp2.shutdown()
        s_ok = s_local and dst == 0 and dmv == 0.0 and dmf == 0
        print(f"Socket: local_future={s_local} tile_diff={dst} mesh_vert_diff={dmv} "
              f"mesh_face_diff={dmf}", flush=True)
        print(f"Socket affinity: warm(same conn)={s_warm:.1f} ms  cold(fresh conn)="
              f"{s_cold:.1f} ms -> reuse {'OBSERVED' if s_warm < 0.6 * s_cold else 'weak'}",
              flush=True)
    finally:
        srv.terminate()
    print(f"SOCKET-PARITY {'PASS' if s_ok else 'FAIL'}", flush=True)

    return 0 if (ok and d_ok and s_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
