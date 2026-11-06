"""GPU-job validation: assignments, rendered pixels/depth, and mapping overhead."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np


def child(expected):
    from ngllib.simulator.render3d import MeshRenderer

    start = time.perf_counter()
    selected = MeshRenderer._egl_device_kw()["device_index"]
    cold = time.perf_counter() - start
    assert selected == expected, (selected, expected)
    start = time.perf_counter()
    for _ in range(1000):
        assert MeshRenderer._egl_device_kw()["device_index"] == expected
    cached = (time.perf_counter() - start) / 1000
    assert cached < .001

    def render():
        renderer = MeshRenderer(160, 144)
        try:
            vertices = np.array([[-500, -500, 0], [500, -500, 0], [0, 500, 0]], dtype="f4")
            renderer.load_mesh("7", vertices, np.array([[0, 1, 2]], dtype="u4"))
            args = ("7", [0, 0, 0], [0, 0, 0, 1], 2000)
            image = renderer.render(*args, color=[1, .3, .2])
            depth, _, _ = renderer.pick_depth(*args)
            assert np.any(image) and np.any(depth < 1)
            return hashlib.sha256(image.tobytes() + depth.tobytes()).hexdigest()
        finally:
            renderer.close()

    auto = render()
    os.environ["NGL_EGL_DEVICE"] = str(expected)
    explicit = render()
    assert auto == explicit, (auto, explicit)
    print(json.dumps(dict(assignment=os.environ["CUDA_VISIBLE_DEVICES"],
                         egl_index=selected, cold_s=cold, cached_s=cached,
                         pixel_depth_sha256=auto)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    if args.child is not None:
        child(args.child)
        return
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    assert len(uuids) >= 3, "Allocate the full GPU node for this probe"
    cases = [("0", 0), ("2", 2), ("2,0", 2), (uuids[2], 2)]
    for assignment, expected in cases:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": assignment}
        env.pop("NGL_EGL_DEVICE", None)
        subprocess.run([sys.executable, __file__, "--child", str(expected)],
                       env=env, check=True, timeout=90)
    print("EGL-MAPPING-PASS", flush=True)


if __name__ == "__main__":
    main()
