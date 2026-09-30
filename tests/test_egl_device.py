"""MeshRenderer picks the EGL device of the GPU this process was assigned.

EGL enumerates physical devices and ignores CUDA_VISIBLE_DEVICES, so without
an explicit device_index every runner's GL context lands on physical GPU0 no
matter where Ray placed it (measured 2026-10-01: 8-GPU split, GPU0 9.8 GB vs
1.7 GB elsewhere). Pure-logic tests of the mapping; no GL context is created.
"""

from ngllib.simulator.render3d import MeshRenderer


def test_follows_first_cuda_visible_device(monkeypatch):
    monkeypatch.delenv("NGL_EGL_DEVICE", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert MeshRenderer._egl_device_kw() == {"device_index": 2}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")
    assert MeshRenderer._egl_device_kw() == {"device_index": 5}


def test_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("NGL_EGL_DEVICE", "7")
    assert MeshRenderer._egl_device_kw() == {"device_index": 7}


def test_falls_back_to_default_when_not_an_index(monkeypatch):
    monkeypatch.delenv("NGL_EGL_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert MeshRenderer._egl_device_kw() == {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert MeshRenderer._egl_device_kw() == {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0a1b2c3d")
    assert MeshRenderer._egl_device_kw() == {}
