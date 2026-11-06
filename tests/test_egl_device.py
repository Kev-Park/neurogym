from ngllib.simulator import egl_device
from ngllib.simulator.render3d import MeshRenderer
import pytest


def test_assignment_uses_driver_mapping_including_uuid(monkeypatch):
    monkeypatch.delenv("NGL_EGL_DEVICE", raising=False)
    seen = []
    monkeypatch.setattr(egl_device, "_assigned_egl_index", lambda value: seen.append(value) or 4)
    for value in ("2", "5,2", "GPU-abcd"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        assert MeshRenderer._egl_device_kw() == {"device_index": 4}
    assert seen == ["2", "5,2", "GPU-abcd"]


def test_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd")
    monkeypatch.setenv("NGL_EGL_DEVICE", "7")
    assert MeshRenderer._egl_device_kw() == {"device_index": 7}
    monkeypatch.setenv("NGL_EGL_DEVICE", "bad")
    with pytest.raises(ValueError):
        MeshRenderer._egl_device_kw()


def test_no_assignment_and_empty_assignment_are_distinct(monkeypatch):
    monkeypatch.delenv("NGL_EGL_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert MeshRenderer._egl_device_kw() == {}
    for value in ("", "-1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
        with pytest.raises(RuntimeError, match="assigns no GPU"):
            MeshRenderer._egl_device_kw()
