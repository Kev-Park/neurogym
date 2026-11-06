"""Resolve CUDA assignments through EGL_NV_device_cuda, once at startup."""

from __future__ import annotations

import ctypes as C
from functools import lru_cache
import logging
import os

logger = logging.getLogger(__name__)


def _check_cuda(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {result}")


@lru_cache(maxsize=16)
def _assigned_egl_index(visible_devices: str) -> int:
    """Map process-visible CUDA device zero into EGL's device enumeration."""
    cuda = C.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [C.c_uint]
    cuda.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
    _check_cuda(cuda.cuInit(0), "cuInit")
    assigned = C.c_int()
    _check_cuda(cuda.cuDeviceGet(C.byref(assigned), 0), "cuDeviceGet(0)")
    egl = C.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [C.c_char_p]
    egl.eglGetProcAddress.restype = C.c_void_p

    def proc(name, result, *args):
        address = egl.eglGetProcAddress(name.encode())
        if not address:
            raise RuntimeError(f"EGL does not provide {name}")
        return C.CFUNCTYPE(result, *args)(address)

    query_devices = proc("eglQueryDevicesEXT", C.c_uint, C.c_int,
                         C.POINTER(C.c_void_p), C.POINTER(C.c_int))
    query_string = proc("eglQueryDeviceStringEXT", C.c_char_p, C.c_void_p, C.c_int)
    query_attribute = proc("eglQueryDeviceAttribEXT", C.c_uint, C.c_void_p,
                           C.c_int, C.POINTER(C.c_ssize_t))
    count = C.c_int()
    if not query_devices(0, None, C.byref(count)) or count.value < 1:
        raise RuntimeError("EGL reported no devices")
    devices = (C.c_void_p * count.value)()
    if not query_devices(count.value, devices, C.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    for index, device in enumerate(devices[:count.value]):
        extensions = (query_string(device, 0x3055) or b"").split()  # EGL_EXTENSIONS
        if b"EGL_NV_device_cuda" not in extensions:
            continue
        cuda_device = C.c_ssize_t()
        if (query_attribute(device, 0x323A, C.byref(cuda_device))  # EGL_CUDA_DEVICE_NV
                and cuda_device.value == assigned.value):
            logger.info("CUDA_VISIBLE_DEVICES=%s: CUDA device %s maps to EGL device %s",
                        visible_devices, assigned.value, index)
            return index
    raise RuntimeError(
        f"No EGL device maps to assigned CUDA device for CUDA_VISIBLE_DEVICES={visible_devices!r}; "
        "check EGL_NV_device_cuda support or set NGL_EGL_DEVICE explicitly")


def device_kwargs() -> dict:
    override = os.environ.get("NGL_EGL_DEVICE")
    if override is not None:
        if not override.isdigit():
            raise ValueError("NGL_EGL_DEVICE must be a nonnegative EGL device index")
        return {"device_index": int(override)}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return {}  # No assignment: preserve the platform's default EGL choice.
    if not visible.strip() or visible.strip() == "-1":
        raise RuntimeError("GPU rendering requested but CUDA_VISIBLE_DEVICES assigns no GPU")
    return {"device_index": _assigned_egl_index(visible)}
