"""moderngl/EGL renderer for the 3D-projection pane.

Promoted from the parity spike harness after 3D-pane block-SSIM 0.845 median
against browser captures. All lighting/overlay constants are NG-source-exact
(google/neuroglancer master, retrieved 2026-08-27):
- mesh/frontend.ts + perspective_view/panel.ts: Gouraud lighting
  factor = |dot(n, l)| * 0.8 + 0.2, light = -(R(q) @ z) (headlight).
- axes_lines.ts + panel.ts: axis lines through position, pure R/G/B
  alpha 0.5, 1px, half-length = zoom * min(w,h)/h / 4, ONE-SIDED toward
  +axis (browser-observed).
- panel.ts drawSliceViews: the section plane is the CROSS-SECTION
  VIEWPORT's rect (xs_scale x 900 CSS px x 4nm wide, x 867 tall), centered
  at position in the z-plane, EM-textured, opaque, lit by 0.2 + |l_z|*0.8.

VRAM discipline (native-renderer design constraint): meshes are indexed
buffers (~3x smaller than tri-soup) held under an LRU byte budget — GPU
memory is statically bounded, no growth-until-restart.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from .camera import projection_camera


class MeshRenderer:
    """Offscreen renderer for one (width, height) pane geometry.

    `width`/`height` are the pane's true captured pixels BELOW the toolbar
    (450 x 433 at capture scale 0.5) so the optical center and aspect match
    the browser.
    """

    VAO_BUDGET_BYTES = 2 << 30

    @classmethod
    def vao_budget_bytes(cls, requested: int | None = None) -> int:
        """GPU-resident mesh budget for THIS process (NGL_NATIVE_VAO_LRU_MB).

        One MeshRenderer per process, so the card carries processes x this.
        The 2 GB default was sized for a lone renderer: at 24-32 runners per
        3090 it is a 48-64 GB claim on a 24 GB card, and under random actions
        (a new neuron every 300 steps, ~10 MB of VAO each) the caches filled
        at ~1.2 GB/process/hour until the card was full (2026-09-12).
        Production 32x1 survived only because a trained policy turns neurons
        over far more slowly. Size it as (VRAM_MB - baseline) / processes_per_GPU;
        ~200 MB still holds dozens of meshes for an env that shows one at a time.
        """
        if requested is not None:
            return int(requested)
        mb = os.environ.get("NGL_NATIVE_VAO_LRU_MB")
        return (int(mb) << 20) if mb else cls.VAO_BUDGET_BYTES

    def __init__(self, width: int, height: int,
                 mesh_budget_bytes: int | None = None,
                 cuda_ipc: bool = False):
        import moderngl

        self._moderngl = moderngl
        self.width, self.height = width, height
        self._cuda_ipc = bool(cuda_ipc)
        if self._cuda_ipc:
            # In the RLlib/Ray runner the default EGL device is NOT necessarily
            # the physical GPU torch's CUDA context lives on (all 8 are 3090s;
            # Ray re-indexes CUDA_VISIBLE_DEVICES), and GL<->CUDA interop then
            # fails with cudaErrorInvalidGraphicsContext(208). Pick the EGL
            # device_index whose interop actually works.
            self.ctx = self._create_interop_context(moderngl, width, height)
        else:
            self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._color = self.ctx.texture((width, height), 4)
        self._depth = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self._color], depth_attachment=self._depth)
        self.prog = self.ctx.program(
            vertex_shader="""#version 330
                uniform mat4 mvp;
                uniform vec4 light;   // xyz dir (pre-scaled 0.8), w ambient
                in vec3 pos; in vec3 nrm;
                out float v_l;
                void main() {
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_l = abs(dot(normalize(nrm), light.xyz)) + light.w;
                }""",
            fragment_shader="""#version 330
                uniform vec3 color;
                in float v_l;
                out vec4 frag;
                void main() { frag = vec4(color * v_l, 1.0); }""",
        )
        self.line_prog = self.ctx.program(
            vertex_shader="""#version 330
                uniform mat4 mvp;
                in vec3 pos; in vec4 col;
                out vec4 v_c;
                void main() {
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_c = col;
                }""",
            fragment_shader="""#version 330
                in vec4 v_c; out vec4 frag;
                void main() { frag = v_c; }""",
        )
        self.plane_prog = self.ctx.program(
            vertex_shader="""#version 330
                uniform mat4 mvp;
                in vec3 pos; in vec2 uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_uv = uv;
                }""",
            fragment_shader="""#version 330
                uniform sampler2D em;
                uniform float lfac;
                in vec2 v_uv; out vec4 frag;
                void main() {
                    // RGB: Neuroglancer draws the SAME segmentation layer on
                    // the perspective view's cross-section, so the plane is
                    // tinted, not grey. A greyscale plane is uploaded with
                    // its three channels equal, so this covers both.
                    frag = vec4(texture(em, v_uv).rgb * lfac, 1.0);
                }""",
        )
        # LRU mesh VAOs: root_id -> (vao, [vbo, ibo], bytes)
        self._vaos: OrderedDict[str, tuple] = OrderedDict()
        # Per-frame geometry lives in FIXED buffers written in place. The
        # plane quad, the axis lines and the EM texture used to be created and
        # released on every step; the driver does not hand released VRAM back
        # to the process (measured 2026-09-12: one env, torch allocator flat at
        # 114 MiB and the mesh LRU bounded at 200 MiB, yet the process grew
        # ~4.5 GB/h of GL memory -- alloc/free churn fragmenting the pool).
        # A buffer that is only ever rewritten cannot fragment anything.
        self._quad_vbo = self.ctx.buffer(reserve=4 * 5 * 4)
        self._quad_vao = self.ctx.vertex_array(
            self.plane_prog, [(self._quad_vbo, "3f 2f", "pos", "uv")])
        self._line_vbo = self.ctx.buffer(reserve=6 * 7 * 4)
        self._line_vao = self.ctx.vertex_array(
            self.line_prog, [(self._line_vbo, "3f 4f", "pos", "col")])
        # EM plane textures, one per tile SHAPE ever seen (fine tile, coarse
        # tile, pick_depth's 2x2 dummy -- a handful per run), each rewritten.
        self._plane_tex: dict[tuple[int, int], object] = {}
        self._budget = self.vao_budget_bytes(mesh_budget_bytes)
        # Mesh storage is a POOL OF SLOTS, not one allocation per mesh. A slot
        # is a VBO + IBO + VAO created on first use and grown (orphaned) only
        # when a mesh exceeds its capacity, so capacities are monotone and GL
        # allocation events stay O(slots + growths) instead of O(episodes).
        # The per-mesh alloc/release LRU this replaces leaked ~half of every
        # evicted mesh to driver-side fragmentation (2026-09-12, see the
        # per-frame note above); eviction here is a dict pop, no GL call.
        # The byte budget sets the slot COUNT. Capacity floats with the
        # largest mesh each slot has held (~95 MB average after 12 min of
        # random neurons, 2026-09-12), so the count is what bounds VRAM:
        # a trained policy shows one neuron at a time and needs 2-3 slots;
        # evicting a still-selected mesh only costs an async refetch.
        self._n_slots = max(2, min(64, self._budget // self.SLOT_NOMINAL_BYTES))
        self._slots: list[dict] = []          # created lazily, index = slot id
        # root_id -> (slot id, index count); insertion order is LRU order.
        self._vaos: OrderedDict[str, tuple[int, int]] = OrderedDict()

    SLOT_NOMINAL_BYTES = 32 << 20   # budget / this = slot count

    @property
    def _vao_bytes(self) -> int:
        """Bytes of GL storage the slots currently hold (capacity, not use)."""
        return sum(sl["vcap"] + sl["icap"] for sl in self._slots)

    def has_mesh(self, root_id: str) -> bool:
        return root_id in self._vaos

    def _acquire_slot(self) -> int:
        if len(self._slots) < self._n_slots:
            vbo = self.ctx.buffer(reserve=1024)
            ibo = self.ctx.buffer(reserve=1024)
            vao = self.ctx.vertex_array(
                self.prog, [(vbo, "3f 3f", "pos", "nrm")], index_buffer=ibo)
            self._slots.append({"vbo": vbo, "ibo": ibo, "vao": vao,
                                "vcap": 1024, "icap": 1024})
            return len(self._slots) - 1
        # Every slot holds a mesh: evict the least recently drawn.
        _, (sid, _) = self._vaos.popitem(last=False)
        return sid

    def _fill_slot(self, sid: int, vdata: bytes, idata: bytes) -> None:
        sl = self._slots[sid]
        if len(vdata) > sl["vcap"]:
            sl["vbo"].orphan(len(vdata)); sl["vcap"] = len(vdata)
        if len(idata) > sl["icap"]:
            sl["ibo"].orphan(len(idata)); sl["icap"] = len(idata)
        sl["vbo"].write(vdata)
        sl["ibo"].write(idata)

    def load_mesh(self, root_id: str, vertices_nm, faces,
                  normals=None, replace: bool = False) -> None:
        """Indexed draw with smooth per-vertex normals (precomputed via
        `normals`, e.g. by em.worker_mesh, or derived here); LRU-evicts
        past budget.

        `replace=True` swaps an already-resident mesh, which is how the
        progressive path refines a coarse level once the fine one lands.
        """
        sid = None
        if root_id in self._vaos:
            if not replace:
                self._vaos.move_to_end(root_id)
                return
            sid, _ = self._vaos.pop(root_id)   # refine in place, same slot
        v = np.asarray(vertices_nm, dtype="f4")
        f = np.asarray(faces, dtype="i4")
        if normals is not None:
            vn = np.asarray(normals, dtype="f4")
        else:
            e1 = v[f[:, 1]] - v[f[:, 0]]
            e2 = v[f[:, 2]] - v[f[:, 0]]
            fn = np.cross(e1, e2)
            vn = np.zeros_like(v)
            for k in range(3):
                np.add.at(vn, f[:, k], fn)
            vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
        if sid is None:
            sid = self._acquire_slot()
        self._fill_slot(sid, np.hstack([v, vn.astype("f4")]).tobytes(),
                        f.tobytes())
        self._vaos[root_id] = (sid, int(f.size))

    @staticmethod
    def _as_ids(root_id):
        """`select` toggles a SET of segments, so every draw path takes either
        a single id (legacy callers) or a sequence of them."""
        if isinstance(root_id, (str, int)):
            return [str(root_id)]
        return [str(r) for r in root_id]

    def _draw_meshes(self, ids, colors) -> None:
        """One draw per loaded segment, each in its own colour. Segments whose
        mesh has not arrived yet are simply skipped -- the pane shows what is
        loaded rather than raising mid-frame."""
        for i, rid in enumerate(ids):
            entry = self._vaos.get(rid)
            if entry is None:
                continue
            sid, n_idx = entry
            self.prog["color"].value = tuple(float(c) for c in colors[i])
            self._slots[sid]["vao"].render(mode=4, vertices=n_idx)
            self._vaos.move_to_end(rid)

    @staticmethod
    def _rot(q):
        x, y, z, w = q
        n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def _matrices(self, position_nm, quat, zoom_nm):
        view, proj = projection_camera(
            position_nm, quat, zoom_nm, self.width, self.height)
        return view, proj

    def _draw_plane(self, mvp_b, pos, em_tile, em_extent_nm, lfac):
        arr = np.ascontiguousarray(em_tile)
        if arr.ndim != 3:
            # Greyscale: replicate so one shader path serves both.
            arr = np.ascontiguousarray(np.repeat(arr[..., None], 3, axis=2))
        shape = (arr.shape[1], arr.shape[0])
        tex = self._plane_tex.get(shape)
        if tex is None:
            tex = self._plane_tex[shape] = self.ctx.texture(shape, 3)
        tex.write(arr.tobytes())
        tex.use(0)
        hx, hy = em_extent_nm[0] / 2.0, em_extent_nm[1] / 2.0
        quad = np.array([
            pos[0] - hx, pos[1] - hy, pos[2], 0, 0,
            pos[0] + hx, pos[1] - hy, pos[2], 1, 0,
            pos[0] - hx, pos[1] + hy, pos[2], 0, 1,
            pos[0] + hx, pos[1] + hy, pos[2], 1, 1,
        ], dtype="f4")
        self._quad_vbo.write(quad.tobytes())
        self.plane_prog["mvp"].write(mvp_b)
        self.plane_prog["lfac"].value = float(lfac)
        self._quad_vao.render(mode=5)  # TRIANGLE_STRIP

    def _create_interop_context(self, moderngl, width, height):
        """Return a moderngl EGL context on an EGL device whose GL<->CUDA interop
        works with torch's CUDA device (probes device_index candidates)."""
        import logging

        import torch
        from cuda.bindings import runtime as rt

        torch.cuda.init()
        rt.cudaSetDevice(torch.cuda.current_device())
        RO = rt.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
        log = logging.getLogger("ngllib.simulator.render3d")
        last = None
        for idx in [None, 0, 1, 2, 3, 4, 5, 6, 7]:  # None = moderngl's default device
            try:
                kw = {} if idx is None else {"device_index": idx}
                ctx = moderngl.create_context(standalone=True, backend="egl", **kw)
            except Exception as ex:
                last = f"create idx={idx}: {ex}"
                continue
            try:
                tex = ctx.texture((8, 8), 4)
                e, res = rt.cudaGraphicsGLRegisterImage(tex.glo, 0x0DE1, RO)
                if int(e) != 0:
                    raise RuntimeError(f"register {int(e)}")
                em = rt.cudaGraphicsMapResources(1, res, 0)[0]
                if int(em) != 0:
                    raise RuntimeError(f"map {int(em)}")
                rt.cudaGraphicsUnmapResources(1, res, 0)
                rt.cudaGraphicsUnregisterResource(res)
                tex.release()
                log.info("cuda_ipc: EGL device_index=%r gives working GL-CUDA "
                         "interop (gl=%s)", idx, ctx.info.get("GL_RENDERER", "?"))
                return ctx
            except Exception as ex:
                last = f"interop idx={idx}: {ex}"
                try:
                    ctx.release()
                except Exception:
                    pass
        raise RuntimeError(
            f"cuda_ipc: no EGL device_index gave working GL-CUDA interop; last={last}")

    def _ensure_cuda(self):
        """Lazily set up GL->CUDA interop for the color attachment (dino-server
        CUDA-IPC path). Registers self._color once and allocates a persistent
        torch CUDA tensor the mapped array is copied into each render. Imports
        torch + cuda-python only here, so non-IPC runs never pull them in."""
        if getattr(self, "_cuda_res", None) is not None:
            return
        import logging
        import os

        import torch
        from cuda.bindings import runtime as rt

        self._torch = torch
        self._rt = rt
        torch.cuda.init()
        # Pin cuda-python's runtime context to torch's CUDA device so the GL-CUDA
        # interop register/map run on the SAME context torch owns the dst tensor
        # in (the map returned cudaErrorInvalidGraphicsContext(208) in the RLlib
        # runner where they diverged; harmless when already aligned).
        dev = torch.cuda.current_device()
        rt.cudaSetDevice(dev)
        try:
            pci = torch.cuda.get_device_properties(dev).pci_bus_id
        except Exception:
            pci = "?"
        logging.getLogger("ngllib.simulator.render3d").info(
            "cuda_ipc setup: CUDA_VISIBLE_DEVICES=%r torch_dev=%d pci=%s gl=%s",
            os.environ.get("CUDA_VISIBLE_DEVICES"), dev, pci,
            self.ctx.info.get("GL_RENDERER", "?"))
        # (H, W, 4) uint8, GL orientation (bottom-up), RGBA — the server flips /
        # drops alpha / resizes on the GPU.
        self._cuda_dst = torch.empty((self.height, self.width, 4),
                                     dtype=torch.uint8, device="cuda")
        # Interop (register AND map) fails cudaErrorInvalidGraphicsContext(208)
        # while self._color is bound as the active FBO color attachment. Unbind it
        # (bind a scratch FBO, glFinish) BEFORE registering AND before each map —
        # a fresh, unbound texture registers+maps fine (init probe), a bound one
        # does not.
        self._unbind_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((1, 1), 4)])
        self.ctx.finish()
        self._unbind_fbo.use()
        err, res = rt.cudaGraphicsGLRegisterImage(
            self._color.glo, 0x0DE1,  # GL_TEXTURE_2D
            rt.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly)
        if int(err) != 0:
            raise RuntimeError(f"cudaGraphicsGLRegisterImage failed: {int(err)}")
        self._cuda_res = res
        # The dst tensor's VRAM address is stable, so build the CUDA-IPC payload
        # ONCE and reuse it. Calling reduce_tensor every step spawns a new IPC
        # ref-counter shared-memory segment per step, and the resource_tracker
        # churn crashes Ray workers (KeyError '/mp-...'). One payload => one segment.
        from torch.multiprocessing.reductions import reduce_tensor
        self._ipc_payload = reduce_tensor(self._cuda_dst)

    def _copy_fbo_to_cuda(self):
        """Map the registered color texture and copy it into self._cuda_dst
        (device->device), then sync. Returns the persistent CUDA tensor."""
        rt = self._rt
        W, H = self.width, self.height
        # Finish the render and unbind self._color (bind the scratch FBO) so the
        # texture is not the active render target — else map returns 208.
        self.ctx.finish()
        self._unbind_fbo.use()
        e = rt.cudaGraphicsMapResources(1, self._cuda_res, 0)[0]
        if int(e) != 0:
            raise RuntimeError(f"MapResources failed: {int(e)}")
        e, arr = rt.cudaGraphicsSubResourceGetMappedArray(self._cuda_res, 0, 0)
        if int(e) != 0:
            raise RuntimeError(f"GetMappedArray failed: {int(e)}")
        (e,) = rt.cudaMemcpy2DFromArray(   # cuda-python returns a 1-tuple (err,)
            self._cuda_dst.data_ptr(), W * 4, arr, 0, 0, W * 4, H,
            rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        if int(e) != 0:
            raise RuntimeError(f"Memcpy2DFromArray failed: {int(e)}")
        rt.cudaGraphicsUnmapResources(1, self._cuda_res, 0)
        self._torch.cuda.synchronize()
        return self._cuda_dst

    def _ensure_em_gl(self):
        """Lazy GL program + textures + FBO for the 2D EM pane compositor
        (GPU equivalent of pane2d.compose_left_parts). Renders EM->RGB with
        per-visible-segment 0.5 tint, entirely on the GPU so the pane can stay
        in VRAM for the CUDA-IPC feed."""
        if getattr(self, "_em_prog", None) is not None:
            return
        ctx = self.ctx
        self._em_prog = ctx.program(
            vertex_shader="""#version 330
                in vec2 pos; out vec2 uv;
                void main(){ uv = pos*0.5+0.5; gl_Position = vec4(pos,0.0,1.0); }""",
            fragment_shader="""#version 330
                uniform sampler2D em;      // grayscale, .r in [0,1]
                uniform usampler2D idx;    // per-pixel compact segment index
                uniform sampler2D lut;     // K x 1 RGBA8: rgb=color, a=visible
                uniform int show_all;      // 1 => SHOW_ALL (every segment paints)
                uniform vec2 ch_center;    // crosshair centre, GL (bottom-up) px
                uniform float ch_len;
                in vec2 uv; out vec4 frag;
                void main(){
                    float g = texture(em, uv).r;
                    uint k = texture(idx, uv).r;
                    vec4 e = texelFetch(lut, ivec2(int(k),0), 0);
                    float vis = (show_all==1) ? 1.0 : e.a;
                    vec3 c = (vis > 0.5) ? (0.5*e.rgb + 0.5*vec3(g)) : vec3(g);
                    vec2 fc = gl_FragCoord.xy;
                    if (abs(fc.y-ch_center.y)<0.5 && fc.x>=ch_center.x && fc.x<=ch_center.x+ch_len)
                        c = 0.5*vec3(1.0,0.0,0.0) + 0.5*c;     // red +x
                    if (abs(fc.x-ch_center.x)<0.5 && fc.y>=ch_center.y && fc.y<=ch_center.y+ch_len)
                        c = 0.5*vec3(0.0,1.0,0.0) + 0.5*c;     // green +y (top-down)
                    frag = vec4(c, 1.0);
                }""",
        )
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._em_vbo = ctx.buffer(quad.tobytes())
        self._em_vao = ctx.vertex_array(self._em_prog, [(self._em_vbo, "2f", "pos")])
        self._em_color = ctx.texture((self.width, self.height), 4)
        self._em_fbo = ctx.framebuffer(color_attachments=[self._em_color])
        self._em_tex = self._idx_tex = self._lut_tex = None
        self._em_tile_key = None
        self._uniq = np.zeros(0, dtype="u8")

    def render_em(self, em_gray, ids, visible, tile_key=None):
        """GPU-composited 2D EM pane -> (PANE, PANE, 3) uint8, matching
        pane2d.compose_left_parts. `tile_key` (fetch generation) caches the EM +
        index textures across steps; only the tiny per-step LUT changes."""
        from .pane2d import PANE, PANE_H, TOOLBAR
        import moderngl

        self._ensure_em_gl()
        ctx = self.ctx
        H, W = int(em_gray.shape[0]), int(em_gray.shape[1])
        key = tile_key if tile_key is not None else (id(em_gray), id(ids))
        if key != self._em_tile_key:
            emb = np.ascontiguousarray(np.asarray(em_gray, dtype=np.uint8))
            if self._em_tex is None or self._em_tex.size != (W, H):
                self._em_tex = ctx.texture((W, H), 1, dtype="u1")
                self._em_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._em_tex.write(emb.tobytes())
            if ids is not None:
                uniq, inv = np.unique(ids, return_inverse=True)
                self._uniq = uniq
                idxmap = inv.reshape(ids.shape).astype("u2")
                ih, iw = ids.shape
                if self._idx_tex is None or self._idx_tex.size != (iw, ih):
                    self._idx_tex = ctx.texture((iw, ih), 1, dtype="u2")
                    self._idx_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self._idx_tex.write(np.ascontiguousarray(idxmap).tobytes())
            else:
                self._uniq = np.zeros(0, dtype="u8")
                if self._idx_tex is None or self._idx_tex.size != (W, H):
                    self._idx_tex = ctx.texture((W, H), 1, dtype="u2")
                    self._idx_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self._idx_tex.write(np.zeros((H, W), dtype="u2").tobytes())
            self._em_tile_key = key
        uniq = self._uniq
        K = max(1, len(uniq))
        visset = {int(v) for v in visible}
        show_all = 1 if not visset else 0
        lut = np.zeros((K, 4), dtype="u1")
        for k, rid in enumerate(uniq):
            lut[k, 0:3] = (np.asarray(segment_color(int(rid))) * 255.0).astype("u1")
            lut[k, 3] = 255 if (show_all or int(rid) in visset) else 0
        if self._lut_tex is None or self._lut_tex.size != (K, 1):
            self._lut_tex = ctx.texture((K, 1), 4, dtype="u1")
            self._lut_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._lut_tex.write(np.ascontiguousarray(lut).tobytes())

        self._em_fbo.use()
        self._em_fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._em_tex.use(0); self._idx_tex.use(1); self._lut_tex.use(2)
        self._em_prog["em"].value = 0
        self._em_prog["idx"].value = 1
        self._em_prog["lut"].value = 2
        self._em_prog["show_all"].value = show_all
        # crosshair centre in GL (bottom-up) px; matches draw_crosshair (top-down).
        cy, cx = PANE_H // 2, PANE // 2
        # px row r == fbo row r (no flip) == gl_y r, so the crosshair centre in GL
        # coords is (cx, cy) directly.
        self._em_prog["ch_center"].value = (float(cx), float(cy))
        self._em_prog["ch_len"].value = float(int(min(900, 867) / 4 / 2))
        self._em_vao.render(mode=moderngl.TRIANGLE_STRIP)
        # No [::-1] flip: the EM texture uploads data row 0 -> GL v=0 (fbo bottom),
        # so reading bottom-up already yields top-down order (unlike the 3D scene,
        # whose orientation comes from the projection and needs the flip).
        px = np.frombuffer(self._em_fbo.read(components=4), dtype=np.uint8)
        px = px.reshape(self.height, self.width, 4)[:, :, :3]
        canvas = np.zeros((PANE, PANE, 3), dtype=np.uint8)
        canvas[TOOLBAR:] = px
        return canvas

    def render(self, root_id, position_nm, quat, zoom_nm,
               color, em_tile=None, em_extent_nm=None,
               em_gain: float = 1.0, to_cuda: bool = False):
        """Full 3D pane: mesh(es) + section plane + axis lines.

        Default returns (H, W, 3) uint8 (numpy). With `to_cuda=True` (dino-server
        CUDA-IPC path) it skips the CPU readback and instead returns a persistent
        torch CUDA tensor (H, W, 4) uint8 in GL orientation — the caller ships its
        IPC handle to the DINO server, which flips/drops-alpha/resizes on the GPU.

        `root_id` is one id or a sequence of them; `color` is correspondingly
        one RGB triple or one per id.
        """
        view, proj = self._matrices(position_nm, quat, zoom_nm)
        mvp = (proj @ view).astype("f4")
        mvp_b = mvp.T.copy().tobytes()  # column-major
        pos = np.asarray(position_nm, dtype="f4")
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)

        ldir = -(self._rot(quat) @ np.array([0.0, 0.0, 1.0]))
        ldir /= np.linalg.norm(ldir) + 1e-9

        self.prog["mvp"].write(mvp_b)
        self.prog["light"].value = (*(ldir * 0.8), 0.2)
        ids = self._as_ids(root_id)
        # `color` is one RGB triple (single-id callers) or one per id. An empty
        # selection draws no mesh at all -- NG's state with everything hidden.
        single = isinstance(root_id, (str, int)) or (
            len(color) > 0 and not isinstance(color[0],
                                              (list, tuple, np.ndarray)))
        self._draw_meshes(ids, [color] * len(ids) if single else list(color))

        if em_tile is not None:
            self._draw_plane(mvp_b, pos, em_tile, em_extent_nm,
                             (0.2 + abs(ldir[2]) * 0.8) * em_gain)

        # Axis lines, one-sided toward +axis.
        al = zoom_nm * (min(self.width, self.height) / self.height) / 4.0
        self.ctx.enable(self._moderngl.BLEND)
        verts = []
        for i, col in enumerate([(1, 0, 0, 0.5), (0, 1, 0, 0.5),
                                 (0, 0, 1, 0.5)]):
            a = np.zeros(3); a[i] = al
            verts += [*pos, *col, *(pos + a), *col]
        self._line_vbo.write(np.array(verts, dtype="f4").tobytes())
        self.line_prog["mvp"].write(mvp_b)
        self._line_vao.render(mode=1)  # LINES
        self.ctx.disable(self._moderngl.BLEND)

        if to_cuda:
            self._ensure_cuda()
            self._copy_fbo_to_cuda()          # refresh self._cuda_dst in place
            return self._ipc_payload          # stable CUDA-IPC (rebuild, args)
        px = np.frombuffer(self.fbo.read(components=4), dtype=np.uint8)
        return px.reshape(self.height, self.width, 4)[::-1, :, :3].copy()

    def pick_depth(self, root_id, position_nm, quat, zoom_nm,
                   plane_extent_nm=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Depth buffer of the PICKABLE content (mesh + section plane) at the
        given state, plus (view, proj) for unprojection.

        NG's move-to-mouse-position picks whatever pickable layer is under
        the cursor — the cross-section EM slice included — so the plane
        participates here even though clicks on it land on the z-plane.
        Returns depth as (H, W) float32, GL origin already flipped to
        image order (row 0 = top).
        """
        view, proj = self._matrices(position_nm, quat, zoom_nm)
        mvp_b = (proj @ view).astype("f4").T.copy().tobytes()
        pos = np.asarray(position_nm, dtype="f4")
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.prog["mvp"].write(mvp_b)
        self.prog["light"].value = (0.0, 0.0, 0.8, 0.2)
        ids = self._as_ids(root_id)
        self._draw_meshes(ids, [(1.0, 1.0, 1.0)] * len(ids))
        if plane_extent_nm is not None:
            # Depth-only participation: a 1px dummy texture is enough.
            dummy = np.zeros((2, 2), dtype=np.uint8)
            self._draw_plane(mvp_b, pos, dummy, plane_extent_nm, 1.0)
        depth = np.frombuffer(self._depth.read(), dtype="f4").reshape(
            self.height, self.width)[::-1].copy()
        return depth, view, proj

    def close(self) -> None:
        self.ctx.release()
