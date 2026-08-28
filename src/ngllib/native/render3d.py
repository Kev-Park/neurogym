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

from collections import OrderedDict

import numpy as np

from .camera import projection_camera


class MeshRenderer:
    """Offscreen renderer for one (width, height) pane geometry.

    `width`/`height` are the pane's true captured pixels BELOW the toolbar
    (450 x 433 at capture scale 0.5) so the optical center and aspect match
    the browser.
    """

    def __init__(self, width: int, height: int,
                 mesh_budget_bytes: int = 2 << 30):
        import moderngl

        self._moderngl = moderngl
        self.width, self.height = width, height
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
                    float g = texture(em, v_uv).r * lfac;
                    frag = vec4(g, g, g, 1.0);
                }""",
        )
        # LRU mesh VAOs: root_id -> (vao, [vbo, ibo], bytes)
        self._vaos: OrderedDict[str, tuple] = OrderedDict()
        self._vao_bytes = 0
        self._budget = mesh_budget_bytes

    def has_mesh(self, root_id: str) -> bool:
        return root_id in self._vaos

    def load_mesh(self, root_id: str, vertices_nm, faces,
                  normals=None) -> None:
        """Indexed draw with smooth per-vertex normals (precomputed via
        `normals`, e.g. by em.worker_mesh, or derived here); LRU-evicts
        past budget."""
        if root_id in self._vaos:
            self._vaos.move_to_end(root_id)
            return
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
        vbo = self.ctx.buffer(np.hstack([v, vn.astype("f4")]).tobytes())
        ibo = self.ctx.buffer(f.tobytes())
        vao = self.ctx.vertex_array(
            self.prog, [(vbo, "3f 3f", "pos", "nrm")], index_buffer=ibo)
        nbytes = vbo.size + ibo.size
        self._vaos[root_id] = (vao, [vbo, ibo], nbytes)
        self._vao_bytes += nbytes
        while self._vao_bytes > self._budget and len(self._vaos) > 1:
            _, (old_vao, old_bufs, old_bytes) = self._vaos.popitem(last=False)
            old_vao.release()
            for b in old_bufs:
                b.release()
            self._vao_bytes -= old_bytes

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
        tex = self.ctx.texture(em_tile.shape[::-1], 1,
                               np.ascontiguousarray(em_tile).tobytes())
        tex.use(0)
        hx, hy = em_extent_nm[0] / 2.0, em_extent_nm[1] / 2.0
        quad = np.array([
            pos[0] - hx, pos[1] - hy, pos[2], 0, 0,
            pos[0] + hx, pos[1] - hy, pos[2], 1, 0,
            pos[0] - hx, pos[1] + hy, pos[2], 0, 1,
            pos[0] + hx, pos[1] + hy, pos[2], 1, 1,
        ], dtype="f4")
        vbo = self.ctx.buffer(quad.tobytes())
        vao = self.ctx.vertex_array(
            self.plane_prog, [(vbo, "3f 2f", "pos", "uv")])
        self.plane_prog["mvp"].write(mvp_b)
        self.plane_prog["lfac"].value = float(lfac)
        vao.render(mode=5)  # TRIANGLE_STRIP
        vao.release(); vbo.release(); tex.release()

    def render(self, root_id: str, position_nm, quat, zoom_nm,
               color, em_tile=None, em_extent_nm=None,
               em_gain: float = 1.0) -> np.ndarray:
        """Full 3D pane: mesh + section plane + axis lines. (H, W, 3) uint8."""
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
        self.prog["color"].value = tuple(float(c) for c in color)
        vao, _, _ = self._vaos[root_id]
        self._vaos.move_to_end(root_id)
        vao.render(mode=4)

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
        vbo = self.ctx.buffer(np.array(verts, dtype="f4").tobytes())
        lvao = self.ctx.vertex_array(
            self.line_prog, [(vbo, "3f 4f", "pos", "col")])
        self.line_prog["mvp"].write(mvp_b)
        lvao.render(mode=1)  # LINES
        lvao.release(); vbo.release()
        self.ctx.disable(self._moderngl.BLEND)

        px = np.frombuffer(self.fbo.read(components=4), dtype=np.uint8)
        return px.reshape(self.height, self.width, 4)[::-1, :, :3].copy()

    def pick_depth(self, root_id: str, position_nm, quat, zoom_nm,
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
        self.prog["color"].value = (1.0, 1.0, 1.0)
        vao, _, _ = self._vaos[root_id]
        vao.render(mode=4)
        if plane_extent_nm is not None:
            # Depth-only participation: a 1px dummy texture is enough.
            dummy = np.zeros((2, 2), dtype=np.uint8)
            self._draw_plane(mvp_b, pos, dummy, plane_extent_nm, 1.0)
        depth = np.frombuffer(self._depth.read(), dtype="f4").reshape(
            self.height, self.width)[::-1].copy()
        return depth, view, proj

    def close(self) -> None:
        self.ctx.release()
