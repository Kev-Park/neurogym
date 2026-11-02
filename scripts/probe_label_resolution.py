"""Compare two ngllib checkouts against identical settled Chrome captures.

Run capture once with --cases (a list of {name, state}), then simulator in
each checkout with a different --tag, then compare. GPU allocation required
for capture and simulator. Axes are disabled in both to measure labels only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ngllib.simulator import pane2d


def scores(a, b, right=False):
    a, b = pane2d.mask_ui(a), pane2d.mask_ui(b)
    a, b = (a[:, 450:], b[:, 450:]) if right else (a[:, :450], b[:, :450])
    ca = np.ptp(a.astype(np.int16), axis=2) > 25
    cb = np.ptp(b.astype(np.int16), axis=2) > 25
    ga, gb = a.mean(2), b.mean(2)
    h, w = (ga.shape[0] // 16) * 16, (ga.shape[1] // 16) * 16
    ta = ga[:h, :w].reshape(h//16, 16, w//16, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    tb = gb[:h, :w].reshape(h//16, 16, w//16, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    ma, mb = ta.mean(1), tb.mean(1)
    va, vb = ta.var(1), tb.var(1)
    cov = ((ta-ma[:, None])*(tb-mb[:, None])).mean(1)
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    ssim = ((2*ma*mb+c1)*(2*cov+c2)/((ma*ma+mb*mb+c1)*(va+vb+c2))).mean()
    return dict(iou=float((ca & cb).sum()/max(1, (ca | cb).sum())),
                ssim=float(ssim), mad=float(np.abs(a.astype(float)-b).mean()),
                reference_pixels=int(ca.sum()))


def capture(args):
    from ngllib.chrome import ChromeRenderer
    r = ChromeRenderer(window_size=(1800, 900), capture_scale=0.5,
                       left_pane=True, right_pane=True, screenshot_format="png")
    cases = json.loads(Path(args.cases).read_text())
    r.open()
    try:
        for case in cases:
            r.reset_to(case["state"])
            r.page.evaluate("viewer.showAxisLines.value = false")
            gpu = r.page.evaluate("""() => {const g=viewer.display.gl;
              return g.getParameter(g.getExtension('WEBGL_debug_renderer_info').UNMASKED_RENDERER_WEBGL)}""")
            assert "NVIDIA" in gpu and "SwiftShader" not in gpu, gpu
            r.page.wait_for_timeout(4000)
            state, frame = r.observe()
            stable = False
            for _ in range(4):
                r.page.wait_for_timeout(1500)
                state, later = r.observe()
                stable = np.array_equal(pane2d.mask_ui(frame), pane2d.mask_ui(later))
                frame = later
                if stable:
                    break
            assert stable, f"Chrome did not settle: {case['name']}"
            case["state"] = state
            Image.fromarray(frame).save(args.out/f"chrome_{case['name']}.png")
            print("CAPTURE", case["name"], gpu, flush=True)
        (args.out/"cases.json").write_text(json.dumps(cases, indent=2))
    finally:
        r.close()


def simulator(args):
    from ngllib.simulator import SimulatorRenderer
    pane2d.draw_crosshair = lambda rgb: None
    r = SimulatorRenderer(window_size=(1800, 900), capture_scale=0.5,
                          left_pane=True, right_pane=True)
    rows = []
    try:
        for case in json.loads((args.out/"cases.json").read_text()):
            start = time.perf_counter()
            r.reset_to(case["state"])
            state, image = r.observe()
            reset_s = time.perf_counter()-start
            assert state == case["state"], (case["name"], state, case["state"])
            assert image.shape == (450, 900, 3)
            assert r._tiles["ids"] is not None
            if args.tag == "after":
                assert r._tiles["ids"].shape == (867, 900)
            Image.fromarray(image).save(args.out/f"{args.tag}_{case['name']}.png")
            times = []
            for _ in range(12):
                start = time.perf_counter()
                r.observe()
                times.append(time.perf_counter()-start)
            row = dict(name=case["name"], reset_s=reset_s,
                       cached_median_s=float(np.median(times)), ids_shape=r._tiles["ids"].shape)
            rows.append(row)
            print(args.tag, row, flush=True)
        (args.out/f"{args.tag}_timing.json").write_text(json.dumps(rows, indent=2))
    finally:
        r.close()


def compare(args):
    rows = []
    for case in json.loads((args.out/"cases.json").read_text()):
        frames = {tag: np.asarray(Image.open(args.out/f"{tag}_{case['name']}.png"))
                  for tag in ("chrome", "before", "after")}
        row = dict(name=case["name"], state=case["state"])
        for tag in ("before", "after"):
            row[tag] = {"left": scores(frames["chrome"], frames[tag]),
                        "right": scores(frames["chrome"], frames[tag], right=True)}
        rows.append(row)
        print("PARITY", json.dumps(row), flush=True)
    selected = [r for r in rows if r["state"]["segments"] and r["before"]["left"]["reference_pixels"] >= 50]
    assert len(selected) >= 8, "Insufficient nonempty segmentation comparisons"
    summary = {tag: {pane: {metric: float(np.mean([r[tag][pane][metric] for r in selected]))
                           for metric in ("iou", "ssim", "mad")}
                     for pane in ("left", "right")} for tag in ("before", "after")}
    summary["nonempty_cases"] = len(selected)
    summary["worst_iou_delta"] = min(r["after"]["left"]["iou"]-r["before"]["left"]["iou"] for r in selected)
    summary["timing"] = {tag: json.loads((args.out/f"{tag}_timing.json").read_text()) for tag in ("before", "after")}
    (args.out/"summary.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)
    assert summary["after"]["left"]["iou"] > summary["before"]["left"]["iou"] + 0.02
    assert summary["worst_iou_delta"] >= -0.03
    assert summary["after"]["right"]["ssim"] >= summary["before"]["right"]["ssim"] - 0.01
    print("LABEL-RESOLUTION-PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("capture", "simulator", "compare"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases")
    ap.add_argument("--tag", choices=("before", "after"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
