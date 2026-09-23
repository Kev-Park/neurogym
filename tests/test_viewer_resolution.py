"""Which viewer gets served, and where its provenance comes from."""

import json

import pytest

from ngllib.chrome import (
    HOSTED_VIEWER,
    PACKAGED_VIEWER,
    packaged_viewer_dir,
    resolve_viewer,
    viewer_provenance,
)


def fake_dist(tmp_path, name="dist", build=None):
    d = tmp_path / name
    d.mkdir()
    (d / "index.html").write_text("<html>")
    if build is not None:
        (d / "build.json").write_text(json.dumps(build))
    return d


def test_explicit_path_wins(tmp_path, monkeypatch):
    d = fake_dist(tmp_path)
    monkeypatch.setenv("NGL_VIEWER_DIST", str(fake_dist(tmp_path, "other")))
    assert resolve_viewer(str(d)) == d.resolve()


def test_env_var_is_next(tmp_path, monkeypatch):
    d = fake_dist(tmp_path)
    monkeypatch.setenv("NGL_VIEWER_DIST", str(d))
    assert resolve_viewer(None) == d.resolve()


def test_hosted_must_be_asked_for(monkeypatch):
    monkeypatch.delenv("NGL_VIEWER_DIST", raising=False)
    assert resolve_viewer(HOSTED_VIEWER) == HOSTED_VIEWER


def test_packaged_is_the_default(monkeypatch):
    monkeypatch.delenv("NGL_VIEWER_DIST", raising=False)
    packaged = packaged_viewer_dir()
    assert packaged is not None, "the build should ship in ngllib/viewer"
    assert resolve_viewer(None) == packaged
    assert resolve_viewer(PACKAGED_VIEWER) == packaged


def test_a_directory_without_index_html_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("NGL_VIEWER_DIST", raising=False)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no index.html"):
        resolve_viewer(str(tmp_path / "empty"))


def test_provenance_reads_build_json(tmp_path):
    d = fake_dist(tmp_path, "withbuild", build={
        "branch": "rl-recording", "commit": "2556d44570921c225c", "dirty": False,
        "built_at": "2026-09-18T15:45:20Z"})
    line = viewer_provenance(d)
    assert "rl-recording@2556d4457092" in line and "built 2026-09-18" in line
    assert "DIRTY" not in line


def test_provenance_flags_a_dirty_build(tmp_path):
    d = fake_dist(tmp_path, "dirtybuild", build={"branch": "x", "commit": "abc", "dirty": True})
    assert "DIRTY" in viewer_provenance(d)


def test_provenance_survives_a_missing_build_json(tmp_path):
    assert "no build.json" in viewer_provenance(fake_dist(tmp_path, "nobuild"))
