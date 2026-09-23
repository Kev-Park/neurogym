"""CAVE credential resolution: config, files, and the CloudVolume bridge."""

import json
import os

import pytest

from ngllib import auth
from ngllib.errors import RendererError


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Never touch the caller's real ~/.cloudvolume in a test."""
    monkeypatch.setattr(auth, "_MATERIALIZED", None)
    monkeypatch.setenv("CLOUD_VOLUME_DIR", str(tmp_path / "cv"))
    yield


def secret_file(tmp_path, token="tok-from-file"):
    f = tmp_path / "cave-secret.json"
    f.write_text(json.dumps({"token": token}))
    return f


def test_inline_token_wins(tmp_path):
    f = secret_file(tmp_path)
    cfg = {"cave_token": "tok-inline", "cave_secret": str(f)}
    assert auth.cave_token_from_config(cfg) == "tok-inline"


def test_config_secret_path_is_read(tmp_path):
    cfg = {"cave_token": "", "cave_secret": str(secret_file(tmp_path))}
    assert auth.cave_token_from_config(cfg) == "tok-from-file"


def test_example_config_yields_no_token():
    # The shipped config.json carries the fields empty; a public dataset must
    # not need a token, so this has to be None rather than an error.
    assert auth.cave_token_from_config({"cave_token": "", "cave_secret": ""}) is None
    assert auth.cave_token_from_config({}) is None


def test_read_cave_token_errors_name_the_fix(tmp_path):
    with pytest.raises(RendererError, match="setup_token"):
        auth.read_cave_token(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nope": 1}))
    with pytest.raises(RendererError, match="no 'token' field"):
        auth.read_cave_token(str(bad))


def test_ensure_cave_secret_is_visible_to_cloudvolume(tmp_path):
    root = auth.ensure_cave_secret("tok-materialised")
    written = root / "secrets" / "cave-secret.json"
    assert json.loads(written.read_text()) == {"token": "tok-materialised"}
    # CloudVolume reads $CLOUD_VOLUME_DIR/secrets/...; spawned workers inherit it.
    assert os.environ["CLOUD_VOLUME_DIR"] == str(root)
    assert auth.cloudvolume_secret_exists()
    if os.name != "nt":
        assert oct(written.stat().st_mode)[-3:] == "600"


def test_ensure_cave_secret_is_idempotent():
    first = auth.ensure_cave_secret("tok")
    assert auth.ensure_cave_secret("tok") == first


def test_other_secrets_survive_the_redirect(tmp_path, monkeypatch):
    home = tmp_path / "cv" / "secrets"
    home.mkdir(parents=True)
    (home / "google-secret.json").write_text('{"type": "service_account"}')
    root = auth.ensure_cave_secret("tok")
    # Relocating CLOUD_VOLUME_DIR must not cost the user other credentials.
    assert (root / "secrets" / "google-secret.json").is_file()


def test_cloudvolume_secret_exists_is_false_when_empty(tmp_path):
    assert not auth.cloudvolume_secret_exists()
