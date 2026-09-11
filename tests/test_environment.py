"""`Environment` over a fake renderer (gate 1): spaces, the reset/step
skeleton, action dispatch, reset-ahead, error propagation, the UI mask."""

from __future__ import annotations

import numpy as np
import pytest

from ngllib import ChromeRenderer, Environment, Renderer, SimulatorRenderer
from ngllib.errors import RendererError


def _edit(dpos=(0, 0, 0), dps=0.0):
    return {"action_type": 3, "mouse_xy": np.zeros(2, np.float32),
            "modifiers": np.zeros(3, np.int8), "delta_pos": np.asarray(dpos, np.float32),
            "delta_xs_scale": np.zeros(1, np.float32), "delta_orient": np.zeros(3, np.float32),
            "delta_proj_scale": np.asarray([dps], np.float32)}


def _click(kind, x, y, mods=(0, 0, 0)):
    return {"action_type": kind, "mouse_xy": np.asarray([x, y], np.float32),
            "modifiers": np.asarray(mods, np.int8), "delta_pos": np.zeros(3, np.float32),
            "delta_xs_scale": np.zeros(1, np.float32), "delta_orient": np.zeros(3, np.float32),
            "delta_proj_scale": np.zeros(1, np.float32)}


def test_spaces_follow_the_renderer_layout(fake_renderer):
    env = Environment(backend=fake_renderer(window_size=(1800, 900), capture_scale=0.5,
                                            left_pane=True, right_pane=True),
                      orientation="euler")
    assert env.observation_space["image"].shape == (450, 900, 3)
    assert env.observation_space["orientation"].shape == (3,)
    assert env.action_space["mouse_xy"].high.tolist() == [1800.0, 900.0]
    single = Environment(backend=fake_renderer(image_size=(224, 224)))
    assert single.observation_space["image"].shape == (224, 224, 3)
    assert single.observation_space["orientation"].shape == (4,)


def test_backend_must_be_a_renderer():
    with pytest.raises(TypeError):
        Environment(backend=object())


def test_shipped_renderers_satisfy_the_protocol():
    """Constructing either needs neither a browser nor GL."""
    assert isinstance(ChromeRenderer(), Renderer)
    assert isinstance(SimulatorRenderer(), Renderer)


def test_reset_and_step_skeleton(fake_renderer):
    r = fake_renderer()
    env = Environment(backend=r, orientation="euler")
    obs, info = env.reset(seed=0)
    assert r.opened
    assert info["step"] == 0 and info["json_state"]["segments"] == ["42"]
    assert obs["image"].shape == (900, 900, 3)
    assert obs["proj_scale"][0] == 1000.0

    obs, reward, term, trunc, info = env.step(_edit(dpos=(10, 0, 0), dps=-500))
    assert r.calls[-1][0] == "set_state"
    assert obs["position"].tolist() == [10.0, 0.0, 0.0]
    assert obs["proj_scale"][0] == 500.0
    assert info["json_state"]["position"] == [10.0, 0.0, 0.0]
    assert (reward, term, trunc) == (0.0, False, False)
    env.close()
    assert r.closed


def test_zoom_rule_applies_through_the_env(fake_renderer):
    env = Environment(backend=fake_renderer())
    env.reset()
    obs, *_ = env.step(_edit(dps=-1000))          # 1000 -> 0: rejected, keeps 1000
    assert obs["proj_scale"][0] == 1000.0


def test_click_dispatch_reaches_the_renderer_with_ng_kind_names(fake_renderer):
    r = fake_renderer()
    env = Environment(backend=r)
    env.reset()
    env.step(_click(0, 5, 6))
    env.step(_click(1, 950.0, 400.0, mods=(1, 0, 1)))
    obs, *_ = env.step(_click(2, 100.0, 100.0))
    kinds = [c[1] for c in r.calls if c[0] == "click"]
    assert kinds == ["left_click", "right_click", "double_click"]
    assert r.calls[-2][4] == "Shift, Alt"
    assert obs["position"][:2].tolist() == [950.0, 400.0]
    assert env._state["segments"] == ["42", "7"]
    with pytest.raises(ValueError):
        env.step(_click(4, 0, 0))


def test_hooks_receive_task_info_and_prev_obs(fake_renderer, provider):
    seen = {}

    def reward_factory(task_info):
        seen["task_info"] = task_info
        return lambda obs, action, prev_obs, terminated: float(obs["position"][0] - prev_obs["position"][0])

    def termination_factory(task_info):
        return lambda obs, action, prev_obs: obs["position"][0] >= 3

    env = Environment(backend=fake_renderer(), reset_state_provider=provider,
                      reward_factory=reward_factory, termination_factory=termination_factory)
    obs, info = env.reset()
    assert info["task_info"] == {"segment_id": "1"} == seen["task_info"]
    obs, reward, term, *_ = env.step(_edit(dpos=(2, 0, 0)))
    assert reward == 2.0 and term is True


def test_reset_ahead_draws_once_and_warms_the_renderer(fake_renderer, provider):
    r = fake_renderer()
    env = Environment(backend=r, reset_state_provider=provider, reset_ahead=True)
    env.reset()                                    # draw 1 (this episode) + draw 2 (warmed)
    assert provider.draws == 2 and len(r.warmed) == 1
    assert r.warmed[0]["segments"] == ["2"]
    env.step(_edit()); env.step(_edit())
    assert provider.draws == 2                     # no re-draw within the episode
    obs, info = env.reset()                        # adopts draw 2, warms draw 3
    assert info["task_info"] == {"segment_id": "2"} and r.calls[-1][1]["segments"] == ["2"]
    assert provider.draws == 3


def test_reset_ahead_respects_the_delay(fake_renderer, provider):
    r = fake_renderer()
    env = Environment(backend=r, reset_state_provider=provider, reset_ahead=True,
                      reset_ahead_after_steps=2)
    env.reset()
    assert len(r.warmed) == 0
    env.step(_edit())
    assert len(r.warmed) == 0
    env.step(_edit())
    assert len(r.warmed) == 1


def test_explicit_state_resets_never_consume_provider_draws(fake_renderer, provider):
    r = fake_renderer()
    env = Environment(backend=r, reset_state_provider=provider, reset_ahead=True)
    st = {"position": [9, 9, 9], "crossSectionScale": 1.0, "projectionScale": 100.0,
          "segments": ["x"]}
    obs, info = env.reset(options={"state": st})
    assert provider.draws == 0 and r.warmed == []
    assert info["task_info"] == {"segment_id": "x"}        # via task_info_from_state
    assert obs["position"].tolist() == [9.0, 9.0, 9.0]


def test_seeded_reset_discards_a_warmed_state(fake_renderer, provider):
    r = fake_renderer()
    env = Environment(backend=r, reset_state_provider=provider, reset_ahead=True)
    env.reset()
    obs, info = env.reset(seed=123)                # reseeded: inline draw, not the warm one
    assert info["task_info"] == {"segment_id": "3"}


def test_renderer_errors_propagate_untouched(fake_renderer):
    r = fake_renderer()
    env = Environment(backend=r)
    env.reset()
    r.fail_next_observe = True
    with pytest.raises(RendererError):
        env.step(_edit())
    assert env._last_step_glitched


def test_ui_mask_only_on_the_calibrated_two_pane_frame(fake_renderer, monkeypatch):
    monkeypatch.delenv("NGL_MASK_UI", raising=False)
    two = Environment(backend=fake_renderer(capture_scale=0.5, left_pane=True, right_pane=True))
    obs, _ = two.reset()
    assert obs["image"].shape == (450, 900, 3)
    assert obs["image"][0, 0].tolist() == [0, 0, 0]        # toolbar strip blanked
    assert obs["image"][200, 450].tolist() == [200, 200, 200]
    one = Environment(backend=fake_renderer(capture_scale=0.5))
    obs, _ = one.reset()
    assert obs["image"][0, 0].tolist() == [200, 200, 200]  # single pane: untouched
    off = Environment(backend=fake_renderer(capture_scale=0.5, left_pane=True, right_pane=True),
                      mask_ui=False)
    assert off.reset()[0]["image"][0, 0].tolist() == [200, 200, 200]


def test_image_size_resizes_after_the_mask(fake_renderer):
    env = Environment(backend=fake_renderer(capture_scale=0.5, left_pane=True, right_pane=True,
                                            image_size=(90, 45)))
    obs, _ = env.reset()
    assert obs["image"].shape == (45, 90, 3)
    assert obs["image"][0, 0].tolist() == [0, 0, 0]        # masked at capture scale, then resized


def test_gym_make_with_backend(fake_renderer):
    import gymnasium as gym

    env = gym.make("Neuroglancer-v0", backend=fake_renderer(), max_episode_steps=2)
    env.reset()
    env.step(_edit())
    _, _, _, truncated, _ = env.step(_edit())
    assert truncated
