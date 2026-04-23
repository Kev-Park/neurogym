"""
Minimal REINFORCE training loop for neurogym using Tinker as the LoRA
training + sampling backend.

Status: scaffolding only. The Tinker SDK surface below follows the
documented primitives (sample / forward_backward / optim_step /
save_weights_and_get_sampling_client) but exact import paths and
argument names may differ between SDK versions -- adjust as needed
when you first run it.

Usage:
    export TINKER_API_KEY=...
    python train_tinker.py
"""

import os
import sys
import json
import logging
import re
import traceback
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

import wandb
import tinker
from tinker.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ModelInput,
    SamplingParams,
    TensorData,
)
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)

from ngllib import Environment


# ---------------------------------------------------------------------------
# Logging (file + stdout)
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train")
log.info("log file: %s", LOG_PATH)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL        = "Qwen/Qwen3.5-4B"   
RENDERER_NAME     = "qwen3_5_disable_thinking"   
LORA_RANK         = 16
N_STEPS           = 30
GROUP_SIZE        = 4                    
EPISODE_LEN       = 8
MAX_NEW_TOKENS    = 1024
TEMPERATURE       = 1.0
LEARNING_RATE     = 1e-5

ACTION_SCHEMA = (
    '{"dx": int in [-50,50], "dy": int in [-50,50], "dz": int in [-50,50], '
    '"zoom": int in [500,3000]}'
)

ACTION_EXAMPLE = '{"dx": 0, "dy": 0, "dz": 30, "zoom": 1500}'

SYSTEM = (
    "You control a 3D Neuroglancer viewer by emitting one action per step.\n"
    "\n"
    "GOAL: move the viewpoint as FAR UP along the +z axis as possible. "
    "Every step you gain z-height you receive positive reward; every step "
    "you lose z-height you receive negative reward. Maximize total z travelled up.\n"
    "\n"
    f"ACTION SCHEMA: {ACTION_SCHEMA}\n"
    "  - dx, dy, dz: change in position along each axis (voxels per step)\n"
    "  - zoom: projection scale (larger = more zoomed out)\n"
    "\n"
    f"EXAMPLE (a strong action for this goal): {ACTION_EXAMPLE}\n"
    "  -> stays put in x/y, moves +30 along z (up), keeps zoom moderate.\n"
    "\n"
    "OUTPUT RULES: respond with ONLY the JSON object, no prose, no code fences, no extra keys."
)


# ---------------------------------------------------------------------------
# Action <-> text
# ---------------------------------------------------------------------------

ZERO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1500]

_JSON_RE = re.compile(r"\{.*?\}", re.S)

def parse_action(text: str) -> tuple[list, bool, str]:
    """Parse JSON action from model output.

    Returns (action_17d, parsed_ok, reason).
    On failure, returns ZERO_ACTION and a short reason for logging.
    """
    m = _JSON_RE.search(text)
    if not m:
        return list(ZERO_ACTION), False, "no_json_found"
    try:
        d = json.loads(m.group(0))
    except Exception as e:
        return list(ZERO_ACTION), False, f"json_decode: {e}"
    try:
        dx, dy, dz = int(d.get("dx", 0)), int(d.get("dy", 0)), int(d.get("dz", 0))
        zoom       = int(d.get("zoom", 1500))
    except Exception as e:
        return list(ZERO_ACTION), False, f"field_cast: {e}"

    # 17-dim layout from main.py:
    # [l_click, r_click, dbl_click, x, y, mod1, mod2, mod3,
    #  json_change, pos_dx, pos_dy, pos_dz, cross_scale,
    #  euler_x, euler_y, euler_z, proj_scale]
    return [0, 0, 0, 0, 0, 0, 0, 0, 1, dx, dy, dz, 0, 0, 0, 0, zoom], True, "ok"


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def build_prompt(pos_state, image_np, renderer):
    pos = pos_state[0]
    user_text = (
        f"Current position (x,y,z) = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}). "
        "Screenshot attached. Output the next action."
    )
    # tinker_cookbook Message is a TypedDict: role + content parts
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": np_to_pil(image_np)},
            {"type": "text",  "text": user_text},
        ]},
    ]
    return renderer.build_generation_prompt(messages)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class Step:
    prompt: "tinker.ModelInput"   
    output_tokens: list
    output_logprobs: list
    reward: float
    parsed: bool = True


def rollout(env, sampling_client, renderer, tokenizer, sampling_params,
            step_idx: int, g_idx: int) -> list[Step]:
    steps = []
    (pos_state, image_np), _ = env.prepare_state()
    for t in range(EPISODE_LEN):
        prompt = build_prompt(pos_state, image_np, renderer)
        try:
            fut = sampling_client.sample(
                prompt=prompt, num_samples=1, sampling_params=sampling_params
            )
            out = fut.result() if hasattr(fut, "result") else fut
        except Exception:
            log.exception("[step=%d g=%d t=%d] sample() failed", step_idx, g_idx, t)
            raise

        seq  = out.sequences[0]
        tok  = list(seq.tokens)
        lp   = list(seq.logprobs)
        text = tokenizer.decode(tok)
        action, ok, reason = parse_action(text)

        prev_z = float(pos_state[0][2])
        try:
            (pos_state, image_np), _env_reward, done, _ = env.step(action)
        except Exception:
            log.exception("[step=%d g=%d t=%d] env.step() failed, action=%s",
                          step_idx, g_idx, t, action)
            raise
        new_z = float(pos_state[0][2])
        reward = new_z - prev_z   # env's built-in reward is broken (aliasing bug)

        log.info(
            "[step=%d g=%d t=%d] z=%.2f dz=%+.2f reward=%+.3f parsed=%s "
            "reason=%s raw=%r action=%s",
            step_idx, g_idx, t, new_z, new_z - prev_z, reward,
            ok, reason, text[:200], action,
        )

        steps.append(Step(prompt=prompt, output_tokens=tok,
                          output_logprobs=lp, reward=float(reward),
                          parsed=bool(ok)))
        if done:
            break
    return steps


def build_datum(step: Step, advantage: float) -> Datum:
    """Build a REINFORCE Datum for Tinker's importance_sampling loss.

    Uses the cookbook helper to produce correctly-shifted targets that include
    real prompt-token IDs (image positions are zeros). Non-output positions
    get advantage=0 which zeros their contribution to the loss.
    """
    full_chunks = list(step.prompt.chunks) + [
        EncodedTextChunk(tokens=list(step.output_tokens))
    ]
    model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        full_chunks
    )

    total_len  = model_input.length              # = N - 1
    prompt_len = step.prompt.length
    out_len    = len(step.output_tokens)

    logprobs   = [0.0] * total_len
    advantages = [0.0] * total_len
    # Position prompt_len-1 predicts output[0], prompt_len predicts output[1], ...
    for j in range(out_len):
        pos = prompt_len - 1 + j
        if 0 <= pos < total_len:
            logprobs[pos]   = float(step.output_logprobs[j])
            advantages[pos] = float(advantage)

    return Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.int64)),
            "logprobs":      TensorData.from_torch(torch.tensor(logprobs,      dtype=torch.float32)),
            "advantages":    TensorData.from_torch(torch.tensor(advantages,    dtype=torch.float32)),
        },
    )


TINKER_API_KEY = "xxx"


def main():
    os.environ["TINKER_API_KEY"] = TINKER_API_KEY

    log.info("config: base_model=%s lora_rank=%d n_steps=%d group=%d ep_len=%d "
             "max_tok=%d temp=%.2f lr=%.1e",
             BASE_MODEL, LORA_RANK, N_STEPS, GROUP_SIZE, EPISODE_LEN,
             MAX_NEW_TOKENS, TEMPERATURE, LEARNING_RATE)

    wandb.init(
        project="neurogym-tinker",
        name=f"train_{datetime.now():%Y%m%d_%H%M%S}",
        config={
            "base_model": BASE_MODEL,
            "renderer": RENDERER_NAME,
            "lora_rank": LORA_RANK,
            "n_steps": N_STEPS,
            "group_size": GROUP_SIZE,
            "episode_len": EPISODE_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "learning_rate": LEARNING_RATE,
        },
    )

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=BASE_MODEL, rank=LORA_RANK,
    )
    tokenizer       = training_client.get_tokenizer()
    image_processor = get_image_processor(BASE_MODEL)
    renderer        = renderers.get_renderer(
        RENDERER_NAME, tokenizer,
        image_processor=image_processor, model_name=BASE_MODEL,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = AdamParams(learning_rate=LEARNING_RATE)

    env = Environment(headless=True, config_path="config.json", verbose=False)
    env.start_session(euler_angles=True, resize=False, add_mouse=False,
                      fast=True, image_path=None)

    try:
        for step in range(N_STEPS):
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"iter_{step}"
            )

            group = []
            for g in range(GROUP_SIZE):
                env.reset()
                group.append(rollout(env, sampling_client, renderer, tokenizer,
                                     sampling_params, step, g))

            returns  = np.array([sum(s.reward for s in traj) for traj in group])
            baseline = returns.mean()
            advs     = returns - baseline

            datums = []
            for traj, adv in zip(group, advs):
                datums.extend(build_datum(s, float(adv)) for s in traj)

            try:
                fb = training_client.forward_backward(
                    datums, loss_fn="importance_sampling"
                )
                if hasattr(fb, "result"):
                    fb.result()
                os_fut = training_client.optim_step(adam_params)
                if hasattr(os_fut, "result"):
                    os_fut.result()
            except Exception:
                log.exception("[step=%d] optimizer update failed", step)
                raise

            ep_lens      = [len(traj) for traj in group]
            all_rewards  = [s.reward for traj in group for s in traj]
            total_steps  = sum(ep_lens)
            parsed_count = sum(1 for traj in group for s in traj if s.parsed)
            parse_ok_rate = parsed_count / total_steps if total_steps else 0.0

            log.info("[step=%d] mean_return=%.3f min=%.3f max=%.3f n_datums=%d",
                     step, baseline, returns.min(), returns.max(), len(datums))

            wandb.log(
                {
                    "train/mean_return": float(baseline),
                    "train/min_return": float(returns.min()),
                    "train/max_return": float(returns.max()),
                    "train/std_return": float(returns.std()),
                    "train/advantage_abs_mean": float(np.abs(advs).mean()),
                    "train/mean_reward_per_step": float(np.mean(all_rewards)) if all_rewards else 0.0,
                    "train/mean_episode_len": float(np.mean(ep_lens)),
                    "train/parse_ok_rate": float(parse_ok_rate),
                    "train/n_datums": len(datums),
                },
                step=step,
            )
    except Exception:
        log.exception("training crashed")
        raise
    finally:
        try:
            env.end_session()
        except Exception:
            log.exception("env.end_session() failed")
        try:
            wandb.finish()
        except Exception:
            log.exception("wandb.finish() failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("unhandled:\n%s", traceback.format_exc())
        sys.exit(1)
