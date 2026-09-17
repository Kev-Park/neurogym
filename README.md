# Neuroglancer Integrations for NGL_bench

---

This repository provides a Python library called `ngllib`, which contains a Gymnasium environment optimized for training and evaluating connectome proofreading agents in [Neuroglancer](https://github.com/seung-lab/neuroglancer), a web-based 3D connectomics viewer. The library abstracts many interactions with Neuroglancer (extracting state information, handling action inputs, etc.) behind an easy-to-use programmatic interface. 

---

## Features

Some of the capabilities `ngllib` provides (not exhaustive) includes:

- Headless GPU-accelerated Neuroglancer rendering using Chromium with automatic browser restarting and error handling for long training run stability
- Distributed (multi-node) environment stepping with support for low-latency socket or filesystem-based communication
- Already-validated observation and action spaces for policy training and deployment
- Customizable reset behavior with support for reset curriculums
- Custom reward function handling
- Custom episode termination handling
- Error handling

Other notable work-in-progress features include:

- An alternative browser-less Neuroglancer renderer with **3.4x** the sampling rate of Chromium-based renderers
- Automatic login handling for private volume loading

---

## Getting Started

### Installation

1. Clone the repo using:

  `git clone [https://github.com/seung-lab/ngl_bench.git](https://github.com/seung-lab/ngl_bench.git)`[.](https://github.com/seung-lab/ngl_bench.git)
2. Change directory to the repository root:  
`cd ngl_bench`
3. Install `ngllib`:  
`pip install -e .`
4. Install Chromium via Playwright:  
`playwright install chromium`

Please note `ngllib` requires Python 3.12 or newer.

### Quickstart

The following script locally opens a Chromium-based view of Neuroglancer and spins around a neuron:

```python
import numpy as np
from ngllib import Environment

env = Environment(orientation="euler")      # headless Chromium, default Neuroglancer view
obs, info = env.reset()

action = {                                   # a viewer-state edit: rotate 0.1 rad about y each step
    "action_type": 3,
    "mouse_xy": np.zeros(2, np.float32),
    "modifiers": np.zeros(3, np.int8),
    "delta_pos": np.zeros(3, np.float32),
    "delta_xs_scale": np.zeros(1, np.float32),
    "delta_orient": np.array([0.0, 0.1, 0.0], np.float32),
    "delta_proj_scale": np.zeros(1, np.float32),
}
for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs["orientation"], obs["image"].shape)

env.close()
```

Documentation is WIP; please reach out to [kp0374@princeton.edu](mailto:kp0374@princeton.edu) if you have any questions.

---

## License

Apache 2.0 - see [LICENSE](LICENSE).