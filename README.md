# Neurogym

## 1. Playwright & Chromium Installation

Install Playwright and let it download its own Chromium — no sudo or manual ChromeDriver setup required.

```bash
pip install playwright
playwright install chromium
```

On PEP 668 / externally-managed clusters (where pip warns about system packages):

```bash
pip install playwright --break-system-packages
playwright install chromium
```

Chromium is downloaded to `~/.cache/ms-playwright/` (~450 MB). To redirect to scratch storage:

```bash
PLAYWRIGHT_BROWSERS_PATH=/path/to/scratch playwright install chromium
```

No changes to `config.json` are needed for browser paths.

## 2. Screenshot Rendering

At first, the screenshots come out blank because of rendering issues. I configure Chrome to use SwiftShader to provide complete WebGL support without a physical GPU display:

```
--headless=new
--use-gl=angle
--use-angle=vulkan
--enable-features=Vulkan
--enable-unsafe-swiftshader
```

These flags are set in `environment.py` → `initialize_chrome_options()` when `headless=True`.
