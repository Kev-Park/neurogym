# Neurogym

## 1. Chrome & ChromeDriver Installation

Install Chrome for Testing and the matching ChromeDriver.

```bash
mkdir -p chrome-install && cd chrome-install

# Check latest stable version
curl -s https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json | python3 -m json.tool | head -10

# Download Chrome + ChromeDriver
VERSION="147.0.7727.50"
curl -O "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chrome-linux64.zip"
curl -O "https://storage.googleapis.com/chrome-for-testing-public/${VERSION}/linux64/chromedriver-linux64.zip"


unzip chrome-linux64.zip && unzip chromedriver-linux64.zip
chmod +x chrome-linux64/chrome chromedriver-linux64/chromedriver
```

Then update `config.json` with the correct paths:

```json
{
    "driver_path_linux": "/path/to/chrome-install/chromedriver-linux64/chromedriver",
    "chrome_binary_path_linux": "/path/to/chrome-install/chrome-linux64/chrome"
}
```

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
