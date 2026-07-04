"""Headless Puter image generation via Playwright — no auth token in .env.

The Puter session lives in a persistent browser profile (puter-session/),
so nothing shareable/leakable sits in config. Log in once, then generate
headlessly.

Usage:
    python puter_headless.py login                 # one-time, opens a visible browser
    python puter_headless.py gen "a red fox" out.png [model]

Requires: pip install playwright && playwright install chromium
"""
import base64
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
SESSION_DIR = HERE / "puter-session"
BRIDGE = (HERE / "bridge.html").as_uri()


def login():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(SESSION_DIR), headless=False)
        page = ctx.pages[0]
        page.goto(BRIDGE)
        page.evaluate("puter.auth.signIn()")
        print("Log in to Puter in the browser window, then close it.")
        page.wait_for_event("close", timeout=0)  # wait until user closes the window
        ctx.close()
    print("Session saved to", SESSION_DIR)


def generate(prompt: str, output_path: str, model: str = "black-forest-labs/FLUX.1-schnell") -> str:
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(SESSION_DIR), headless=True)
        page = ctx.pages[0]
        page.goto(BRIDGE)
        data_url = page.evaluate("generateImage", [prompt, model])
        ctx.close()
    Path(output_path).write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
    return output_path


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "login":
        login()
    elif len(sys.argv) >= 4 and sys.argv[1] == "gen":
        print(generate(sys.argv[2], sys.argv[3], *sys.argv[4:5]))
    else:
        print(__doc__)
        sys.exit(1)
