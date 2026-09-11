"""
Record the live rig to a video file.

Opens http://localhost:4651 in a Playwright browser that is recording video,
presses START, waits for the run to finish, then closes the context so the
video is finalised and converts it to mp4.

This captures the interface itself - the brain canvas, the streamed pump.fun
frames, the stepper and telemetry - at full resolution, with no desktop, no
taskbar and no cursor of yours in shot.

The rig must already be running:  py rhlive.py --port 4651

  py record.py                 record whatever the rig is armed for
  py record.py --dry           refuse to run if FLY_LIVE=1
"""
import argparse
import asyncio
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "build" / "recordings"
UI = "http://localhost:4651"   # overridden by --port


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=360,
                    help="seconds to wait for the run to finish")
    ap.add_argument("--dry", action="store_true",
                    help="abort if the rig is armed to mint")
    ap.add_argument("--name", default=None)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--port", type=int, default=4651,
                    help="port the rig listens on; rhlive.py uses 4651")
    a = ap.parse_args()

    global UI
    UI = f"http://localhost:{a.port}"

    from playwright.async_api import async_playwright
    import requests

    st = requests.get(f"{UI}/status", timeout=10).json()
    print(f"rig: {st.get('venue', 'pump.fun')}  {st.get('chain', 'solana')}")
    print(f"     wallet {str(st.get('wallet'))[:10]}...  "
          f"{st.get('sol', 0):.6f} {st.get('unit','SOL')}  live={st.get('live')}")
    if a.dry and st.get("live"):
        raise SystemExit("rig is armed to mint and --dry was given; aborting")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": a.width, "height": a.height},
            record_video_dir=str(OUT),
            record_video_size={"width": a.width, "height": a.height},
        )
        page = await ctx.new_page()
        page.on("console", lambda m: m.type == "error"
                and print(f"  [ui.console] {m.text[:150]}"))

        await page.goto(UI, wait_until="domcontentloaded", timeout=45000)
        # let the connectome load and the idle brain get going, so the clip
        # does not open on a blank panel
        await page.wait_for_function(
            "() => document.getElementById('ncount')"
            " && /of 165,122/.test(document.getElementById('ncount').textContent)",
            timeout=60000)
        await page.wait_for_timeout(2500)
        print("connectome loaded in the UI")

        if a.name:
            await page.fill("#name", a.name)
        if a.ticker:
            await page.fill("#ticker", a.ticker)

        await page.click("#start")
        print("START pressed - recording ...")

        deadline = time.time() + a.timeout
        outcome = None
        while time.time() < deadline:
            await page.wait_for_timeout(2000)
            outcome = await page.evaluate(
                "() => document.getElementById('t-out').textContent.trim()")
            if outcome in ("minted", "error", "dry run", "incomplete", "disarmed"):
                break
            print(f"   ... {outcome}", end="\r", flush=True)
        print(f"\nrun finished: {outcome}")

        # hold on the final frame so the ending is readable
        await page.wait_for_timeout(6000)
        logs = await page.evaluate(
            """() => [...document.getElementById('log').children]
                 .slice(-6).map(e => e.textContent)""")
        for line in logs:
            print("  " + line.encode("ascii", "replace").decode())

        video = page.video
        await ctx.close()          # finalises the file
        await browser.close()
        raw = Path(await video.path())

    webm = OUT / f"flybrain-{stamp}.webm"
    raw.rename(webm)
    print(f"\nwrote {webm}  ({webm.stat().st_size/1e6:.1f} MB)")

    if shutil.which("ffmpeg"):
        mp4 = OUT / f"flybrain-{stamp}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                        "-pix_fmt", "yuv420p", str(mp4)], check=False)
        if mp4.exists():
            print(f"wrote {mp4}  ({mp4.stat().st_size/1e6:.1f} MB)")
    return outcome


if __name__ == "__main__":
    asyncio.run(main())
