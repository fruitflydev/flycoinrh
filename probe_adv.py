import asyncio
from envcfg import load_env
from rhwallet import account, CHAIN_ID, RPC
from rhprovider import attach
from rhlive import accept_terms, dismiss_banners, go_dark

URL = "https://www.ponsfamily.com/launchpad/create"

PROBE = """() => {
  const ins = [...document.querySelectorAll('input,textarea,select')].map(e => {
    const r = e.getBoundingClientRect();
    return {ph: e.placeholder || '', id: e.id || '', type: e.type,
            val: (e.value || '').slice(0, 20),
            y: Math.round(r.y + scrollY), w: Math.round(r.width)}; })
    .filter(o => o.w > 30);
  const btns = [...document.querySelectorAll('button')].map(e => {
    const r = e.getBoundingClientRect();
    return {t: (e.textContent || '').trim().slice(0, 44),
            y: Math.round(r.y + scrollY), w: Math.round(r.width)}; })
    .filter(o => o.t && o.w > 40);
  const lines = document.body.innerText.split(String.fromCharCode(10))
    .filter(l => /creator|tax|fee|%/i.test(l)).slice(0, 16);
  return {inputs: ins, buttons: btns, mentions: lines}; }"""


async def main():
    from playwright.async_api import async_playwright
    env = load_env()
    acct = account(env)
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        page = await b.new_page(viewport={"width": 1280, "height": 900})
        await attach(page, acct, env.get("FLY_RH_RPC", RPC), CHAIN_ID,
                     allow_send=False)
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)
        await go_dark(page)
        await accept_terms(page)
        await dismiss_banners(page)
        await page.wait_for_timeout(1500)

        print("=== BEFORE Advanced ===")
        before = await page.evaluate(PROBE)
        for o in before["mentions"]:
            print("  ", o[:90])

        try:
            await page.get_by_role("button", name="Advanced").first.click(timeout=5000)
            print("\nclicked Advanced")
        except Exception as e:
            print("\nadvanced click failed:", str(e)[:100])
        await page.wait_for_timeout(2500)

        info = await page.evaluate(PROBE)
        for k in ("inputs", "buttons", "mentions"):
            print(f"\n== {k} ==")
            for o in info[k]:
                print("  ", str(o)[:120])
        await page.screenshot(path="build/pons_advanced.png")
        await b.close()


asyncio.run(main())
