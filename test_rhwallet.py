"""
Does ponsfamily.com/launchpad accept the injected Robinhood Chain wallet?

Detection, then the connect flow, then whether the launch button stops saying
"Connect wallet". Broadcasting is blocked (allow_send=False), so this cannot
spend anything.

  py test_rhwallet.py
  py test_rhwallet.py --headful
"""
import argparse
import asyncio

from envcfg import load_env
from rhwallet import account, CHAIN_ID, RPC
from rhprovider import attach

URL = "https://www.ponsfamily.com/launchpad/create"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--connect", action="store_true")
    a = ap.parse_args()

    from playwright.async_api import async_playwright

    env = load_env()
    acct = account(env)
    rpc = env.get("FLY_RH_RPC", RPC)
    print(f"wallet {acct.address}")
    print(f"rpc    {rpc}\n")

    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=not a.headful)
        page = await b.new_page(viewport={"width": 1280, "height": 720})
        page.on("pageerror", lambda e: print(f"  [pageerror] {str(e)[:150]}"))
        page.on("console", lambda m: m.type == "error"
                and print(f"  [console] {m.text[:150]}"))

        await attach(page, acct, rpc, CHAIN_ID, allow_send=False,
                     log=lambda m: print(f"  [wallet] {m}"))
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        det = await page.evaluate("""() => ({
            ethereum: !!window.ethereum,
            isMetaMask: !!(window.ethereum && window.ethereum.isMetaMask),
            selected: window.ethereum && window.ethereum.selectedAddress,
            chainId: window.ethereum && window.ethereum.chainId })""")
        print("detection:", det)

        try:
            probe = await page.evaluate("""async () => {
                const a = await window.ethereum.request({method:'eth_requestAccounts'});
                const c = await window.ethereum.request({method:'eth_chainId'});
                const s = await window.ethereum.request(
                    {method:'personal_sign', params:['0x666c79', a[0]]});
                return {accounts:a, chainId:c, sigLen:s.length}; }""")
            print("provider round-trip:", probe)
        except Exception as e:
            print("provider round-trip FAILED:", str(e)[:200])

        n6963 = await page.evaluate("""async () => {
            let n = 0;
            const h = () => n++;
            window.addEventListener('eip6963:announceProvider', h);
            window.dispatchEvent(new Event('eip6963:requestProvider'));
            await new Promise(r => setTimeout(r, 400));
            window.removeEventListener('eip6963:announceProvider', h);
            return n; }""")
        print(f"eip-6963 announcements heard: {n6963}")

        btns = await page.evaluate("""() => [...document.querySelectorAll('button')]
            .map(b => (b.textContent||'').trim())
            .filter(t => /connect|launch|create/i.test(t)).slice(0,10)""")
        print("buttons before connect:", btns)

        if a.connect:
            print("\nattempting connect ...")
            for label in ("Connect wallet", "Connect"):
                try:
                    await page.get_by_role("button", name=label).first.click(timeout=4000)
                    print(f"  clicked '{label}'")
                    await page.wait_for_timeout(3000)
                    break
                except Exception:
                    continue
            body = await page.evaluate("() => document.body.innerText.slice(0,600)")
            print("  modal text:\n   " +
                  body.encode("ascii", "replace").decode().replace("\n", "\n   ")[:600])
            opts = await page.evaluate("""() => [...document.querySelectorAll(
                'button,[role=button],li')].map(b=>(b.textContent||'').trim())
                .filter(t=>t && t.length<40).slice(0,30)""")
            print("  options:", sorted(set(opts))[:24])

        await page.screenshot(path="build/rh_wallet_test.png")
        print("\nscreenshot -> build/rh_wallet_test.png")
        await b.close()


if __name__ == "__main__":
    asyncio.run(main())
