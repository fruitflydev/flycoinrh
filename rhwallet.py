"""
The fly's Robinhood Chain wallet.

Robinhood Chain is an Arbitrum Nitro L2 - EVM, secp256k1 keys, gas paid in ETH
- so none of the Solana wallet carries over. This is a separate keypair in a
separate .env entry, and the two can never be confused for each other.

Same rule as the Solana side: the secret is generated here, written straight
into .env (gitignored), and never printed. Only the address is shown.

  py rhwallet.py new       create the fly's Robinhood Chain wallet
  py rhwallet.py show      address, balance, chain
"""
import sys
from pathlib import Path

import requests
from eth_account import Account

from envcfg import load_env

ROOT = Path(__file__).parent
ENV = ROOT / ".env"

CHAIN_ID = 4663
RPC = "https://rpc.mainnet.chain.robinhood.com"
EXPLORER = "https://robinhoodchain.blockscout.com"
LAUNCH_FEE_ETH = 0.0005          # the fee ponsfamily.com/launchpad quotes


def _write_env_key(key, value):
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    out, done = [], False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    ENV.write_text("\n".join(out) + "\n")


def rpc_call(method, params=None, rpc=None):
    r = requests.post(rpc or RPC, json={"jsonrpc": "2.0", "id": 1,
                                        "method": method,
                                        "params": params or []}, timeout=25)
    return r.json()


def new():
    env = load_env()
    if env.get("FLY_RH_SECRET"):
        print("FLY_RH_SECRET is already set in .env - refusing to overwrite.")
        print("Delete that line by hand first if you really want a new wallet.")
        return
    Account.enable_unaudited_hdwallet_features()
    acct = Account.create()
    _write_env_key("FLY_RH_SECRET", acct.key.hex())
    _write_env_key("FLY_RH_RPC", RPC)
    _write_env_key("FLY_RH_LIVE", "0")
    print("created the fly's Robinhood Chain wallet\n")
    print(f"  address   {acct.address}")
    print(f"  chain     Robinhood Chain (id {CHAIN_ID})")
    print(f"  secret    written to .env (gitignored), not shown here")
    print(f"\nFund it with ETH on Robinhood Chain, then:  py rhwallet.py show")
    print(f"The launchpad quotes a {LAUNCH_FEE_ETH} ETH launch fee, plus gas.")
    print(f"Explorer: {EXPLORER}/address/{acct.address}")


def account(env=None):
    env = env or load_env()
    sec = env.get("FLY_RH_SECRET")
    if not sec:
        print("no Robinhood Chain wallet yet - run: py rhwallet.py new")
        sys.exit(1)
    return Account.from_key(sec)


def balance(env=None, quiet=False):
    env = env or load_env()
    acct = account(env)
    rpc = env.get("FLY_RH_RPC", RPC)
    wei = 0
    chain = None
    try:
        r = rpc_call("eth_getBalance", [acct.address, "latest"], rpc)
        wei = int(r.get("result", "0x0"), 16)
        c = rpc_call("eth_chainId", [], rpc)
        chain = int(c.get("result", "0x0"), 16)
    except Exception as e:
        if not quiet:
            print(f"  rpc error: {str(e)[:120]}")
    eth = wei / 1e18
    if not quiet:
        print(f"  address   {acct.address}")
        print(f"  chain id  {chain}  ({'ok' if chain == CHAIN_ID else 'UNEXPECTED'})")
        print(f"  balance   {eth:.6f} ETH")
        print(f"  live      {'ARMED' if env.get('FLY_RH_LIVE') == '1' else 'no (FLY_RH_LIVE=0)'}")
        if eth < LAUNCH_FEE_ETH:
            print(f"\n  not enough to launch - the launchpad fee alone is "
                  f"{LAUNCH_FEE_ETH} ETH, plus gas")
        print(f"\n  {EXPLORER}/address/{acct.address}")
    return eth


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "new":
        new()
    elif cmd in ("show", "balance"):
        balance()
    else:
        print(__doc__)
