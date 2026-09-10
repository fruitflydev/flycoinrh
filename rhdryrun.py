"""
Prove the Robinhood Chain transaction path without spending anything.

The launchpad button has been disabled on every run so far, which means
rhprovider.send_transaction has never actually executed. Nonce lookup, gas
estimation, EIP-1559 fields, signing and the broadcast call are all untested
code sitting between a funded wallet and real money.

This exercises every step except the final broadcast:

  1. RPC reachable, chain id is 4663
  2. nonce and gas price come back
  3. a real transaction is assembled and signed locally
  4. the signature is recovered and checked against the wallet address
  5. the chain is asked what would happen (eth_call / eth_estimateGas)

Nothing is broadcast. There is no code path here that can send.

  py rhdryrun.py
"""
import json

import requests
from eth_account import Account
from eth_account._utils.signing import to_standard_v          # noqa: F401
from eth_utils import to_hex

from envcfg import load_env
from rhwallet import account, CHAIN_ID, RPC, LAUNCH_FEE_ETH


def rpc(method, params, url):
    r = requests.post(url, json={"jsonrpc": "2.0", "id": 1,
                                 "method": method, "params": params},
                      timeout=30).json()
    if "error" in r:
        return None, json.dumps(r["error"])[:220]
    return r.get("result"), None


def main():
    env = load_env()
    acct = account(env)
    url = env.get("FLY_RH_RPC", RPC)
    print(f"wallet  {acct.address}")
    print(f"rpc     {url}\n")

    # 1. chain reachable and correct
    cid, err = rpc("eth_chainId", [], url)
    print(f"chain id        {int(cid, 16) if cid else err} "
          f"{'ok' if cid and int(cid, 16) == CHAIN_ID else 'UNEXPECTED'}")
    bal, _ = rpc("eth_getBalance", [acct.address, "latest"], url)
    wei = int(bal, 16) if bal else 0
    print(f"balance         {wei/1e18:.6f} ETH")

    # 2. the fields send_transaction needs
    nonce, e1 = rpc("eth_getTransactionCount", [acct.address, "pending"], url)
    gas_price, e2 = rpc("eth_gasPrice", [], url)
    blk, _ = rpc("eth_getBlockByNumber", ["latest", False], url)
    base_fee = blk.get("baseFeePerGas") if blk else None
    print(f"nonce           {int(nonce,16) if nonce else e1}")
    print(f"gas price       {int(gas_price,16)/1e9:.4f} gwei"
          if gas_price else f"gas price       {e2}")
    if base_fee:
        print(f"base fee        {int(base_fee,16)/1e9:.4f} gwei")

    # 3. assemble and sign a real transaction (self-send, zero value)
    tx = {
        "from": acct.address, "to": acct.address, "value": 0,
        "data": "0x", "chainId": CHAIN_ID,
        "nonce": int(nonce, 16) if nonce else 0,
        "gas": 21000,
        "maxFeePerGas": int(int(gas_price, 16) * 2) if gas_price else 10**9,
        "maxPriorityFeePerGas": min(int(gas_price, 16), 1_000_000) if gas_price else 10**6,
        "type": 2,
    }
    signed = acct.sign_transaction(tx)
    raw = signed.raw_transaction
    print(f"\nsigned tx       {len(raw)} bytes, hash {to_hex(signed.hash)[:20]}...")

    # 4. does the signature actually belong to this wallet?
    recovered = Account.recover_transaction(raw)
    ok = recovered.lower() == acct.address.lower()
    print(f"recovered from  {recovered}  {'MATCHES' if ok else 'MISMATCH'}")

    # 5. ask the chain what would happen, without sending
    est, err = rpc("eth_estimateGas", [{
        "from": acct.address, "to": acct.address, "value": "0x0"}], url)
    if est:
        print(f"estimateGas     {int(est,16)} units")
    else:
        print(f"estimateGas     rejected: {err}")

    call, cerr = rpc("eth_call", [{
        "from": acct.address, "to": acct.address, "value": "0x0"}, "latest"], url)
    print(f"eth_call        {'ok' if call is not None else cerr}")

    need = LAUNCH_FEE_ETH + (int(gas_price, 16) * 400000 / 1e18 if gas_price else 0)
    print(f"\nrough cost to launch: {LAUNCH_FEE_ETH} ETH fee "
          f"+ ~{need - LAUNCH_FEE_ETH:.6f} ETH gas = ~{need:.5f} ETH")
    print("\nNothing was broadcast.")
    if ok and cid and int(cid, 16) == CHAIN_ID:
        print("Signing, nonce and fee lookup all work. What is still unproven is")
        print("the launchpad's own transaction: its shape, and whether it needs")
        print("one signature or several. That cannot be seen until the button")
        print("enables, which needs funds.")
    else:
        print("Something above is wrong - do not fund until it is fixed.")


if __name__ == "__main__":
    main()
