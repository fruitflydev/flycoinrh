"""
An EIP-1193 wallet for the fly, injected into the page.

The Solana side injects a Phantom-shaped provider. Robinhood Chain is EVM, so
this injects `window.ethereum` instead: eth_requestAccounts, eth_chainId,
personal_sign, eth_signTypedData_v4, eth_sendTransaction, and the usual event
emitter. It also announces itself over EIP-6963, which is how wagmi/RainbowKit
style connectors discover wallets now.

One lesson carried over from the Solana build: a *partial* discovery
implementation is worse than none. A half-built Wallet Standard object crashed
pump.fun's entire sign-in modal. So the 6963 announcement here is complete, and
if anything about it is wrong the legacy window.ethereum path still works.

The private key never enters the page. Every signature and every transaction is
built and signed in Python through `page.expose_function`; the browser only
ever sees an address and a hex signature.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

PROVIDER_JS = r"""
(() => {
  const ADDR = "__ADDR__";
  const CHAIN_HEX = "__CHAIN_HEX__";
  const RDNS = "xyz.flybrain.wallet";

  const listeners = {};
  const emit = (ev, ...a) => (listeners[ev] || []).forEach(f => {
    try { f(...a); } catch (e) {} });

  async function request(args) {
    const method = args && args.method;
    const params = (args && args.params) || [];
    switch (method) {
      case "eth_requestAccounts":
      case "eth_accounts":
        emit("connect", { chainId: CHAIN_HEX });
        emit("accountsChanged", [ADDR]);
        return [ADDR];
      case "eth_chainId":
        return CHAIN_HEX;
      case "net_version":
        return String(parseInt(CHAIN_HEX, 16));
      case "wallet_switchEthereumChain":
      case "wallet_addEthereumChain":
        emit("chainChanged", CHAIN_HEX);
        return null;
      case "wallet_requestPermissions":
        return [{ parentCapability: "eth_accounts" }];
      case "wallet_getPermissions":
        return [{ parentCapability: "eth_accounts" }];
      case "personal_sign":
      case "eth_sign":
        return await window.__flyEthSign(method, params);
      case "eth_signTypedData":
      case "eth_signTypedData_v3":
      case "eth_signTypedData_v4":
        return await window.__flyEthSignTyped(params);
      case "eth_sendTransaction":
        return await window.__flyEthSend(params[0]);
      default:
        // everything else is a plain read: let the node answer
        return await window.__flyEthRpc(method, params);
    }
  }

  const provider = {
    isMetaMask: true,
    isFlybrain: true,
    chainId: CHAIN_HEX,
    networkVersion: String(parseInt(CHAIN_HEX, 16)),
    selectedAddress: ADDR,
    request,
    // legacy shims some connectors still poke at
    enable: () => request({ method: "eth_requestAccounts" }),
    send: (m, p) => (typeof m === "string"
      ? request({ method: m, params: p })
      : request(m)),
    sendAsync: (payload, cb) => request(payload)
      .then(r => cb(null, { id: payload.id, jsonrpc: "2.0", result: r }))
      .catch(e => cb(e)),
    on(ev, f) { (listeners[ev] = listeners[ev] || []).push(f); return provider; },
    removeListener(ev, f) {
      listeners[ev] = (listeners[ev] || []).filter(x => x !== f); return provider; },
    removeAllListeners() { for (const k in listeners) delete listeners[k]; },
    isConnected: () => true,
  };

  try {
    Object.defineProperty(window, "ethereum", {
      value: provider, writable: false, configurable: true });
  } catch (e) { window.ethereum = provider; }

  // EIP-6963 - complete, or not at all
  const info = {
    uuid: "6f1d2e3c-4b5a-4c7d-8e9f-0a1b2c3d4e5f",
    name: "Flybrain",
    icon: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PHJlY3Qgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiBmaWxsPSIjNjNlNmZmIi8+PC9zdmc+",
    rdns: RDNS,
  };
  const announce = () => window.dispatchEvent(new CustomEvent(
    "eip6963:announceProvider",
    { detail: Object.freeze({ info, provider }) }));
  window.addEventListener("eip6963:requestProvider", announce);
  announce();

  window.dispatchEvent(new Event("ethereum#initialized"));
})();
"""


async def attach(page, acct, rpc, chain_id, allow_send=False, on_send=None,
                 log=None):
    """
    Install the EVM provider on `page`. Call before navigation.

    allow_send=False means eth_sendTransaction refuses rather than broadcasting,
    so the page cannot spend anything without an explicit opt-in.
    """
    import requests
    from eth_account.messages import encode_defunct

    def _say(m):
        if log:
            log(m)

    async def eth_sign(method, params):
        # personal_sign is (message, address); eth_sign is (address, message)
        raw = params[0] if method == "personal_sign" else params[1]
        if isinstance(raw, str) and raw.startswith("0x"):
            data = bytes.fromhex(raw[2:])
        else:
            data = str(raw).encode()
        _say(f"eth sign request ({method}, {len(data)} bytes)")
        sig = acct.sign_message(encode_defunct(primitive=data))
        return sig.signature.hex()

    async def eth_sign_typed(params):
        from eth_account.messages import encode_typed_data
        payload = params[1] if len(params) > 1 else params[0]
        if isinstance(payload, str):
            payload = json.loads(payload)
        _say("eth signTypedData request")
        sig = acct.sign_message(encode_typed_data(full_message=payload))
        return sig.signature.hex()

    def _rpc(method, prms):
        r = requests.post(rpc, json={"jsonrpc": "2.0", "id": 1,
                                     "method": method, "params": prms},
                          timeout=25).json()
        if "error" in r:
            raise RuntimeError(json.dumps(r["error"])[:200])
        return r.get("result")

    async def eth_rpc(method, prms):
        return _rpc(method, prms)

    async def eth_send(tx):
        if on_send:
            return await on_send(tx)
        if not allow_send:
            raise RuntimeError("eth_sendTransaction blocked (allow_send=False)")
        return send_transaction(acct, tx, rpc, chain_id, say=_say)

    await page.expose_function("__flyEthSign", eth_sign)
    await page.expose_function("__flyEthSignTyped", eth_sign_typed)
    await page.expose_function("__flyEthRpc", eth_rpc)
    await page.expose_function("__flyEthSend", eth_send)
    await page.add_init_script(
        PROVIDER_JS.replace("__ADDR__", acct.address)
                   .replace("__CHAIN_HEX__", hex(chain_id)))
    return acct.address


def send_transaction(acct, tx, rpc, chain_id, say=None):
    """Fill, sign and broadcast a transaction the page asked for."""
    import requests

    def rpc_call(method, prms):
        r = requests.post(rpc, json={"jsonrpc": "2.0", "id": 1,
                                     "method": method, "params": prms},
                          timeout=30).json()
        if "error" in r:
            raise RuntimeError(json.dumps(r["error"])[:300])
        return r.get("result")

    def as_int(v, default=0):
        if v is None:
            return default
        if isinstance(v, int):
            return v
        return int(v, 16) if str(v).startswith("0x") else int(v)

    body = {
        "from": acct.address,
        "to": tx.get("to"),
        "value": as_int(tx.get("value"), 0),
        "data": tx.get("data") or "0x",
        "chainId": chain_id,
        "nonce": as_int(rpc_call("eth_getTransactionCount",
                                 [acct.address, "pending"])),
    }
    if tx.get("gas"):
        body["gas"] = as_int(tx["gas"])
    else:
        est = rpc_call("eth_estimateGas", [{
            "from": acct.address, "to": body["to"],
            "value": hex(body["value"]), "data": body["data"]}])
        body["gas"] = int(as_int(est) * 1.25)

    # Nitro chains price in EIP-1559 terms
    try:
        base = as_int(rpc_call("eth_gasPrice", []))
        body["maxFeePerGas"] = int(base * 2)
        body["maxPriorityFeePerGas"] = min(int(base), 1_000_000)
        body["type"] = 2
    except Exception:
        body["gasPrice"] = as_int(rpc_call("eth_gasPrice", []))

    if say:
        say(f"signing tx to {str(body['to'])[:12]}... "
            f"value {body['value']/1e18:.6f} ETH gas {body['gas']}")
    signed = acct.sign_transaction(body)
    h = rpc_call("eth_sendRawTransaction", [signed.raw_transaction.hex()])
    if say:
        say(f"broadcast {h}")
    return h
