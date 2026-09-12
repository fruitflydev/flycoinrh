"""
The pons launchpad on Robinhood Chain: the board of coins, each coin's market,
exact quotes, and the unsigned transactions that trade it.

Nothing here signs or sends. It reads the chain and builds calldata.

Verified before this was written (2026-09-12), by simulation or receipt:
  * the listing API https://www.ponsfamily.com/api/pons-launches returns the
    active (bonding-curve) coins with name, symbol, logo, market cap,
    graduation progress and quote asset;
  * factory getLaunchedToken(token) gives the curve, pair token, pool fee,
    tick spacing, creator tax and phase: 0 on the curve, 1 settling into its
    pool (no trading), 2 trading in its pool, 3 closed (no market);
  * curve buy(quoteIn, minTokensOut, recipient), selector 0x59a87bc1, is
    payable; on a native-quote curve msg.value must equal quoteIn exactly, and
    an ERC-20 quote is pulled by transferFrom after approve(curve);
  * curve sell(tokensIn, minQuoteOut, recipient), selector 0xd04c6983, pulls
    the token by transferFrom after approve(curve);
  * curve quotes are pons's own client arithmetic, reproduced below;
  * a graduated coin trades in a Uniswap v4 pool through pons's modified
    Universal Router: execute(0x10, [actions 0x060c0f, swap params that carry
    an extra minHopPriceX36, SETTLE_ALL, TAKE_ALL], deadline). The pons quoter
    quotes it exactly, hook fee included. An ERC-20 input needs
    ERC20.approve(Permit2) and Permit2.approve(token, router, amount, expiry);
  * a single trade is not capped at 3% price impact: a 1 ETH curve buy and a
    1%-of-supply pool sell both simulate without reverting.
"""
import os
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum

import requests
from eth_abi import decode, encode
from eth_utils import keccak, to_checksum_address

RPC = os.environ.get("FLY_RH_RPC", "https://rpc.mainnet.chain.robinhood.com")
CHAIN_ID = 4663
LISTING = "https://www.ponsfamily.com/api/pons-launches"
IPFS = "https://www.ponsfamily.com/api/ipfs/content/{cid}?variant=card"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 flybrain-backroom/1.0")

ZERO = "0x0000000000000000000000000000000000000000"
FACTORY = to_checksum_address("0x7eD598BcEf8bd9Edd8C97A195C6d13f40801EC7e")
MEME_HOOK = to_checksum_address("0xE5e702641Ea86F4ae6cC3cDaeD2B886f976Be044")
POOL_MANAGER = to_checksum_address("0x8366a39CC670B4001A1121B8F6A443A643e40951")
UNIVERSAL_ROUTER = to_checksum_address("0x8876789976dEcBfCbBbe364623C63652db8C0904")
V4_QUOTER = to_checksum_address("0xe202BB8dd524eE9C5E679e5B5809f7A373a982Ef")
PERMIT2 = to_checksum_address("0x000000000022D473030F116dDEE9F6B43aC78BA3")
POOL_REGISTERED_TOPIC = "0x01bf263a1db1652580721573296e1a1fa70b3d4c87f61d02a69c4e1109d2d573"
# factory event for every launch, from the pons client ABI; lets the backroom
# find coins from the chain when the listing API is slow or down
TOKEN_LAUNCHED_SIG = "TokenLaunched(address,address,address,address,uint256,uint256)"
TOKEN_LAUNCHED_TOPIC = "0x" + keccak(text=TOKEN_LAUNCHED_SIG).hex()

SORTS = ("recentBuys", "newest", "oldest", "marketCap", "volume")
AGES = ("all", "24h", "7d")
BPS = 10_000
UINT160_MAX = 2 ** 160 - 1

LAUNCH_TUPLE = "(address,address,address,address,address,uint256,uint24,int24,uint16,bool,uint8,uint256,uint256,uint256,bool)"
POOL_KEY = "(address,address,uint24,int24,address)"
QUOTER_SIG = "quoteExactInputSingle(((address,address,uint24,int24,address),bool,uint128,bytes))"
QUOTE_PARAMS = "((address,address,uint24,int24,address),bool,uint128,bytes)"
SWAP_PARAMS = "((address,address,uint24,int24,address),bool,uint128,uint128,uint256,bytes)"
V4_SWAP = 0x10
ACTIONS_EXACT_IN_SINGLE = bytes([0x06, 0x0C, 0x0F])     # SWAP_EXACT_IN_SINGLE, SETTLE_ALL, TAKE_ALL


def selector(signature):
    return "0x" + keccak(text=signature)[:4].hex()


def is_zero(address):
    return int(address, 16) == 0


# --------------------------------------------------------------------------
# the board
# --------------------------------------------------------------------------
@dataclass
class Coin:
    token: str
    name: str
    symbol: str
    description: str
    logo_url: str
    market_cap_usd: float | None
    price_usd: float | None
    progress_pct: float | None
    quote_symbol: str
    quote_address: str
    quote_decimals: int
    quote_is_native: bool
    venue: str
    graduated: bool
    launched_at: str | None
    latest_buy_at: str | None
    raw: dict = field(default_factory=dict, repr=False)


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def logo_url(logo):
    if not logo:
        return ""
    m = re.match(r"ipfs://(.+)", logo)
    return IPFS.format(cid=m.group(1)) if m else logo


def parse_item(it):
    q = it.get("quoteAsset") or {}       # some listing items carry no quote asset at all
    return Coin(
        token=to_checksum_address(it["token"]),
        name=str(it.get("name") or ""),
        symbol=str(it.get("symbol") or ""),
        description=str(it.get("description") or ""),
        logo_url=logo_url(it.get("logo")),
        market_cap_usd=_num(it.get("marketCapUsd")),
        price_usd=_num(it.get("priceUsd")),
        progress_pct=_num(it.get("graduationProgressPct")),
        quote_symbol=str(q.get("symbol") or ""),
        quote_address=str(q.get("address") or ZERO),
        quote_decimals=int(q.get("decimals") or 18),
        quote_is_native=bool(q.get("isNative")),
        venue=str(it.get("venue") or ""),
        graduated=bool(it.get("graduated")),
        launched_at=it.get("launchedAt"),
        latest_buy_at=it.get("latestBuyAt"),
        raw=it,
    )


def fetch_board(sort="recentBuys", age="all", page_size=20, page=1, session=requests, timeout=40,
                attempts=3, backoff_s=5.0, sleep=time.sleep):
    """
    The board as the pons explore page loads it: active (curve) coins, one page.
    The pons backend has run in a degraded mode; a slow answer is retried with
    backoff, and after `attempts` it raises RuntimeError rather than a raw timeout.
    """
    if sort not in SORTS or age not in AGES:
        raise ValueError(f"sort must be one of {SORTS}, age one of {AGES}")
    last = None
    for i in range(attempts):
        try:
            r = session.get(LISTING, timeout=timeout, headers={"User-Agent": UA, "Accept": "application/json"}, params={
                "explore": 1, "sort": sort, "age": age, "page": page, "pageSize": page_size,
                "graduatedPage": 1, "graduatedPageSize": 6, "includeGraduated": 0, "version": "all", "v": 22})
            r.raise_for_status()
            body = r.json()
            items = ((body.get("active") or {}).get("items") or [])
            return [parse_item(it) for it in items if it.get("token")]
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if i + 1 < attempts:
                sleep(backoff_s * (2 ** i))
    raise RuntimeError(f"pons listing unavailable after {attempts} attempts: {str(last)[:160]}")


# --------------------------------------------------------------------------
# the chain
# --------------------------------------------------------------------------
class Chain:
    """Minimal JSON-RPC client. Read-only: eth_call, eth_getBalance, eth_blockNumber, eth_getLogs."""

    def __init__(self, rpc=RPC, session=requests, timeout=25):
        self.rpc, self.session, self.timeout = rpc, session, timeout

    def request(self, method, params):
        r = self.session.post(self.rpc, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                              timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise RuntimeError(f"{method}: {body['error']}")
        return body.get("result")

    def block_number(self):
        return int(self.request("eth_blockNumber", []), 16)

    def call(self, to, data, value=0, sender=None, block="latest"):
        tx = {"to": to, "data": data}
        if value:
            tx["value"] = hex(int(value))
        if sender:
            tx["from"] = sender
        return self.request("eth_call", [tx, hex(block) if isinstance(block, int) else block])

    def read(self, to, signature, out_types, args=(), arg_types=(), block="latest"):
        data = selector(signature) + (encode(list(arg_types), list(args)).hex() if arg_types else "")
        raw = self.call(to, data, block=block)
        if not raw or raw == "0x":
            raise RuntimeError(f"{signature} on {to} returned nothing")
        vals = decode(list(out_types), bytes.fromhex(raw[2:]))
        return vals[0] if len(vals) == 1 else vals


# --------------------------------------------------------------------------
# a coin's market: phase, curve state, pool key
# --------------------------------------------------------------------------
class Phase(IntEnum):
    CURVE = 0       # NotGraduated: trades on the bonding curve
    SETTLING = 1    # Swept: the curve filled and the pool is being seeded; no trading
    POOL = 2        # PoolCreated: trades in its Uniswap v4 pool
    CLOSED = 3      # Rescued: reserves were returned; no market


@dataclass
class Launch:
    token: str
    curve: str
    deployer: str
    creator_fee_recipient: str
    pair_token: str
    graduation_threshold: int
    pool_fee: int
    tick_spacing: int
    creator_tax_bps: int
    buyback_enabled: bool
    phase: Phase
    exists: bool

    @property
    def native_quote(self):
        return is_zero(self.pair_token)

    @property
    def tradable(self):
        return self.exists and self.phase in (Phase.CURVE, Phase.POOL)


def launched(chain, token, block="latest"):
    t = chain.read(FACTORY, "getLaunchedToken(address)", [LAUNCH_TUPLE], [to_checksum_address(token)], ["address"],
                   block=block)
    return Launch(token=to_checksum_address(t[0]), curve=to_checksum_address(t[1]), deployer=to_checksum_address(t[2]),
                  creator_fee_recipient=to_checksum_address(t[3]), pair_token=to_checksum_address(t[4]),
                  graduation_threshold=int(t[5]), pool_fee=int(t[6]), tick_spacing=int(t[7]),
                  creator_tax_bps=int(t[8]), buyback_enabled=bool(t[9]), phase=Phase(int(t[10])), exists=bool(t[14]))


@dataclass
class CurveState:
    curve: str
    token: str
    pair_token: str
    native_quote: bool
    graduated: bool
    ready_to_graduate: bool
    quote_reserve: int
    token_reserve: int
    sellable_tokens: int
    reserved_tokens: int
    real_quote_reserve: int
    phantom_quote: int
    graduation_threshold: int
    fee_bps: int
    creator_tax_bps: int


@dataclass
class LaunchEvent:
    token: str
    curve: str
    deployer: str
    pair_token: str
    launch_config_id: int
    graduation_threshold: int
    block: int


def recent_launches(chain, from_block, to_block, factory=FACTORY, window=20_000):
    """
    Every TokenLaunched the factory emitted between two blocks, oldest first,
    read in windows the public RPC accepts. Needs no listing API.
    """
    out = []
    start = int(from_block)
    while start <= int(to_block):
        end = min(int(to_block), start + window - 1)
        logs = chain.request("eth_getLogs", [{"address": factory, "fromBlock": hex(start), "toBlock": hex(end),
                                              "topics": [TOKEN_LAUNCHED_TOPIC]}]) or []
        for lg in logs:
            t = lg["topics"]
            pair, cfg, threshold = decode(["address", "uint256", "uint256"], bytes.fromhex(lg["data"][2:]))
            out.append(LaunchEvent(token=to_checksum_address("0x" + t[1][-40:]), curve=to_checksum_address("0x" + t[2][-40:]),
                                   deployer=to_checksum_address("0x" + t[3][-40:]), pair_token=to_checksum_address(pair),
                                   launch_config_id=int(cfg), graduation_threshold=int(threshold),
                                   block=int(lg["blockNumber"], 16)))
        start = end + 1
    return out


def curve_of(chain, token, block="latest"):
    return to_checksum_address(chain.read(token, "curve()", ["address"], block=block))


def curve_state(chain, curve, block="latest"):
    """Everything pons's client reads before quoting, pinned to one block so the reserves agree."""
    if is_zero(curve):
        # the listing includes coins the v2 factory never launched; their record has exists=false and curve 0x0
        raise ValueError("no curve: this token has no pons v2 launch record")
    r = lambda sig, types: chain.read(curve, sig, types, block=block)  # noqa: E731
    q, t = r("getReserves()", ["uint256", "uint256"])
    return CurveState(
        curve=to_checksum_address(curve), token=to_checksum_address(r("token()", ["address"])),
        pair_token=to_checksum_address(r("pairToken()", ["address"])),
        native_quote=bool(r("isNativeQuote()", ["bool"])), graduated=bool(r("graduated()", ["bool"])),
        ready_to_graduate=bool(r("readyToGraduate()", ["bool"])), quote_reserve=int(q), token_reserve=int(t),
        sellable_tokens=int(r("sellableTokens()", ["uint256"])), reserved_tokens=int(r("reservedTokens()", ["uint256"])),
        real_quote_reserve=int(r("realQuoteReserve()", ["uint256"])), phantom_quote=int(r("phantomQuote()", ["uint256"])),
        graduation_threshold=int(r("graduationThreshold()", ["uint256"])), fee_bps=int(r("feeBps()", ["uint256"])),
        creator_tax_bps=int(r("creatorTaxBps()", ["uint256"])))


# --------------------------------------------------------------------------
# exact curve quotes: pons's client arithmetic
# --------------------------------------------------------------------------
def cp_out(amount_in, reserve_in, reserve_out, fee_bps=0):
    """Constant-product output, integer arithmetic, as pons computes it."""
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0
    a = amount_in * (BPS - fee_bps)
    return a * reserve_out // (BPS * reserve_in + a)


def quote_curve_buy(state, quote_in):
    """
    Tokens out for `quote_in` of the quote asset. Fee and creator tax come off
    the input. A buy larger than the sellable supply takes all of it, spends
    only what that costs and refunds the rest.
    """
    zero = {"tokens_out": 0, "spent": 0, "refund": 0, "fee": 0, "tax": 0, "completes_curve": False}
    if quote_in <= 0 or state.graduated or state.sellable_tokens <= 0:
        return zero
    spent = quote_in
    fee = spent * state.fee_bps // BPS
    tax = spent * state.creator_tax_bps // BPS
    out = cp_out(spent - fee - tax, state.quote_reserve, state.token_reserve)
    if out > state.sellable_tokens:
        out = state.sellable_tokens
        u, r, t = state.sellable_tokens, state.quote_reserve, state.token_reserve
        net = 0 if (u <= 0 or r <= 0 or t <= u) else u * r * BPS // ((t - u) * BPS) + 1
        keep = BPS - state.fee_bps - state.creator_tax_bps
        gross = (net * BPS + keep - 1) // keep
        spent = min(gross, quote_in)
        fee = spent * state.fee_bps // BPS
        tax = spent * state.creator_tax_bps // BPS
    return {"tokens_out": out, "spent": spent, "refund": quote_in - spent, "fee": fee, "tax": tax,
            "completes_curve": out >= state.sellable_tokens}


def quote_curve_sell(state, tokens_in):
    """Quote asset out for `tokens_in`. Fee and creator tax come off the output. Zero once graduation is due."""
    if tokens_in <= 0 or state.graduated or state.ready_to_graduate:
        return {"quote_out": 0, "fee": 0, "tax": 0}
    gross = cp_out(tokens_in, state.token_reserve, state.quote_reserve)
    fee = gross * state.fee_bps // BPS
    tax = gross * state.creator_tax_bps // BPS
    return {"quote_out": gross - fee - tax, "fee": fee, "tax": tax}


def simulate_curve_buy(chain, state, quote_in, sender, block="latest"):
    """The real buy() by eth_call from `sender` (native quote only: it must carry the value). Tokens out."""
    if not state.native_quote:
        raise NotImplementedError("an ERC-20-quoted buy needs the sender to hold and approve the quote asset")
    tx = curve_buy_tx(state, quote_in, 0, sender)
    raw = chain.call(tx["to"], tx["data"], value=tx["value"], sender=sender, block=block)
    if not raw or raw == "0x":
        raise RuntimeError("buy simulation returned nothing")
    return int(decode(["uint256"], bytes.fromhex(raw[2:]))[0])


# --------------------------------------------------------------------------
# unsigned transactions
# --------------------------------------------------------------------------
def curve_buy_tx(state, quote_in, min_tokens_out, recipient):
    data = selector("buy(uint256,uint256,address)") + encode(
        ["uint256", "uint256", "address"], [int(quote_in), int(min_tokens_out), to_checksum_address(recipient)]).hex()
    return {"to": state.curve, "value": int(quote_in) if state.native_quote else 0, "data": data}


def curve_sell_tx(state, tokens_in, min_quote_out, recipient):
    data = selector("sell(uint256,uint256,address)") + encode(
        ["uint256", "uint256", "address"], [int(tokens_in), int(min_quote_out), to_checksum_address(recipient)]).hex()
    return {"to": state.curve, "value": 0, "data": data}


def erc20_approve_tx(token, spender, amount):
    data = selector("approve(address,uint256)") + encode(
        ["address", "uint256"], [to_checksum_address(spender), int(amount)]).hex()
    return {"to": to_checksum_address(token), "value": 0, "data": data}


def erc20_allowance(chain, token, owner, spender, block="latest"):
    return int(chain.read(token, "allowance(address,address)", ["uint256"],
                          [to_checksum_address(owner), to_checksum_address(spender)], ["address", "address"], block=block))


def erc20_balance(chain, token, owner, block="latest"):
    return int(chain.read(token, "balanceOf(address)", ["uint256"], [to_checksum_address(owner)], ["address"], block=block))


def erc20_decimals(chain, token, block="latest"):
    """How many raw units make one whole coin. Every pons launch seen so far is 18."""
    return int(chain.read(token, "decimals()", ["uint8"], block=block))


def erc20_text(chain, token, which, block="latest"):
    """
    symbol() or name() as a string. Some ERC-20s answer with a bytes32 instead,
    which this cannot decode; the caller gets "" rather than an exception,
    because a coin's label is decoration and must never stop a quote.
    """
    if which not in ("symbol", "name"):
        raise ValueError("which must be 'symbol' or 'name'")
    try:
        return str(chain.read(token, f"{which}()", ["string"], block=block))
    except Exception:                                     # noqa: BLE001 - a label is not worth failing on
        return ""


def gas_price(chain):
    """eth_gasPrice in wei. Robinhood Chain has been running at about 0.365 gwei."""
    return int(chain.request("eth_gasPrice", []), 16)


# --------------------------------------------------------------------------
# graduated coins: the v4 pool
# --------------------------------------------------------------------------
def pool_key(launch, hook=MEME_HOOK):
    """(currency0, currency1, fee, tickSpacing, hooks), currency0 the lower address; native ETH is 0x0."""
    a, b = sorted([launch.token.lower(), launch.pair_token.lower()], key=lambda x: int(x, 16))
    return (to_checksum_address(a), to_checksum_address(b), int(launch.pool_fee), int(launch.tick_spacing),
            to_checksum_address(hook))


def pool_id(key):
    return "0x" + keccak(encode(["address", "address", "uint24", "int24", "address"], list(key))).hex()


def zero_for_one(key, token, side):
    """Swap direction: selling the coin moves it out of its slot; buying moves the quote in."""
    token_is_c0 = token.lower() == key[0].lower()
    if side == "sell":
        return token_is_c0
    if side == "buy":
        return not token_is_c0
    raise ValueError("side must be 'buy' or 'sell'")


def quote_pool(chain, key, zfo, amount_in, block="latest"):
    """(amount_out, gas_estimate) from the pons v4 quoter; the hook's cut is already taken."""
    data = selector(QUOTER_SIG) + encode([QUOTE_PARAMS], [(key, bool(zfo), int(amount_in), b"")]).hex()
    raw = chain.call(V4_QUOTER, data, block=block)
    if not raw or raw == "0x":
        raise RuntimeError("quoter returned nothing")
    out, gas = decode(["uint256", "uint256"], bytes.fromhex(raw[2:]))
    return int(out), int(gas)


def quote_pool_side(chain, key, token, side, amount_in, block="latest"):
    """
    quote_pool for one side of a coin's own pool, with the direction worked out
    from the key: a buy puts the quote asset in, a sell puts the coin in.
    """
    return quote_pool(chain, key, zero_for_one(key, token, side), amount_in, block=block)


def pool_swap_tx(key, token, side, amount_in, min_out, deadline):
    """The unsigned router call for one exact-input swap in a coin's pool."""
    zfo = zero_for_one(key, token, side)
    currency_in, currency_out = (key[0], key[1]) if zfo else (key[1], key[0])
    swap = encode([SWAP_PARAMS], [(key, zfo, int(amount_in), int(min_out), 0, b"")])
    settle = encode(["address", "uint256"], [currency_in, int(amount_in)])
    take = encode(["address", "uint256"], [currency_out, int(min_out)])
    v4 = encode(["bytes", "bytes[]"], [ACTIONS_EXACT_IN_SINGLE, [swap, settle, take]])
    data = selector("execute(bytes,bytes[],uint256)") + encode(
        ["bytes", "bytes[]", "uint256"], [bytes([V4_SWAP]), [v4], int(deadline)]).hex()
    return {"to": UNIVERSAL_ROUTER, "value": int(amount_in) if is_zero(currency_in) else 0, "data": data}


def permit2_allowance(chain, owner, token, spender=UNIVERSAL_ROUTER, block="latest"):
    """(amount, expiration, nonce) Permit2 has granted `spender` over `owner`'s `token`."""
    a, e, n = chain.read(PERMIT2, "allowance(address,address,address)", ["uint160", "uint48", "uint48"],
                         [to_checksum_address(owner), to_checksum_address(token), to_checksum_address(spender)],
                         ["address", "address", "address"], block=block)
    return int(a), int(e), int(n)


def permit2_approve_tx(token, expiration, spender=UNIVERSAL_ROUTER, amount=UINT160_MAX):
    data = selector("approve(address,address,uint160,uint48)") + encode(
        ["address", "address", "uint160", "uint48"],
        [to_checksum_address(token), to_checksum_address(spender), int(amount), int(expiration)]).hex()
    return {"to": PERMIT2, "value": 0, "data": data}
