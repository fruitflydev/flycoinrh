"""
pons.py without the network: listing and chain are faked; every constant the
exit-path scout verified on chain is pinned here.

  py -m unittest test_pons -v
"""
import unittest

from eth_abi import decode, encode
from eth_utils import keccak

import pons

TOKEN = "0x94fa961993480f2786d91d756476a766432d5ab7"
CURVE = "0xfca246b2d47ce1f7ae255f5c5eda8374af09b2b6"
ME = "0x6ce4085EfB52a6eBDb7d6989beb8860847f4b42A"
FLY = "0x4eb990547bce4a982432ca88cf5fae7eed1a2d35"
GOOGL = "0x2e0847e8910a9732eb3fb1bb4b70a580adad4fe3"

ITEM = {"version": "v2", "token": TOKEN, "name": "erebus", "symbol": "EREBUS",
        "logo": "ipfs://QmPJei7bQBKtKk6o3rsu8vbN65v5wx4kaFuftoxRXmRRcG",
        "description": "Infrastructure for private agent-to-agent transactions",
        "priceUsd": 3.0126936459020567e-05, "marketCapUsd": 30126.936459020566, "graduated": False,
        "graduationProgressPct": 65.73, "latestBuyAt": "2026-09-11T18:17:58.000Z",
        "launchedAt": "2026-09-11T18:15:31.000Z", "venue": "curve",
        "quoteAsset": {"address": pons.ZERO, "symbol": "ETH", "name": "Ether", "decimals": 18, "isNative": True}}


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeSession:
    """GET answers the listing; POST answers eth_call by selector and eth_blockNumber."""

    def __init__(self, answers=None):
        self.gets, self.posts = [], []
        self.answers = answers or {}

    def get(self, url, **kw):
        self.gets.append((url, kw))
        return FakeResponse({"active": {"items": [ITEM, {"name": "no token"}, {"token": TOKEN.upper().replace("0X", "0x")}]},
                             "graduated": {"items": []}})

    def post(self, url, json=None, **kw):
        self.posts.append(json)
        if json["method"] == "eth_blockNumber":
            return FakeResponse({"jsonrpc": "2.0", "id": 1, "result": hex(60_500_000)})
        if json["method"] == "eth_gasPrice":
            return FakeResponse({"jsonrpc": "2.0", "id": 1,
                                 "result": self.answers.get("eth_gasPrice", hex(365_000_000))})
        tx = json["params"][0]
        ans = self.answers.get(tx["data"][:10])
        if callable(ans):
            ans = ans(tx)
        return FakeResponse({"jsonrpc": "2.0", "id": 1, "result": ans})


def ret(types, vals):
    return "0x" + encode(types, vals).hex()


def state(**over):
    base = dict(curve=pons.to_checksum_address(CURVE), token=pons.to_checksum_address(TOKEN), pair_token=pons.ZERO,
                native_quote=True, graduated=False, ready_to_graduate=False,
                quote_reserve=4_646_157_047_298_115_655, token_reserve=361_589_154_842_919_513_074_871_874,
                sellable_tokens=300_000_000 * 10 ** 18, reserved_tokens=0, real_quote_reserve=2_966_157_047_298_115_655,
                phantom_quote=1_680_000_000_000_000_000, graduation_threshold=4_200_000_000_000_000_000,
                fee_bps=100, creator_tax_bps=0)
    base.update(over)
    return pons.CurveState(**base)


class Board(unittest.TestCase):
    def test_listing_request_matches_the_site(self):
        s = FakeSession()
        coins = pons.fetch_board(session=s)
        url, kw = s.gets[0]
        self.assertEqual(url, pons.LISTING)
        self.assertEqual(kw["params"]["sort"], "recentBuys")
        self.assertEqual(kw["params"]["includeGraduated"], 0)
        self.assertEqual(len(coins), 2)                 # an item without a token is skipped

    def test_item_parsed(self):
        c = pons.fetch_board(session=FakeSession())[0]
        self.assertEqual(c.token, "0x94FA961993480f2786D91d756476a766432d5Ab7")
        self.assertEqual(c.symbol, "EREBUS")
        self.assertTrue(c.quote_is_native)
        self.assertAlmostEqual(c.progress_pct, 65.73)
        self.assertEqual(c.logo_url, "https://www.ponsfamily.com/api/ipfs/content/QmPJei7bQBKtKk6o3rsu8vbN65v5wx4kaFuftoxRXmRRcG?variant=card")

    def test_item_without_quote_asset_is_not_native(self):
        c = pons.fetch_board(session=FakeSession())[1]
        self.assertFalse(c.quote_is_native)
        self.assertEqual(c.quote_symbol, "")

    def test_bad_sort_refused(self):
        with self.assertRaises(ValueError):
            pons.fetch_board(sort="pumping", session=FakeSession())

    def test_a_slow_listing_is_retried_with_backoff(self):
        class Flaky(FakeSession):
            def __init__(self):
                super().__init__()
                self.fails = 2

            def get(self, url, **kw):
                if self.fails:
                    self.fails -= 1
                    raise pons.requests.exceptions.ReadTimeout("read timed out")
                return super().get(url, **kw)

        slept = []
        coins = pons.fetch_board(session=Flaky(), attempts=3, backoff_s=5.0, sleep=slept.append)
        self.assertEqual(len(coins), 2)
        self.assertEqual(slept, [5.0, 10.0])

    def test_a_dead_listing_fails_with_a_clear_error(self):
        class Dead(FakeSession):
            def get(self, url, **kw):
                raise pons.requests.exceptions.ConnectionError("down")

        with self.assertRaises(RuntimeError) as cm:
            pons.fetch_board(session=Dead(), attempts=2, sleep=lambda s: None)
        self.assertIn("after 2 attempts", str(cm.exception))


class VerifiedConstants(unittest.TestCase):
    """Selectors, topics and ids recorded by simulation or receipt on Robinhood Chain."""

    def test_selectors(self):
        self.assertEqual(pons.selector("buy(uint256,uint256,address)"), "0x59a87bc1")
        self.assertEqual(pons.selector("sell(uint256,uint256,address)"), "0xd04c6983")
        self.assertEqual(pons.selector("execute(bytes,bytes[],uint256)"), "0x3593564c")
        self.assertEqual(pons.selector(pons.QUOTER_SIG), "0xaa9d21cb")
        self.assertEqual(pons.selector("createGraduatedPool(address)"), "0x2f53ef2f")
        self.assertEqual(pons.selector("approve(address,uint256)"), "0x095ea7b3")
        self.assertEqual(pons.selector("transfer(address,uint256)"), "0xa9059cbb")

    def test_pool_registered_topic(self):
        self.assertEqual("0x" + keccak(text="PoolRegistered(bytes32,address,address,address)").hex(), pons.POOL_REGISTERED_TOPIC)

    def test_flybrain_pool_id(self):
        launch = pons.Launch(token=pons.to_checksum_address(FLY), curve=pons.ZERO, deployer=pons.ZERO,
                             creator_fee_recipient=pons.ZERO, pair_token=pons.to_checksum_address(GOOGL),
                             graduation_threshold=0, pool_fee=0, tick_spacing=200, creator_tax_bps=100,
                             buyback_enabled=False, phase=pons.Phase.POOL, exists=True)
        key = pons.pool_key(launch)
        self.assertEqual(key[0].lower(), GOOGL)            # GOOGL is the lower address
        self.assertEqual(pons.pool_id(key), "0x6f638b29e275cab8f584df0a4c6a045922217aed8747f2459eedb034abb31ac7")
        self.assertFalse(pons.zero_for_one(key, launch.token, "sell"))   # FLY is currency1
        self.assertTrue(pons.zero_for_one(key, launch.token, "buy"))

    def test_native_pool_key_puts_eth_first(self):
        launch = pons.Launch(token=pons.to_checksum_address(TOKEN), curve=pons.ZERO, deployer=pons.ZERO,
                             creator_fee_recipient=pons.ZERO, pair_token=pons.ZERO, graduation_threshold=0,
                             pool_fee=0, tick_spacing=200, creator_tax_bps=0, buyback_enabled=False,
                             phase=pons.Phase.POOL, exists=True)
        key = pons.pool_key(launch)
        self.assertEqual(key[0], pons.ZERO)
        self.assertTrue(launch.native_quote)


class CurveMath(unittest.TestCase):
    def test_cp_out_integer_arithmetic(self):
        self.assertEqual(pons.cp_out(100, 1000, 1000), 90)
        self.assertEqual(pons.cp_out(0, 1000, 1000), 0)
        self.assertEqual(pons.cp_out(100, 0, 1000), 0)

    def test_buy_takes_fee_and_tax_off_the_input(self):
        st = state(creator_tax_bps=100)
        q = pons.quote_curve_buy(st, 10 ** 15)
        self.assertEqual(q["fee"], 10 ** 13)
        self.assertEqual(q["tax"], 10 ** 13)
        self.assertEqual(q["tokens_out"], pons.cp_out(10 ** 15 - 2 * 10 ** 13, st.quote_reserve, st.token_reserve))
        self.assertEqual(q["refund"], 0)
        self.assertFalse(q["completes_curve"])

    def test_buy_past_the_sellable_supply_takes_it_all_and_refunds(self):
        st = state(sellable_tokens=10 ** 21)
        q = pons.quote_curve_buy(st, 10 ** 18)
        self.assertTrue(q["completes_curve"])
        self.assertEqual(q["tokens_out"], st.sellable_tokens)
        self.assertLess(q["spent"], 10 ** 18)
        self.assertEqual(q["refund"], 10 ** 18 - q["spent"])
        # what was spent is enough, after fee and tax, to buy the whole remainder
        net = q["spent"] - q["fee"] - q["tax"]
        self.assertGreaterEqual(pons.cp_out(net, st.quote_reserve, st.token_reserve), st.sellable_tokens)

    def test_sell_takes_fee_and_tax_off_the_output(self):
        st = state(creator_tax_bps=100)
        q = pons.quote_curve_sell(st, 10 ** 24)
        gross = pons.cp_out(10 ** 24, st.token_reserve, st.quote_reserve)
        self.assertEqual(q["fee"] + q["tax"] + q["quote_out"], gross)
        self.assertEqual(q["fee"], gross * 100 // 10_000)

    def test_no_quotes_where_the_curve_does_not_trade(self):
        self.assertEqual(pons.quote_curve_buy(state(graduated=True), 10 ** 15)["tokens_out"], 0)
        self.assertEqual(pons.quote_curve_sell(state(graduated=True), 10 ** 20)["quote_out"], 0)
        self.assertEqual(pons.quote_curve_sell(state(ready_to_graduate=True), 10 ** 20)["quote_out"], 0)
        self.assertEqual(pons.quote_curve_buy(state(sellable_tokens=0), 10 ** 15)["tokens_out"], 0)


class Transactions(unittest.TestCase):
    def test_native_curve_buy_carries_exactly_the_input(self):
        tx = pons.curve_buy_tx(state(), 123, 456, ME)
        self.assertEqual(tx["value"], 123)
        self.assertTrue(tx["data"].startswith("0x59a87bc1"))
        self.assertEqual(decode(["uint256", "uint256", "address"], bytes.fromhex(tx["data"][10:]))[:2], (123, 456))

    def test_erc20_quote_curve_buy_carries_no_value(self):
        self.assertEqual(pons.curve_buy_tx(state(native_quote=False), 123, 0, ME)["value"], 0)

    def test_curve_sell(self):
        tx = pons.curve_sell_tx(state(), 10 ** 20, 7, ME)
        self.assertEqual(tx["value"], 0)
        self.assertTrue(tx["data"].startswith("0xd04c6983"))
        amt, min_out, rcpt = decode(["uint256", "uint256", "address"], bytes.fromhex(tx["data"][10:]))
        self.assertEqual((amt, min_out, rcpt.lower()), (10 ** 20, 7, ME.lower()))

    def test_approve(self):
        tx = pons.erc20_approve_tx(TOKEN, CURVE, 5)
        self.assertTrue(tx["data"].startswith("0x095ea7b3"))
        self.assertEqual(tx["to"].lower(), TOKEN)

    def test_pool_swap_layout_matches_the_verified_router_call(self):
        key = (pons.to_checksum_address(GOOGL), pons.to_checksum_address(FLY), 0, 200, pons.MEME_HOOK)
        tx = pons.pool_swap_tx(key, FLY, "sell", 1000 * 10 ** 18, 37 * 10 ** 15, 1_800_000_000)
        self.assertEqual(tx["to"], pons.UNIVERSAL_ROUTER)
        self.assertEqual(tx["value"], 0)
        self.assertTrue(tx["data"].startswith("0x3593564c"))
        commands, inputs, deadline = decode(["bytes", "bytes[]", "uint256"], bytes.fromhex(tx["data"][10:]))
        self.assertEqual((commands, deadline), (bytes([0x10]), 1_800_000_000))
        actions, params = decode(["bytes", "bytes[]"], inputs[0])
        self.assertEqual(actions, bytes([0x06, 0x0C, 0x0F]))
        (k, zfo, amt, min_out, min_hop, hook_data), = decode([pons.SWAP_PARAMS], params[0])
        self.assertEqual((zfo, amt, min_out, min_hop, hook_data), (False, 1000 * 10 ** 18, 37 * 10 ** 15, 0, b""))
        self.assertEqual(k[3], 200)
        cin, amount = decode(["address", "uint256"], params[1])
        cout, minimum = decode(["address", "uint256"], params[2])
        self.assertEqual((cin.lower(), amount, cout.lower(), minimum), (FLY, 1000 * 10 ** 18, GOOGL, 37 * 10 ** 15))

    def test_native_pool_buy_sends_value(self):
        key = (pons.ZERO, pons.to_checksum_address(TOKEN), 0, 200, pons.MEME_HOOK)
        tx = pons.pool_swap_tx(key, TOKEN, "buy", 10 ** 15, 1, 1_800_000_000)
        self.assertEqual(tx["value"], 10 ** 15)

    def test_permit2_approve(self):
        tx = pons.permit2_approve_tx(GOOGL, 1_800_000_000)
        self.assertEqual(tx["to"], pons.PERMIT2)
        self.assertEqual(tx["data"][:10], pons.selector("approve(address,address,uint160,uint48)"))
        token, spender, amount, exp = decode(["address", "address", "uint160", "uint48"], bytes.fromhex(tx["data"][10:]))
        self.assertEqual((token.lower(), spender.lower(), amount, exp),
                         (GOOGL, pons.UNIVERSAL_ROUTER.lower(), pons.UINT160_MAX, 1_800_000_000))


class Reads(unittest.TestCase):
    def test_launched_parses_the_factory_record(self):
        S = pons.selector
        record = (FLY, "0x5019ed9164656cbc8b7d675bc4c97b845ad7d8fe", ME, ME, GOOGL, 24_200_000_000_000_000_000,
                  0, 200, 100, False, 2, 0, 0, 0, True)
        s = FakeSession({S("getLaunchedToken(address)"): ret([pons.LAUNCH_TUPLE], [record])})
        launch = pons.launched(pons.Chain(session=s), FLY)
        self.assertEqual(launch.phase, pons.Phase.POOL)
        self.assertTrue(launch.tradable)
        self.assertFalse(launch.native_quote)
        self.assertEqual((launch.pool_fee, launch.tick_spacing, launch.creator_tax_bps), (0, 200, 100))
        self.assertEqual(s.posts[-1]["params"][0]["to"], pons.FACTORY)

    def test_settling_and_closed_are_not_tradable(self):
        for phase in (pons.Phase.SETTLING, pons.Phase.CLOSED):
            launch = pons.Launch(token=FLY, curve=pons.ZERO, deployer=pons.ZERO, creator_fee_recipient=pons.ZERO,
                                 pair_token=pons.ZERO, graduation_threshold=0, pool_fee=0, tick_spacing=200,
                                 creator_tax_bps=0, buyback_enabled=False, phase=phase, exists=True)
            self.assertFalse(launch.tradable)

    def test_a_token_the_v2_factory_never_launched_is_not_tradable(self):
        launch = pons.Launch(token=FLY, curve=pons.ZERO, deployer=pons.ZERO, creator_fee_recipient=pons.ZERO,
                             pair_token=pons.ZERO, graduation_threshold=0, pool_fee=0, tick_spacing=0,
                             creator_tax_bps=0, buyback_enabled=False, phase=pons.Phase.CURVE, exists=False)
        self.assertFalse(launch.tradable)          # phase reads 0 (curve) but there is no record

    def test_curve_state_refuses_the_zero_address(self):
        s = FakeSession()
        with self.assertRaises(ValueError):
            pons.curve_state(pons.Chain(session=s), pons.ZERO)
        self.assertEqual(s.posts, [])              # nothing was asked of the chain

    def test_reads_pin_the_block(self):
        S = pons.selector
        s = FakeSession({S("feeBps()"): ret(["uint256"], [100])})
        pons.Chain(session=s).read(CURVE, "feeBps()", ["uint256"], block=60_500_000)
        self.assertEqual(s.posts[-1]["params"][1], hex(60_500_000))

    def test_recent_launches_reads_factory_logs_in_windows(self):
        curve = "0xacbbff5622d3a4fa288b8e6b256b2172ffeeb825"

        class Logs:
            def __init__(self):
                self.filters = []

            def post(self, url, json=None, **kw):
                f = json["params"][0]
                self.filters.append(f)
                logs = []
                if f["fromBlock"] == hex(100):
                    logs = [{"topics": [pons.TOKEN_LAUNCHED_TOPIC, "0x" + "00" * 12 + TOKEN[2:], "0x" + "00" * 12 + curve[2:],
                                        "0x" + "00" * 12 + ME[2:].lower()],
                             "data": ret(["address", "uint256", "uint256"], [pons.ZERO, 3, 4_200_000_000_000_000_000]),
                             "blockNumber": hex(105)}]
                return FakeResponse({"jsonrpc": "2.0", "id": 1, "result": logs})

        s = Logs()
        events = pons.recent_launches(pons.Chain(session=s), 100, 149, window=25)
        self.assertEqual([(f["fromBlock"], f["toBlock"]) for f in s.filters], [(hex(100), hex(124)), (hex(125), hex(149))])
        self.assertTrue(all(f["topics"] == [pons.TOKEN_LAUNCHED_TOPIC] and f["address"] == pons.FACTORY for f in s.filters))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual((e.token.lower(), e.curve.lower(), e.deployer.lower()), (TOKEN, curve, ME.lower()))
        self.assertTrue(pons.is_zero(e.pair_token))
        self.assertEqual((e.launch_config_id, e.graduation_threshold, e.block), (3, 4_200_000_000_000_000_000, 105))

    def test_simulated_buy_sends_value_and_sender(self):
        S = pons.selector
        s = FakeSession({S("buy(uint256,uint256,address)"): ret(["uint256"], [39721281065908980248457])})
        out = pons.simulate_curve_buy(pons.Chain(session=s), state(), 500_000_000_000_000, ME)
        self.assertEqual(out, 39721281065908980248457)
        tx = s.posts[-1]["params"][0]
        self.assertEqual((tx["value"], tx["from"]), (hex(500_000_000_000_000), ME))


class ExecutorReads(unittest.TestCase):
    """The read-only helpers the paper executor needs on top of the quotes."""

    def test_gas_price_comes_from_the_chain(self):
        s = FakeSession({"eth_gasPrice": hex(365_000_000)})
        self.assertEqual(pons.gas_price(pons.Chain(session=s)), 365_000_000)
        self.assertEqual(s.posts[-1]["method"], "eth_gasPrice")

    def test_decimals(self):
        s = FakeSession({pons.selector("decimals()"): ret(["uint8"], [18])})
        self.assertEqual(pons.erc20_decimals(pons.Chain(session=s), TOKEN), 18)

    def test_symbol_and_name(self):
        s = FakeSession({pons.selector("symbol()"): ret(["string"], ["EREBUS"]),
                         pons.selector("name()"): ret(["string"], ["erebus"])})
        chain = pons.Chain(session=s)
        self.assertEqual(pons.erc20_text(chain, TOKEN, "symbol"), "EREBUS")
        self.assertEqual(pons.erc20_text(chain, TOKEN, "name"), "erebus")

    def test_a_label_in_another_shape_is_empty_not_an_error(self):
        s = FakeSession({pons.selector("symbol()"): ret(["bytes32"], [b"EREBUS".ljust(32, b"\x00")])})
        self.assertEqual(pons.erc20_text(pons.Chain(session=s), TOKEN, "symbol"), "")

    def test_only_symbol_and_name(self):
        with self.assertRaises(ValueError):
            pons.erc20_text(pons.Chain(session=FakeSession()), TOKEN, "totalSupply")

    def test_quote_pool_side_picks_the_direction_from_the_key(self):
        seen = {}

        def quoter(tx):
            (key, zfo, amount, _hook), = decode([pons.QUOTE_PARAMS], bytes.fromhex(tx["data"][10:]))
            seen.update(zfo=zfo, amount=amount, hook=key[4])
            return ret(["uint256", "uint256"], [amount * 2, 50_000])

        s = FakeSession({pons.selector(pons.QUOTER_SIG): quoter})
        chain = pons.Chain(session=s)
        key = (pons.ZERO, pons.to_checksum_address(TOKEN), 0, 200, pons.MEME_HOOK)
        out, gas = pons.quote_pool_side(chain, key, TOKEN, "sell", 10 ** 18)
        self.assertEqual((seen["zfo"], seen["amount"], out, gas), (False, 10 ** 18, 2 * 10 ** 18, 50_000))
        pons.quote_pool_side(chain, key, TOKEN, "buy", 5)
        self.assertEqual((seen["zfo"], seen["amount"]), (True, 5))     # native ETH is currency0


if __name__ == "__main__":
    unittest.main()
