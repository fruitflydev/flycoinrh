"""
Offline tests for xpost.py. No network: requests.post is replaced with a
recorder, the ledger lives in a temp dir, and load_env is stubbed so the
real .env cannot leak into the assertions.

  py -m unittest test_xpost -v
"""
import base64
import hashlib
import hmac
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import quote

import xpost


# -- pure functions -----------------------------------------------------------

class TestXLength(unittest.TestCase):
    def test_plain_ascii_is_one_per_char(self):
        self.assertEqual(xpost.x_length("hello fly"), 9)

    def test_empty(self):
        self.assertEqual(xpost.x_length(""), 0)

    def test_http_url_counts_23(self):
        url = "https://www.ponsfamily.com/launchpad/0x4eb990547bce4a982432ca88cf5fae7eed1a2d35"
        self.assertGreater(len(url), 23)
        self.assertEqual(xpost.x_length(url), 23)
        self.assertEqual(xpost.x_length("read " + url), 5 + 23)

    def test_short_url_still_counts_23(self):
        self.assertEqual(xpost.x_length("http://x.co"), 23)

    def test_two_urls(self):
        text = "a https://flybrain.online b http://example.com/path c"
        self.assertEqual(xpost.x_length(text), 2 + 23 + 3 + 23 + 2)

    def test_trailing_punctuation_is_text_not_link(self):
        self.assertEqual(xpost.x_length("see https://flybrain.online."), 4 + 23 + 1)
        self.assertEqual(xpost.x_length("(https://flybrain.online)"), 1 + 23 + 1)

    def test_bare_gtld_domain_is_linked(self):
        # X auto-links these and bills 23, even without a scheme.
        self.assertEqual(xpost.x_length("flybrain.online"), 23)
        self.assertEqual(xpost.x_length("on ponsfamily.com today"), 3 + 23 + 6)

    def test_bare_cctld_file_names_are_text(self):
        # roam.py and mushroom.py are file names, not links, on X.
        self.assertEqual(xpost.x_length("roam.py"), 7)
        self.assertEqual(xpost.x_length("mushroom.py depressed 3 synapses"), 32)

    def test_email_is_not_a_link(self):
        self.assertEqual(xpost.x_length("fly@example.com"), 15)

    def test_cjk_and_emoji_weigh_two(self):
        self.assertEqual(xpost.x_length("日本"), 4)       # two CJK chars
        self.assertEqual(xpost.x_length("a\U0001F41Ab"), 1 + 2 + 1)  # fly emoji

    def test_nfc_normalisation(self):
        # e + combining acute (2 code points) normalises to one code point.
        self.assertEqual(xpost.x_length("é"), 1)

    def test_boundary(self):
        self.assertEqual(xpost.x_length("x" * 280), 280)
        self.assertEqual(xpost.x_length("x" * 257 + " https://flybrain.online"), 281)


class TestEnc(unittest.TestCase):
    def test_known_vectors(self):
        self.assertEqual(xpost.enc(" "), "%20")
        self.assertEqual(xpost.enc("!"), "%21")
        self.assertEqual(xpost.enc("*"), "%2A")
        self.assertEqual(xpost.enc("'"), "%27")
        self.assertEqual(xpost.enc("("), "%28")
        self.assertEqual(xpost.enc(")"), "%29")
        self.assertEqual(xpost.enc("~"), "~")
        self.assertEqual(xpost.enc("/"), "%2F")
        self.assertEqual(xpost.enc("+"), "%2B")
        self.assertEqual(xpost.enc("="), "%3D")
        self.assertEqual(xpost.enc("&"), "%26")
        self.assertEqual(xpost.enc("é"), "%C3%A9")

    def test_unreserved_untouched(self):
        s = "ABCxyz019-_.~"
        self.assertEqual(xpost.enc(s), s)

    def test_uppercase_hex(self):
        self.assertEqual(xpost.enc("ÿ"), "%C3%BF")

    def test_non_str_is_stringified(self):
        self.assertEqual(xpost.enc(1318622958), "1318622958")


class TestOAuthHeader(unittest.TestCase):
    CREDS = {"key": "ck", "secret": "cs!", "token": "tk", "tsecret": "ts*"}
    FIELDS = ["oauth_consumer_key", "oauth_nonce", "oauth_signature",
              "oauth_signature_method", "oauth_timestamp", "oauth_token",
              "oauth_version"]

    def _hand_signature(self, method, url, params, secret, tsecret):
        # The same algorithm, written out independently of xpost so the test
        # catches a regression in either the base string or the key.
        e = lambda s: quote(str(s), safe="-_.~")
        ps = "&".join(f"{e(k)}={e(params[k])}" for k in sorted(params))
        base = "&".join([method, e(url), e(ps)])
        key = f"{e(secret)}&{e(tsecret)}".encode()
        return base64.b64encode(hmac.new(key, base.encode(), hashlib.sha1).digest()).decode()

    def test_fixed_vector(self):
        nonce, ts = "0123456789abcdef0123456789abcdef", 1700000000
        hdr = xpost.oauth_header("post", xpost.TWEET_URL, self.CREDS,
                                 nonce=nonce, timestamp=ts)
        expected = self._hand_signature("POST", xpost.TWEET_URL, {
            "oauth_consumer_key": "ck", "oauth_nonce": nonce,
            "oauth_signature_method": "HMAC-SHA1", "oauth_timestamp": str(ts),
            "oauth_token": "tk", "oauth_version": "1.0"}, "cs!", "ts*")
        self.assertIn(f'oauth_signature="{xpost.enc(expected)}"', hdr)

    def test_all_seven_fields_sorted(self):
        hdr = xpost.oauth_header("POST", xpost.TWEET_URL, self.CREDS,
                                 nonce="n", timestamp=1)
        self.assertTrue(hdr.startswith("OAuth "))
        keys = [part.split("=", 1)[0] for part in hdr[len("OAuth "):].split(", ")]
        self.assertEqual(keys, self.FIELDS)
        self.assertEqual(keys, sorted(keys))
        self.assertIn('oauth_signature_method="HMAC-SHA1"', hdr)
        self.assertIn('oauth_version="1.0"', hdr)
        self.assertIn('oauth_timestamp="1"', hdr)

    def test_extra_params_enter_the_base_string_but_not_the_header(self):
        a = xpost.oauth_header("POST", xpost.TWEET_URL, self.CREDS, nonce="n", timestamp=1)
        b = xpost.oauth_header("POST", xpost.TWEET_URL, self.CREDS, {"status": "hi"},
                               nonce="n", timestamp=1)
        self.assertNotEqual(a, b)
        self.assertNotIn("status", b)

    def test_nonce_and_timestamp_default_to_fresh_values(self):
        a = xpost.oauth_header("POST", xpost.TWEET_URL, self.CREDS)
        b = xpost.oauth_header("POST", xpost.TWEET_URL, self.CREDS)
        self.assertNotEqual(a, b)
        self.assertIn('oauth_timestamp="', a)

    def test_x_documentation_vector(self):
        # The worked example from X's "Creating a signature" page. Matching
        # it proves the base string and key are what X's servers compute.
        creds = {"key": "xvz1evFS4wEEPTGEFPHBog",
                 "secret": "kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw",
                 "token": "370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb",
                 "tsecret": "LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE"}
        hdr = xpost.oauth_header(
            "POST", "https://api.twitter.com/1.1/statuses/update.json", creds,
            {"include_entities": "true",
             "status": "Hello Ladies + Gentlemen, a signed OAuth request!"},
            nonce="kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg", timestamp=1318622958)
        self.assertIn('oauth_signature="hCtSmYh%2BiHYCEqBWrE7C7hYmtUk%3D"', hdr)


# -- publish() with a fake network and a temp ledger --------------------------

class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakePost:
    """Stands in for requests.post; records every call, answers per URL."""

    def __init__(self, tweet=None, upload=None):
        self.calls = []
        self.tweet = tweet or (lambda i: FakeResponse(201, {"data": {"id": str(1000 + i)}}))
        self.upload = upload or (lambda i: FakeResponse(200, {"media_id_string": "m1"}))

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        i = len(self.calls)
        if url == xpost.UPLOAD_URL:
            return self.upload(i)
        if url == xpost.TWEET_URL:
            return self.tweet(i)
        raise AssertionError(f"unexpected URL {url}")

    def tweets(self):
        return [kw for url, kw in self.calls if url == xpost.TWEET_URL]


class PublishBase(unittest.TestCase):
    ENV = {"X_API_KEY": "k", "X_API_SECRET": "s", "X_ACCESS_TOKEN": "t",
           "X_ACCESS_SECRET": "ts", "X_ENABLED": "true"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith("X_") or k == "FLY_STATE_DIR":
                del os.environ[k]
        os.environ.update(self.ENV)
        os.environ["FLY_STATE_DIR"] = self.tmp.name
        # The real .env must not reach these tests, in either direction.
        self.saved_load_env = xpost.load_env
        xpost.load_env = lambda: {}
        self.saved_cap = xpost.MAX_PER_DAY
        self.saved_post = xpost.requests.post
        self.fake = FakePost()
        xpost.requests.post = self.fake

    def tearDown(self):
        xpost.requests.post = self.saved_post
        xpost.MAX_PER_DAY = self.saved_cap
        xpost.load_env = self.saved_load_env
        os.environ.clear()
        os.environ.update(self.saved_env)
        self.tmp.cleanup()

    def ledger(self):
        p = Path(self.tmp.name) / "x_ledger.json"
        return json.loads(p.read_text(encoding="utf-8"))["posts"] if p.exists() else []


class TestPublishGuards(PublishBase):
    def test_ledger_lives_in_state_dir(self):
        self.assertEqual(xpost.ledger_path(), Path(self.tmp.name) / "x_ledger.json")

    def test_disabled_skips_without_network(self):
        os.environ["X_ENABLED"] = "false"
        self.assertEqual(xpost.publish("hello"), {"skipped": "X_ENABLED=false"})
        self.assertEqual(self.fake.calls, [])

    def test_unset_is_disabled(self):
        del os.environ["X_ENABLED"]
        self.assertEqual(xpost.publish("hello"), {"skipped": "X_ENABLED=false"})

    def test_missing_credentials_skip_and_never_raise(self):
        del os.environ["X_ACCESS_SECRET"]
        r = xpost.publish("hello")
        self.assertIn("skipped", r)
        self.assertIn("X_ACCESS_SECRET", r["skipped"])
        self.assertEqual(self.fake.calls, [])

    def test_too_long_raises_value_error(self):
        with self.assertRaises(ValueError):
            xpost.publish("x" * 281)
        with self.assertRaises(ValueError):
            xpost.publish("x" * 258 + " https://flybrain.online")   # 258+23 = 281
        self.assertEqual(self.fake.calls, [])

    def test_exactly_280_is_accepted(self):
        self.assertEqual(xpost.publish("x" * 280), {"id": "1001"})

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            xpost.publish("   ")

    def test_daily_cap(self):
        xpost.MAX_PER_DAY = 2
        self.assertEqual(xpost.publish("one"), {"id": "1001"})
        self.assertEqual(xpost.publish("two"), {"id": "1002"})
        self.assertEqual(xpost.publish("three"), {"skipped": "daily cap"})
        self.assertEqual(len(self.fake.tweets()), 2)
        self.assertEqual([p["text"] for p in self.ledger()], ["one", "two"])

    def test_cap_is_a_rolling_24h_window(self):
        xpost.MAX_PER_DAY = 1
        old = {"t": time.time() - xpost.DAY - 60, "id": "9", "text": "yesterday"}
        (Path(self.tmp.name) / "x_ledger.json").write_text(
            json.dumps({"posts": [old]}), encoding="utf-8")
        self.assertEqual(xpost.publish("today"), {"id": "1001"})
        self.assertEqual(xpost.publish("again"), {"skipped": "daily cap"})

    def test_cap_reads_env_default(self):
        self.assertEqual(xpost.MAX_PER_DAY, 12)   # X_MAX_POSTS_PER_DAY unset at import

    def test_dedupe(self):
        self.assertEqual(xpost.publish("same words"), {"id": "1001"})
        self.assertEqual(xpost.publish("same words"), {"skipped": "duplicate"})
        self.assertEqual(xpost.publish("other words"), {"id": "1002"})
        self.assertEqual(len(self.fake.tweets()), 2)

    def test_dedupe_window_is_last_50(self):
        xpost.MAX_PER_DAY = 1000
        posts = [{"t": time.time(), "id": str(i), "text": f"n{i}"} for i in range(60)]
        (Path(self.tmp.name) / "x_ledger.json").write_text(
            json.dumps({"posts": posts}), encoding="utf-8")
        self.assertEqual(xpost.publish("n5"), {"id": "1001"})          # fell out of the window
        self.assertEqual(xpost.publish("n59"), {"skipped": "duplicate"})

    def test_unreadable_ledger_fails_closed(self):
        (Path(self.tmp.name) / "x_ledger.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(xpost.publish("hello"), {"skipped": "ledger unreadable"})
        self.assertEqual(self.fake.calls, [])

    def test_ledger_is_bounded(self):
        xpost.MAX_PER_DAY = 10000
        posts = [{"t": 0, "id": str(i), "text": f"n{i}"} for i in range(xpost.LEDGER_KEEP + 20)]
        (Path(self.tmp.name) / "x_ledger.json").write_text(
            json.dumps({"posts": posts}), encoding="utf-8")
        xpost.publish("fresh")
        self.assertEqual(len(self.ledger()), xpost.LEDGER_KEEP)
        self.assertEqual(self.ledger()[-1]["text"], "fresh")


class TestPublishNetwork(PublishBase):
    def test_text_post_shape(self):
        r = xpost.publish("hello fly")
        self.assertEqual(r, {"id": "1001"})
        url, kw = self.fake.calls[0]
        self.assertEqual(url, xpost.TWEET_URL)
        self.assertEqual(kw["json"], {"text": "hello fly"})
        self.assertTrue(kw["headers"]["Authorization"].startswith("OAuth "))
        self.assertIn('oauth_consumer_key="k"', kw["headers"]["Authorization"])
        self.assertEqual(self.ledger()[0]["id"], "1001")
        self.assertFalse(self.ledger()[0]["media"])

    def test_image_is_uploaded_first_and_attached(self):
        r = xpost.publish("with picture", b"\xff\xd8\xff", "image/jpeg")
        self.assertEqual(r, {"id": "1002"})
        urls = [u for u, _ in self.fake.calls]
        self.assertEqual(urls, [xpost.UPLOAD_URL, xpost.TWEET_URL])
        up = self.fake.calls[0][1]
        self.assertEqual(up["files"]["media_data"][1], base64.b64encode(b"\xff\xd8\xff").decode())
        self.assertNotIn("data", up)   # multipart, not urlencoded: nothing to sign
        self.assertEqual(self.fake.tweets()[0]["json"]["media"], {"media_ids": ["m1"]})
        self.assertTrue(self.ledger()[0]["media"])

    def test_media_failure_does_not_block_text(self):
        self.fake.upload = lambda i: FakeResponse(400, {"errors": [{"message": "bad media"}]})
        r = xpost.publish("words survive", b"\x00")
        self.assertEqual(r, {"id": "1002"})
        self.assertNotIn("media", self.fake.tweets()[0]["json"])

    def test_media_exception_does_not_block_text(self):
        def boom(i):
            raise xpost.requests.ConnectionError("upload host down")
        self.fake.upload = boom
        self.assertEqual(xpost.publish("words survive", b"\x00"), {"id": "1002"})

    def test_api_failure_raises_with_status_and_body(self):
        self.fake.tweet = lambda i: FakeResponse(403, {"detail": "Forbidden " + "z" * 500})
        with self.assertRaises(RuntimeError) as cm:
            xpost.publish("nope")
        self.assertIn("403", str(cm.exception))
        self.assertIn("Forbidden", str(cm.exception))
        self.assertLessEqual(len(str(cm.exception)), 240 + 40)
        self.assertEqual(self.ledger(), [])   # a failed post does not count

    def test_network_error_becomes_runtime_error(self):
        def boom(i):
            raise xpost.requests.ConnectionError("dns")
        self.fake.tweet = boom
        with self.assertRaises(RuntimeError):
            xpost.publish("nope")


if __name__ == "__main__":
    unittest.main()
