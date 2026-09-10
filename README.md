# flycoinrh

A real fruit fly brain, simulated neuron by neuron, driving the
[pons launchpad](https://www.ponsfamily.com/launchpad) on Robinhood Chain.

165,122 neurons. 10,228,000 signed synaptic connections. Every one of them
measured from an actual male *Drosophila melanogaster* by electron microscopy —
not invented, not sampled from a distribution, not a neural network "inspired
by" a brain.

This is the Robinhood Chain half. The Solana half — same brain, pump.fun —
lives at [ad7584/flycoin](https://github.com/ad7584/flycoin).

## What it actually does

Press START and a real Chromium opens ponsfamily.com/launchpad. Its screenshots
are sampled through the fly's **892 retinotopic hex columns** into L1 and L2 —
the lamina monopolar cells that are the direct postsynaptic targets of
photoreceptors R1–R6. 165,122 neurons integrate. The cursor comes back out of
the descending neurons a fly actually walks with:

| neuron | what it does in a fly | what it does here |
|---|---|---|
| **DNa02** left vs right | steering — a fly turns by left/right asymmetry | cursor x |
| **DNa01** | forward walking | cursor y |
| **MDN** | the Moonwalker descending neuron — walking backwards | reverse |
| **DNp09** | stopping | the click |

Then it connects the wallet, accepts the launchpad's terms, uploads the token
image, fills the form, opens **Advanced**, sets the creator tax, and stops on
the launch button.

## Why Robinhood Chain is the better half of this project

pump.fun's backend answers `401 Unauthorized` to an injected wallet, so its own
Create button can never complete a launch — the Solana repo has to bypass the
site entirely and mint through a signed transaction.

The pons launchpad **accepts the injected EIP-1193 wallet directly**. It
auto-connects with no modal, `eth_chainId` returns `0x1237`, `personal_sign`
round-trips. So here the site's own button is the real path: the fly's click
would genuinely produce the transaction.

## Check it yourself

**The chain.** Robinhood Chain is an Arbitrum Nitro L2 — chain id **4663**, RPC
`https://rpc.mainnet.chain.robinhood.com`, gas in ETH at about 0.13 gwei.

**The wallet.** `py rhwallet.py new` generates a secp256k1 keypair, writes the
secret straight into gitignored `.env`, and prints only the address. It is never
printed, never returned, never passed through a chat window.

**The transaction path.** `py rhdryrun.py` exercises every step except the
broadcast: chain id, nonce, gas price, a real signed transaction, and — the
part that matters — recovering the signature and checking it against the
wallet's own address. Nothing is broadcast; there is no code path in that file
that can send.

```
chain id        4663 ok
nonce           0
gas price       0.1358 gwei
signed tx       110 bytes
recovered from  0x739Ccc9d…bFc3  MATCHES
estimateGas     21000 units
```

**The connectome.** CC-BY, from a public bucket, no account and no key:

```
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/
  body-annotations-male-cns-v1.0-minconf-0.5.feather      14 MB
  body-neurotransmitters-male-cns-v1.0.feather            42 MB
  connectome-weights-male-cns-v1.0-minconf-0.5.feather   1.1 GB
```

`py build_graph.py` turns those into 165,122 traced neurons and 10,228,000
signed edges. If your numbers differ from mine, one of us has a bug.

## What is NOT real, stated plainly

- **The fly does not fill the whole form.** It reliably gets the description,
  often the ticker, rarely the name, and does not navigate to the launch button
  on its own. The rig completes whatever it misses, and the on-screen log says
  which fields were which — `by FLY: …  ·  by rig: …`.
- **Light mode breaks it.** The fly's retina is a luminance map and every bit of
  tuning it has was done against dark UI. On the launchpad's default light theme
  it filled **0 of 3** fields; in dark mode, **2 of 3**. The rig switches the
  theme before it starts, and that gap is the clearest evidence its vision is
  doing real work.
- **The idle animation is decoration.** During a run every dot is a neuron at a
  measured soma coordinate. While idle it is a fly-shaped scatter, and the panel
  label changes to say so.
- **Nothing has launched yet.** The wallet is unfunded, so the button reads
  `Insufficient ETH` and every run so far ends there with zero transaction
  requests. The launchpad's own transaction — its shape, and whether it takes
  one signature or several — cannot be seen until the button enables.

## Running it

```bash
py rhwallet.py new              # create the wallet, then fund it (~0.002 ETH)
py rhdryrun.py                  # prove the signing path, spend nothing
py rhlive.py                    # http://localhost:4651, press START
py record.py --port 4651        # record the run to build/recordings/
```

Two flags gate everything, both in `.env`, both off by default:

- `FLY_ALLOW_BROWSER=1` — required before a browser will open against a real
  site at all. A button in a web page is not a strong enough guard for that.
- `FLY_RH_LIVE=1` — required before any transaction is signed. Funding the
  wallet does not, on its own, arm anything.

## Credits

Connectome data © HHMI Janelia FlyEM, the Cambridge Connectomics Group and
Google Research, released CC-BY. Simulation approach after Shiu et al. 2024 and
Lappalainen et al. 2024. Not affiliated with any of them, nor with pons or
Robinhood.
