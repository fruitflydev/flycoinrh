# Stream deck

A 16:9 / 9:16 overlay for putting the fly on socials.

## From @HexperiencePA / @anothertibbir

I'm on the chain. Main bag is $TIBBIR; I traded a slice on the $FLYBRAIN run-up. I'm here because the meme is the title: a real fly brain, actually working. That is the thing people will watch.

You've already got the live socket. This deck is so you can put it on a stream and cut clips — the fly clicking Wikipedia, xkcd, its own token page, different pages, different readouts. Minecraft-stream energy. Bitcoin-chart energy. People sit and watch a thing move if the thing is real.

The run-up is the window to fund that: a live social stream, an agent cutting short clips, posts that are just the fly working. Combine this, clip it, get it out. Happy to suggest more — reply on X and I'll reply. Or comment on the PR.

— [@HexperiencePA](https://x.com/HexperiencePA) · [@anothertibbir](https://x.com/anothertibbir)


The computer sits in an **inner screen**. Around it:

- **Hex retina** — 892 columns. What the fly actually sees, not a decoration.
- **Attention well** — a crop around the cursor. Labelled as a human zoom, not a fly fovea.
- **Experience** — mushroom-body carving. 44,042 KC→MBON synapses, depression only. The big number is novelty encounters. Sugar is invented here, and the caption says so.
- **Walk compass** — DNa02 / DNa01 / MDN / DNp09, the neurons that move the mouse.
- **Nose** — 53 receptor types, sitting unused (step 07). Anatomy is real; the flicker is spontaneous.
- **Soma weather** — firing neurons at measured coordinates when the socket is live.

Same websocket as flybrain.online. Same numbers. Nothing here steers the fly.

## OBS

Add a **Browser Source**.

| | |
|---|---|
| URL | `https://flybrain.online/stream` after deploy, or `http://localhost:4660/stream` next to `roam.py` |
| Width × height | `1920` × `1080` |
| Shutdown when not visible | off |
| Refresh when scene becomes active | on |

Local file also works: open `web/stream.html` directly. It finds the Railway socket the same way the public page does.

## Modes

| query | what you get |
|---|---|
| *(default)* | full cockpit, 16:9 |
| `?mode=overlay` | transparent HUD — put a window capture of the computer underneath |
| `?mode=vertical` | 9:16 for Shorts / Reels / TikTok |
| `?mode=retina` | hex eye only, hypnotic loop |
| `?source=demo` | labelled demo pulse if the fly is between lives |
| `?dock=0` | hide the operator buttons |

**Share a screen** pipes a Chromium window (or the flybrain.online tab) into the inner screen and samples it through the hex eye, so the dual view still works when you are capturing the computer yourself.

## Voice

No trading language on the overlay itself. Experience is not a video-game level. The reward signal is novelty standing in for sugar — the circuit is real, that substitution is not.

## Connectors (every number on the deck)

Same discovery as `site/web/index.html`. Nothing here steers the fly.

1. **Find the machine** — `GET {origin}/status` if this page is being served by `roam.py`; else `GET https://flybrain-production-2b26.up.railway.app/status`; else `site/web/live.json` on `main` (stale after 6h).
2. **Snapshot** — `GET {stream}/state` once, so the first paint has real numbers before the socket ticks.
3. **Socket** — `ws(s)://{stream}/ws`. Watcher only.

| socket `type` | fields used | where they land |
|---|---|---|
| `view` | `jpg`, `cx`, `cy` | inner screen + fly cursor. 1280×800, same as `web/roam.html` |
| `cursor` | `cx`, `cy` | fly sprite + attention-well crop |
| `frame` | `neural`, `stats`, `url`, `events`, `visited`, `cx`, `cy` | every readout |
| `place` | `title` / `url` | "where it has been" |
| `log` | `msg` | event river |
| `done` | — | pip → between lives; life count +1 |

| `neural.*` | widget |
|---|---|
| `firing`, `spikes_per_sec`, `mean_mv`, `visual`, `motor` | breath / spark / header counts |
| `dn.steer_L/R`, `fwd_L/R`, `back`, `stop` | compass + bars (scale 450 Hz, same as `web/live.html`) |
| `out.click` | freeze ring |
| `vision.cols`, `on_hz`, `off_hz`, `columns` | hex retina (live). Screen-share samples the capture instead |
| `scatter` | soma weather |
| `history` | firing sparkline |
| `learning.{synapses,depressed,mean_gain,rewards,punishments}` | carving. Big number = `rewards` (novelty). Caption says sugar is invented |

| `stats.*` | strip |
|---|---|
| `hops` `clicks` `scrolled` `steps` `vetoes` `blocked` | pages / clicks / scrolls / brain steps / vetoed / blocked |

**Share a screen** is the only extra input: `getDisplayMedia` → inner screen, hex eye samples that picture. Demo is labelled and ignores the socket so it cannot pretend to be the fly.

