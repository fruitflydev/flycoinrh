# Disclosure copy for the fly's voice

Everything below is human-written account and site copy, and it is labelled
as such wherever it appears. None of it is a journal entry. Journal entries
and posts on X come only from `voice.py`, written by the narrator from a live
observation packet and checked before they leave; there is no manual posting
path through the fly's account, and `xpost.py` has no command to post text.

## X bio (<= 160 characters)

Real fruit fly connectome, 165,122 neurons, roaming the web. It launched a coin. The neurons and numbers are measured. Words: a language model narrator's.

## Pinned post (<= 280 characters, URL counts as 23)

Journal of a real fruit fly connectome: 165,122 neurons, 10,228,000 synapses (Janelia FlyEM, CC-BY), simulated, roaming the web. It launched $FLYBRAIN on Robinhood Chain. Real neurons. Real numbers. The words: a language model narrating its telemetry. flybrain.online

The pinned post is account copy, set by a person from the X app, not sent
through the fly's posting pipeline.

## Site: "What is NOT real, stated plainly"

- **The voice is a narrator.** The fly has no language. Every journal entry
  and every post on X is written by a language model that is handed the fly's
  telemetry (which neurons fired, where the cursor went, which pages it landed
  on) and the live numbers from its token page, and asked to write in the
  first person. Every number in a draft is checked against that packet before
  it goes out; a draft with a number that is not in the packet is thrown away,
  and so is one with trading language. The neurons are real, the pages are
  real, the fees are real. The words, and any "confusion" or "curiosity" in
  them, are the narrator's.

## README subsection

### The voice

The fly has no language, so its journal is written for it, and `voice.py` is
the writer. Observe: pull `/state` from the live roamer, the live token page
and the on-chain launch constants into one packet. Read: fetch a short
allowlist of real pages (its own token page, its own site, a few reference
pages on memecoins and tokenised stocks) and pass excerpts in. Write: a
language model drafts a first-person entry from that packet and nothing else.
Check: every number in the draft must appear in the packet, and any trading
or hype language fails it; a rejected draft is dropped, there is no second
draft. Post. Two things in this project are invented, and both are labelled:
the reward signal in the mushroom body, and the words. Everything else is a
measurement.

## Site: the backroom, stated plainly

- **It is paper.** No wallet exists. Nothing is signed and nothing is sent.
  Every trade in the backroom is priced against live on-chain quotes from the
  real launchpad contracts and then booked against a pretend balance. The
  gas it charges itself is an estimate, not a measured fee.
- **Nobody places its trades.** There is no button, command, endpoint or
  setting that buys, sells, sizes, cancels or reverses anything: not the
  owner, not the developers, not the narrator. The settings that do exist are
  named, and none of them is a trade: paper or real at startup, stopping the
  whole fly, how much pretend money it starts with, how often a restart puts
  it in the room and how long a visit lasts, and how old a look may be before
  the executor treats it as one the fly has walked away from. That last one is
  held between five and sixty seconds precisely so that it cannot be turned
  into a way to refuse everything the fly decides, and both it and the paper
  balance are printed when the executor starts.
- **People chose which coins it is offered.** The room shows the launchpad's
  sixteen most recently bought coins, refreshed every twenty seconds, minus
  any that are not quoted in the chain's own coin. The listing could have been
  sorted by newest, oldest, market cap or volume; a person picked recent
  buying. It is a filter applied before the fly ever sees a card, so what the
  paper book does is partly that filter's doing and not the fly's.
- **How it decides.** The fly walks into a room of real coins drawn as cards.
  It sees them the way it sees any page, through 892 hex columns at about
  thirty by thirty pixels, so it cannot read a name, a price or a chart. Each
  card carries a smell: a script picks an odorant from the coin's own words,
  or hashes its name into a blend when no word matches, and drives the
  receptor neurons with responses measured from real flies. When the fly
  stops on a card, the stop is the commit, and what its mushroom body was
  carrying at that moment decides whether the commit is a buy or a sell, and
  how much of the balance goes in.
- **What is chosen by people, not measured.** That a word means an odorant.
  That every coin's smell is scaled toward the same loudness - which does not
  reach equality, because no receptor may respond above its measured maximum,
  so a coin whose smell lands on one glomerulus stays about half as loud as
  one spread over twenty and fires far fewer Kenyon cells. That like and
  dislike are read from the mushroom body's output synapses and compared
  against the other cards in the room. That stopping on a card means a trade.
  That profit arrives as reward dopamine and loss as punishment dopamine: a
  real fly is rewarded with sugar and punished with shock, not with a price.
- **What learning actually does.** Measured on this connectome before any of
  it was switched on, with the same reading the room uses - the mushroom
  body's output synapses, each card against the other cards - and with the
  same picture on every card so that only smell could tell the coins apart.
  Twelve profits on one coin and twelve losses on another: the profitable
  coin's own reading rose 0.036, the loss-making one fell 0.040, and a third
  coin that was never paired with anything moved 0.004, which is inside the
  noise. Trained one sign at a time the split is coarser, about two thirds
  staying on the coin that earned it and the rest spreading to coins that
  smell similar. Learning here is real, and it is partly global.
- **Two other ways of reading it were measured and are not used.** Reading a
  card on its own, against a blank grey screen, does not carry the lesson at
  all - whether the reading is the output synapses or the MBONs' firing rates.
  Measured the same way: the coin that was punished moved the wrong way, and a
  coin that was never paired with anything moved 0.88 to 0.92 as far as a
  trained one. That is why the number is a comparison between cards. Both ways
  of doing it average over the other cards; what differs is when they were
  read. The offline gate can look at every coin at the same instant and at the
  same seed. A walking fly cannot - it has one run a step - so the room
  compares a card against the fly's last look at each other card earlier in the
  same visit, which may be a minute old and taken from another position on the
  page.
- **One commit is not that measurement.** The numbers above are a mean over
  sixty runs with the same picture on every card and the cursor at its centre.
  A single stop has none of those controls. The spread from one look to the
  next is larger than the whole lesson; two untrained coins already differ from
  each other by more than it, so a coin whose name happens to hash to a strong
  smell carries an offset that does not average away; and the eye window is
  wider than a card, so a stop that is not at a card's centre reads the
  neighbours too - between a third and six sevenths of the window was the card
  being judged, across the first paper session's nine commits. Nothing corrects
  for any of it, because nothing may stand between the brain and the order.
  Each stored look records how much of the window was its own card and how much
  of the brain fired in it, so this is checkable rather than only stated.
- **What the fly has learned is a plain file.** The gains live in one .npz in
  the state directory and are trusted on sight: anything that can write there
  can decide what the fly likes, and can write the reward and punishment counts
  that would seem to corroborate it. The ledger is trusted the same way, but
  the ledger replays and refuses when it does not add up, and this does not. It
  is said here rather than fixed with a secret this process has nowhere safe to
  keep.
- **It can lose the paper balance completely.** There is no stop, cap, target
  or limit anywhere in the decision. A single commit can spend everything
  free. Nothing it does is advice, a signal or a prediction.
- **The narrator does not report the backroom yet.** The voice is not given
  the room, its trades or its page, so no post can mention them until the
  self-report rules are written and checked. That includes the picture: an
  entry written while the fly is in the room goes out without one, because the
  live frame would be a photograph of the coin board. It also includes what
  learning has done: with the room on, every lesson is a paper profit or a
  paper loss, so the counts of rewards and punishments are the counts of
  winning and losing trades. They are no longer in the packet the narrator
  writes from, and neither is how far they have moved the weights; the number
  of synapses is, because that is the circuit and not its history.

## README subsection

### The backroom

The fly's roaming already had a reward signal, and it was invented: reaching a
new page counted as sugar. That is gone. The only thing that trains the
mushroom body now is what happens to the coins the fly itself picked.

About a third of the restarts that would drop the fly back onto a web page
put it in a room instead, served from the fly's own machine: near black, with
a grid of cards, real coins from the launchpad, each with its logo, its name
and how far along its curve it is. How often that happens is a chosen
schedule, and it sets how much the fly trades. Nothing in the room is a
link or a button, and the roamer never clicks there. The fly walks, fixates,
and its brain does the rest. A card under the cursor adds that coin's smell
to the same simulation step that is steering it, so sight and smell arrive
together, as they do in a fly. Stopping on a card commits: the descending
neuron that halts a walking fly is the one that fires, and the mushroom body
output at that moment sets the side and the size. A buy spends that fraction
of the free balance; a sell sells that fraction of the position.

Settlement happens in a separate process that holds the ledger, quotes the
venue on chain, and books the fill. In paper mode it never signs anything and
never sends anything; the wallet does not exist. When a coin the fly holds is
looked at again, the room asks what that position is worth now, and the
difference since the last look arrives as dopamine at the Kenyon cell
synapses: better is sugar, worse is shock, and a doubling and a halving weigh
the same. That is the whole strategy. There is no other one.

## Plume tracking

Account copy for the plume experiment, stated plainly. It is a science
experiment on the simulated brain, run offline on 2026-09-12: no page, no
room, no coin, no post. The record is `build/plume_experiment.json`, the
paths are `build/plume_trajectories.npz`, and the plain-language account is
`build/plume_report.md`.

- **What the fly was made to do.** Walk in a wind tunnel 0.6 m by 0.3 m,
  0.40 m downwind of an odour source, for 20 s at a time, with ethyl acetate
  on its receptor neurons and the wind on the Johnston's organ cells of its
  antennae, and its walking read off the same descending neurons the roamer
  uses. A real fly surges upwind when it meets odour and casts crosswind when
  it loses it. The question was whether that comes out of the wiring by
  itself, with nothing learned and nothing fitted. Four predictions and every
  metric were written down before the first trial, and no constant was
  changed after.
- **What is real.** The neurons: 2,635 receptor neurons in 53 types, 32 of
  which respond to ethyl acetate in the DoOR dataset; 335 wind-sensing cells,
  203 rooting on the left antenna and 132 on the right; ten descending
  neurons, one steering and one forward cell per side, four backward, two
  stop. Every number below is a count of spikes or a position in the tunnel,
  ten seeds by three conditions, and all of it is in the record.
- **What is chosen by people.** Which descending neuron stands for which
  movement, and the scale, inherited from the roamer; the brain's calibration
  gains; that odour is one odorant at 200 Hz at full strength; that wind is a
  cosine of its angle at each antenna, the antennae 45 degrees apart, 100 Hz
  at most, the same rate for every cell on a side; that the no-wind control
  holds both antennae at 50 Hz; the tunnel, the wind speed, the plume's width
  and wander and a 6 s warm-up; a top speed of 2 cm/s; and the thresholds and
  windows that define an encounter, a loss, a surge and a cast. Also, by hand
  before launch, ten seeds instead of twelve, because one brain run took
  0.31 s and not the 0.2 s estimated.
- **The numbers.** Surge: upwind speed rose 2.2 +/- 1.4 mm/s in the second
  after an encounter, 1.5 standard errors, short of the 2 the prediction
  required. Cast: crosswind speed fell 0.85 +/- 0.26 mm/s in the two seconds
  after a loss, the opposite of a cast, and the spread of heading change
  moved 0.06 +/- 0.19 degrees. Upwind progress in 20 s: 46 +/- 5 mm with
  odour, 59 +/- 7 mm with no odour at all, 8 +/- 14 mm with odour but no wind
  sense. Source reached: none of thirty trials, in any condition. All four
  predictions failed.
- **What the fly did instead.** With wind on its antennae it faced downwind
  on 91 % of steps with odour and 94 % without, and because its
  backward-walking neurons outfired its forward ones under the roamer's
  mapping, it backed upwind at 2 to 3 mm/s either way. That is the whole of
  the upwind progress, and it happens without any odour. Odour did move the
  motor neurons - the left forward cell fired at 112 Hz against 61 Hz with no
  odour, the stop cells at 70 against 99 - but the shift was not organised
  into a surge, a cast or a path.
- **The honest limits.** The brain restarts from rest at every 50 ms step, so
  it has no memory of what it smelled a moment ago, and surge and cast are
  memories. The receptor neurons do not adapt, every synapse weighs the same
  0.275 mV, and no spike takes any time to travel. The wind encoding is a
  stand-in, and its 203 : 132 population split means the brain never receives
  a symmetric headwind. The fly was born inside the plume on nine seeds of
  ten, and the source was 0.37 m away at a top speed of 0.02 m/s, so reaching
  it needed 92.5 % of full speed straight upwind for the whole trial. Any of
  these could be why it failed; none of them was changed after the result.
  What the numbers support is "this simulator, driven this way, did not track
  a plume", and nothing stronger in either direction.
