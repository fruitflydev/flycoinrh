# The fly, on a machine that stays on.
#
# One container, supervised by run_all.py: the brain and the headless Chromium
# it drives, the socket the public site watches, the voice that narrates the
# journal, and - only with FLY_BACKROOM=1 - a paper executor.
#
# There is no wallet in this image and no private key. The roaming browser has
# never had one. The executor keeps a paper ledger of what the fly's clicks
# would have bought and sold, reads the chain for quotes, listens on loopback
# only and signs nothing: eth-account is deliberately absent from the
# requirements, so "paper only" is a property of what is installed here rather
# than a setting someone could flip. FLY_BACKROOM_LIVE=1 makes executor.py
# refuse to start; it is not a live mode, it is a stop.
#
# The model key and the X credentials come from the host's environment at run
# time, never from the image.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLY_ALLOW_BROWSER=1 \
    FLY_HOST=0.0.0.0

WORKDIR /app

# requirements-executor.txt is the ABI codec and the hash backend pons.py needs
# to read the chain: selectors and event topics are keccak and every call is
# ABI-encoded by hand. Without it executor.py dies on `import pons` at every
# restart and the supervisor gives up after eight, taking the roamer, the voice
# and the public stream with it - so it is installed whether the room is on or
# not, and run_all.py names it at boot if it is somehow missing.
COPY requirements-roam.txt requirements-executor.txt ./
RUN pip install -r requirements-roam.txt -r requirements-executor.txt \
 && python -m playwright install --with-deps chromium

# The connectome, derived once from Janelia's CC-BY release, and:
#   build/mb_sides.json          which dopamine cluster reaches each MBON. The
#                                mushroom body will not build without it, and
#                                it is derived and gitignored, so .dockerignore
#                                lets it through deliberately.
#   build/backroom_screen*.json  the offline gate's results - the paired-seed
#                                screen the decision rule was certified on.
#                                Nothing reads them at run time; they travel
#                                with the image so the numbers the site quotes
#                                can be checked inside the container.
COPY build/ build/
COPY data/ data/
# The DoOR receptor response tables. olfaction.py looks in ./door and
# ./data/door and raises if neither is there, which shuts the room.
COPY door/ door/

COPY *.py ./
COPY web/ web/
COPY voice_prompt.md ./

# roam.py, voice.py and - with FLY_BACKROOM=1 - executor.py, supervised
# together, each with its own environment (see run_all.py).
CMD ["python", "run_all.py"]
