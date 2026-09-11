# The fly, on a machine that stays on.
#
# One service: the brain, a headless Chromium it drives, and the socket the
# public site watches. No wallet and no secrets - the roaming browser has never
# had a key and this image has nowhere to put one.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLY_ALLOW_BROWSER=1 \
    FLY_HOST=0.0.0.0

WORKDIR /app

COPY requirements-roam.txt .
RUN pip install -r requirements-roam.txt \
 && python -m playwright install --with-deps chromium

# the connectome, derived once from Janelia's CC-BY release, and the learning
# the fly had already done before it moved house
COPY build/ build/
COPY data/ data/

COPY *.py ./
COPY web/ web/

CMD ["python", "roam.py"]
