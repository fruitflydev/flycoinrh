#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
pkg install -y python python-numpy
python -m pip install -r requirements-flyvue.txt
python -m py_compile feed.py run.py heatmap.py order.py server.py demo.py
test -f build/graph.npz
test -f data/body-annotations.feather
echo 'FLYVUE setup OK'
