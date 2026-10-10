#!/usr/bin/env bash
# Third Eye demo launcher.
#
# Usage: ./run_demo.sh [args passed to third_eye_app.py]
#   e.g. ./run_demo.sh --headless --seconds 60
#        ./run_demo.sh --source clip.mp4 --record out.mp4
#
# Environment:
#   THIRDEYE_SKIP_DIAG=1   skip the quick pre-flight diagnostics
set -euo pipefail

die() {
    echo "run_demo.sh: ERROR: $*" >&2
    exit 1
}

# --- Locate the repo (directory of this script, following symlinks) --------
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
REPO_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
cd "$REPO_DIR"

# --- Python environment ------------------------------------------------------
[ -f .venv/bin/activate ] || die "virtualenv not found at $REPO_DIR/.venv (expected .venv/bin/activate)"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

[ -f yolo26n.pt ] || die "model weights yolo26n.pt not found in $REPO_DIR"
[ -f third_eye_app.py ] || die "third_eye_app.py not found in $REPO_DIR (the app has not been built yet)"

# --- Display -----------------------------------------------------------------
HEADLESS=0
for arg in "$@"; do
    [ "$arg" = "--headless" ] && HEADLESS=1
done

if [ -z "${DISPLAY:-}" ] && [ "$HEADLESS" -eq 0 ]; then
    XSOCK=""
    for sock in /tmp/.X11-unix/X*; do
        [ -S "$sock" ] && { XSOCK="$sock"; break; }
    done
    if [ -n "$XSOCK" ]; then
        export DISPLAY=":${XSOCK##*/X}"
        echo "run_demo.sh: DISPLAY was empty, using local X display $DISPLAY"
    else
        die "no DISPLAY set and no X server found in /tmp/.X11-unix. Run with --headless, e.g. ./run_demo.sh --headless"
    fi
fi

# --- Signal handling (installed before diagnostics so a stop request
# during pre-flight aborts instead of launching the app) -----------------
CHILD=""
STOPPING=0
forward() {
    STOPPING=1
    if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
        kill -"$1" "$CHILD" 2>/dev/null || true
    fi
}
trap 'forward INT' INT
trap 'forward TERM' TERM
trap 'forward HUP' HUP

# --- Pre-flight diagnostics ---------------------------------------------------
if [ "${THIRDEYE_SKIP_DIAG:-0}" = "1" ]; then
    echo "run_demo.sh: skipping diagnostics (THIRDEYE_SKIP_DIAG=1)"
else
    if ! python scripts/diagnostics.py --quick; then
        [ "$STOPPING" -eq 1 ] && { echo; echo "Third Eye stopped."; exit 130; }
        die "pre-flight diagnostics failed (see above). Fix the FAIL lines, or bypass with THIRDEYE_SKIP_DIAG=1 ./run_demo.sh"
    fi
fi

# --- Launch -------------------------------------------------------------------
if [ "$STOPPING" -eq 1 ]; then
    echo "Third Eye stopped."
    exit 130
fi
echo "run_demo.sh: starting Third Eye ($*)"

set +e
# Start the app in its own process group (set -m) so that a terminal Ctrl+C
# reaches only this script, which forwards exactly one SIGINT to the app.
# This also keeps SIGINT un-ignored in the child, so Python raises
# KeyboardInterrupt and the app can release the camera cleanly.
set -m
python third_eye_app.py "$@" &
CHILD=$!
set +m
# wait returns early (>128) when a trapped signal arrives; keep waiting
# until the child has actually exited.
while true; do
    wait "$CHILD"
    STATUS=$?
    if kill -0 "$CHILD" 2>/dev/null; then
        continue
    fi
    break
done
set -e
trap - INT TERM HUP

if [ "$STOPPING" -eq 1 ]; then
    echo
fi
echo "Third Eye stopped."
exit "$STATUS"
