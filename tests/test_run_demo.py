"""run_demo.sh must pass its arguments to third_eye_app.py unchanged.

Runs the real script from a temporary copy of the repo layout with a stub
virtualenv whose `python` only records its argv (no camera, model or GPU).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUB_ACTIVATE = 'export PATH="$(dirname "${BASH_SOURCE[0]}"):$PATH"\n'
STUB_PYTHON = """#!/usr/bin/env bash
# Record app invocations only (diagnostics are skipped via THIRDEYE_SKIP_DIAG).
for arg in "$@"; do printf '%s\\n' "$arg"; done > "$ARGV_OUT"
"""


class RunDemoArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        shutil.copy2(os.path.join(REPO_ROOT, "run_demo.sh"), self.repo)
        bin_dir = os.path.join(self.repo, ".venv", "bin")
        os.makedirs(bin_dir)
        with open(os.path.join(bin_dir, "activate"), "w") as fh:
            fh.write(STUB_ACTIVATE)
        python = os.path.join(bin_dir, "python")
        with open(python, "w") as fh:
            fh.write(STUB_PYTHON)
        os.chmod(python, 0o755)
        for name in ("yolo26n.pt", "third_eye_app.py"):
            open(os.path.join(self.repo, name), "w").close()
        self.argv_out = os.path.join(self.repo, "argv.txt")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def forwarded(self, *args: str) -> list[str]:
        env = dict(os.environ, THIRDEYE_SKIP_DIAG="1", DISPLAY=":0", ARGV_OUT=self.argv_out)
        result = subprocess.run(
            ["bash", os.path.join(self.repo, "run_demo.sh"), *args],
            env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.argv_out) as fh:
            return fh.read().splitlines()

    def test_fullscreen(self) -> None:
        self.assertEqual(self.forwarded("--fullscreen"), ["third_eye_app.py", "--fullscreen"])

    def test_fullscreen_incidents(self) -> None:
        self.assertEqual(
            self.forwarded("--fullscreen", "--incidents"),
            ["third_eye_app.py", "--fullscreen", "--incidents"],
        )

    def test_existing_options_and_values_unchanged(self) -> None:
        args = ["--debug", "--zone", "0.6,0.1,0.95,0.9", "--record-raw", "out dir/raw.mp4",
                "--source", "rehearsal.mp4", "--source-fps", "15.2", "--seconds", "5", "--quiet"]
        self.assertEqual(self.forwarded(*args), ["third_eye_app.py", *args])

    def test_headless_without_display(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
        env.update(THIRDEYE_SKIP_DIAG="1", ARGV_OUT=self.argv_out)
        result = subprocess.run(
            ["bash", os.path.join(self.repo, "run_demo.sh"), "--headless", "--incidents"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(self.argv_out) as fh:
            self.assertEqual(fh.read().splitlines(), ["third_eye_app.py", "--headless", "--incidents"])


if __name__ == "__main__":
    unittest.main()
