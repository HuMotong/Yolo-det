from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class TrainEntrypointTests(unittest.TestCase):
    def test_train_file_can_be_executed_directly(self) -> None:
        train_file = Path("src/mildew_seg/train.py").resolve()
        result = subprocess.run(
            [sys.executable, str(train_file), "--help"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: mildew-seg train", result.stdout)
        self.assertIn("--resume", result.stdout)


if __name__ == "__main__":
    unittest.main()
