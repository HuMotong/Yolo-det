from __future__ import annotations

import unittest
from pathlib import Path

from mildew_seg.audit import discover_records
from mildew_seg.config import load_config


class RealDataRegressionTests(unittest.TestCase):
    @unittest.skipUnless(Path("mildew").is_dir(), "Local mildew dataset is unavailable")
    def test_known_repairs_are_reported(self) -> None:
        config = load_config("configs/default.yaml")
        records, global_issues = discover_records(config, allow_repair=True)
        self.assertEqual(global_issues, [])
        self.assertEqual(len(records), 148)
        inferred = [record for record in records if record.seed_inferred]
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0].json_path.name, "famei (89).json")
        degenerate_count = sum(
            issue["code"] == "degenerate_polygon"
            for record in records
            for issue in record.issues
        )
        self.assertEqual(degenerate_count, 10)


if __name__ == "__main__":
    unittest.main()
