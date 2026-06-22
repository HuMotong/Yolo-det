from __future__ import annotations

import unittest

from mildew_seg.geometry import (
    build_tiles,
    clip_polygon_to_rect,
    polygon_area,
    polygon_centroid,
)


class GeometryTests(unittest.TestCase):
    def test_polygon_area_and_centroid(self) -> None:
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertAlmostEqual(polygon_area(polygon), 100.0)
        self.assertEqual(polygon_centroid(polygon), (5.0, 5.0))

    def test_clip_polygon_to_rectangle(self) -> None:
        polygon = [(-5.0, 5.0), (5.0, -5.0), (15.0, 5.0), (5.0, 15.0)]
        clipped = clip_polygon_to_rect(polygon, 0.0, 0.0, 10.0, 10.0)
        self.assertGreaterEqual(len(clipped), 4)
        self.assertAlmostEqual(polygon_area(clipped), 100.0)
        self.assertTrue(all(0 <= x <= 10 and 0 <= y <= 10 for x, y in clipped))

    def test_overlapping_tiles_have_unique_core_owner(self) -> None:
        tiles = build_tiles((50, 20, 350, 380), 400, 400, 128, 32)
        for y in range(20, 380, 7):
            for x in range(50, 350, 7):
                owners = [tile for tile in tiles if tile.owns((x, y))]
                self.assertEqual(len(owners), 1, (x, y, owners))


if __name__ == "__main__":
    unittest.main()
