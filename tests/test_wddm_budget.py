"""The WDDM budget monitor's decisions (vllm_exl3_plugin/wddm_budget.py), without WSL or a GPU."""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_exl3_plugin import wddm_budget as wb  # noqa: E402

G = wb.GIB


class PickAdapter(unittest.TestCase):
    def test_the_one_carrying_our_allocations(self):
        self.assertEqual(wb.pick_adapter([0, 9 * G, G // 2], 8 * G), 1)

    def test_ambiguity_turns_it_off(self):
        self.assertIsNone(wb.pick_adapter([9 * G, 9 * G], 8 * G))

    def test_nothing_held_turns_it_off(self):
        self.assertIsNone(wb.pick_adapter([G], 0))
        self.assertIsNone(wb.pick_adapter([G // 2], G))


class Watch(unittest.TestCase):
    def test_under_budget_is_silent_but_tracks_headroom(self):
        w = wb.Watch()
        self.assertEqual(w.observe(11 * G, 10 * G, 0.0), [])
        self.assertEqual(w.observe(11 * G, 10 * G + G // 2, 1.0), [])
        self.assertEqual(w.least_headroom, G // 2)

    def test_crossing_warns_once_then_on_growth_then_recovers(self):
        w = wb.Watch()
        w.observe(11 * G, 10 * G, 0.0)
        (level, msg), = w.observe(10 * G, 10 * G + G // 10, 1.0)
        self.assertEqual(level, logging.WARNING)
        self.assertIn("WDDM budget exceeded", msg)
        self.assertEqual(w.observe(10 * G, 10 * G + G // 5, 2.0), [])  # grew 0.1 GiB: below the re-warn step
        (level, msg), = w.observe(10 * G, 10 * G + G // 2, 3.0)
        self.assertIn("still exceeded", msg)
        (level, msg), = w.observe(11 * G, 10 * G, 5.0)
        self.assertEqual(level, logging.INFO)
        self.assertIn("recovered after 4 s", msg)
        self.assertEqual((w.episodes, w.seconds_over, w.worst_over), (1, 4.0, G // 2))

    def test_long_episodes_are_reminded(self):
        w = wb.Watch()
        w.observe(10 * G, 11 * G, 0.0)
        self.assertEqual(w.observe(10 * G, 11 * G, 100.0), [])
        self.assertEqual(len(w.observe(10 * G, 11 * G, wb._REWARN_SECONDS + 1)), 1)


class Enabled(unittest.TestCase):
    def test_off_switch(self):
        os.environ["EXL3_WDDM_MONITOR"] = "0"
        try:
            self.assertFalse(wb.enabled())
        finally:
            del os.environ["EXL3_WDDM_MONITOR"]


if __name__ == "__main__":
    unittest.main()
