"""The margin gate's verdicts (bench/run.py judge_margin), on synthetic runs: no GPU needed."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import run  # noqa: E402

GIB = 1 << 30


def _run(**phase):
    row = dict(phase="long_prefill", seconds=1.0, error=None, allocated_peak=10 * GIB + GIB // 2,
               allocated_after=10 * GIB, reserved_peak=11 * GIB, alloc_retries=0, ooms=0, nontorch_growth=0)
    row.update(phase)
    return dict(cap=12 * GIB, start=dict(allocated=10 * GIB, reserved=10 * GIB, nontorch=0), total=16 * GIB,
                nontorch_share=GIB // 8, phases=[row])


class JudgeMargin(unittest.TestCase):
    def test_room_to_spare_passes(self):
        self.assertEqual(run.judge_margin(_run()), [])

    def test_peak_near_the_cap_is_thin_before_it_ooms(self):
        # 1.7 of 2 GiB room: past the limit, nothing failed yet
        problems = run.judge_margin(_run(allocated_peak=11 * GIB + int(0.7 * GIB)))
        self.assertEqual(len(problems), 1)
        self.assertIn("room", problems[0])

    def test_oom_fails_whatever_the_share(self):
        problems = run.judge_margin(_run(error="OutOfMemoryError: CUDA out of memory.", ooms=1))
        self.assertEqual(len(problems), 1)
        self.assertIn("OutOfMemoryError", problems[0])

    def test_non_torch_growth_past_its_share_fails(self):
        problems = run.judge_margin(_run(nontorch_growth=GIB // 4))
        self.assertEqual(len(problems), 1)
        self.assertIn("outside torch", problems[0])

    def test_retries_alone_do_not_fail(self):
        self.assertEqual(run.judge_margin(_run(alloc_retries=3)), [])


if __name__ == "__main__":
    unittest.main()
