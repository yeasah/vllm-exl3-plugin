"""bench.fixtures._prune: a fixture build leaves one generation per base checkpoint and recipe."""
import os

from bench import fixtures


def test_build_removes_older_generations_of_the_same_fixture(tmp_path):
    keep = "org-model-at-rev-blockq-v1-aaaaaaaaaaaa"
    for name in (keep, "org-model-at-rev-blockq-v1-bbbbbbbbbbbb", "org-model-at-rev-blockq-v0-cccccccccccc",
                 "org-model-at-rev-trellis-v1-dddddddddddd", "org-other-at-rev-blockq-v1-eeeeeeeeeeee",
                 "org-model-at-rev-blockq-v1-ffffffffffff.building-999999999",
                 f"org-model-at-rev-blockq-v1-gggggggggggg.building-{os.getpid()}"):
        (tmp_path / name).mkdir()
    fixtures._prune(str(tmp_path / keep))
    left = sorted(os.listdir(tmp_path))
    assert left == sorted([keep, "org-model-at-rev-trellis-v1-dddddddddddd", "org-other-at-rev-blockq-v1-eeeeeeeeeeee",
                           f"org-model-at-rev-blockq-v1-gggggggggggg.building-{os.getpid()}"]), left
