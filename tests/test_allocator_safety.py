"""allocator_safety: refuse expandable segments on unpatched torch across devices (PyTorch #196258)."""
import hashlib
import os

import pytest
import torch

from vllm_exl3_plugin import allocator_safety as a


def _state(monkeypatch, on, devices, fixed):
    monkeypatch.setattr(a, "expandable_segments_on", lambda: on)
    monkeypatch.setattr(a, "devices_with_memory", lambda: devices)
    monkeypatch.setattr(a, "c10_fix_present", lambda: fixed)


@pytest.mark.parametrize("on, devices, fixed", [
    (False, [0, 1], False),   # expandable segments off
    (True, [0], False),       # one device per process: vLLM's workers
    (True, [0, 1], True),     # the fix is in
])
def test_safe_configurations_pass(monkeypatch, on, devices, fixed):
    _state(monkeypatch, on, devices, fixed)
    a.check("test")


def test_unpatched_expandable_across_devices_is_refused_with_both_ways_out(monkeypatch):
    _state(monkeypatch, True, [0, 1], False)
    with pytest.raises(RuntimeError) as e:
        a.check("test")
    msg = str(e.value)
    assert "devices [0, 1]" in msg and "torch-c10cuda-fix.sh" in msg and "PYTORCH_CUDA_ALLOC_CONF" in msg


def test_fix_probe_reads_the_stamp_against_the_library(monkeypatch, tmp_path):
    """The stamp must hold the library's sha256: a stamp beside a since-reinstalled library does not count."""
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    lib = lib_dir / "libc10_cuda.so"
    lib.write_bytes(b"patched build")
    monkeypatch.setattr(a.torch, "__file__", str(tmp_path / "__init__.py"))
    assert not a.c10_fix_present(), "no stamp, no fix"
    stamp = f"{lib}.fix-{a.FIX}"
    open(stamp, "w").write(hashlib.sha256(b"patched build").hexdigest() + "  -\n")
    assert a.c10_fix_present()
    lib.write_bytes(b"a reinstalled stock build")
    assert not a.c10_fix_present()


def test_the_live_venv_carries_the_fix():
    """This host's venv is patched (tools/torch-c10cuda-fix.sh); if it is not, the probe disagrees with --check."""
    lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libc10_cuda.so")
    if not os.path.exists(f"{lib}.fix-{a.FIX}"):
        pytest.skip("this torch has no fix stamp")
    assert a.c10_fix_present()
