"""SIGTERM stops quant.py's children, as Ctrl-C does. remote.py ends a stage by signalling its process group; the
children run in their own sessions, so before this they never saw it and were left running (2026-11-20)."""

import os
import signal
import subprocess
import sys
import textwrap
import time

HERE = os.path.join(os.path.dirname(__file__), "..", "quantization")

PARENT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {here!r})
    import quant
    {install}
    try:
        quant.run_logged(["sh", "-c", "echo $$ > {pidfile}; exec sleep 300"], {log!r}, echo=False)
    except KeyboardInterrupt:
        quant.stop_children(grace=5)
        raise SystemExit(130)
""")


def _child_after_sigterm(tmp_path, install, sig=signal.SIGTERM):
    pidfile, log = tmp_path / "child.pid", tmp_path / "logs" / "x.log"
    script = PARENT.format(here=os.path.abspath(HERE), pidfile=pidfile, log=str(log),
                           install="quant.interrupt_on_sigterm()" if install else "")
    parent = subprocess.Popen([sys.executable, "-c", script])
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    child = int(pidfile.read_text())
    parent.send_signal(sig)                     # SIGTERM: what remote.py's kill delivers to the stage's process group
    try:
        parent.wait(timeout=20)
    except subprocess.TimeoutExpired:
        parent.kill()
    time.sleep(0.2)
    try:
        os.kill(child, 0)
        alive = True
        os.kill(child, signal.SIGKILL)          # the test's own cleanup
    except ProcessLookupError:
        alive = False
    return parent.returncode, alive


def test_sigterm_stops_the_children(tmp_path):
    rc, alive = _child_after_sigterm(tmp_path, install=True)
    assert rc == 130 and not alive


def test_without_the_handler_the_child_survives(tmp_path):
    """The control: the failure this guards against, reproduced."""
    _, alive = _child_after_sigterm(tmp_path, install=False)
    assert alive


def test_ctrl_c_in_a_single_threaded_step_stops_the_child(tmp_path):
    """The same unwinding hit Ctrl-C: run_logged dropped the child from the live set before stop_children ran."""
    rc, alive = _child_after_sigterm(tmp_path, install=False, sig=signal.SIGINT)
    assert rc == 130 and not alive
