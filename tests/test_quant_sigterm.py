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
        quant.run_logged(["sh", "-c", "echo $$ > {pidfile}; {child}"], {log!r}, echo=False)
    except KeyboardInterrupt:
        quant.stop_children(grace=20)
        raise SystemExit(130)
""")


def _child_after_sigterm(tmp_path, install, sig=signal.SIGTERM, ignore_sigint=False, child="exec sleep 300"):
    pidfile, log = tmp_path / "child.pid", tmp_path / "logs" / "x.log"
    script = PARENT.format(here=os.path.abspath(HERE), pidfile=pidfile, log=str(log), child=child,
                           install="quant.interrupt_on_sigterm()" if install else "")
    # ssh host 'cmd &' starts cmd with SIGINT ignored (a background job without job control)
    preexec = (lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)) if ignore_sigint else None
    t0 = time.time()
    parent = subprocess.Popen([sys.executable, "-c", script], preexec_fn=preexec)
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    child = int(pidfile.read_text())
    t0 = time.time()
    parent.send_signal(sig)                     # SIGTERM: what remote.py's kill delivers to the stage's process group
    try:
        parent.wait(timeout=60)
    except subprocess.TimeoutExpired:
        parent.kill()
    time.sleep(0.2)
    try:
        os.kill(child, 0)
        alive = True
        os.kill(child, signal.SIGKILL)          # the test's own cleanup
    except ProcessLookupError:
        alive = False
    return parent.returncode, alive, time.time() - t0


def test_sigterm_stops_the_children(tmp_path):
    rc, alive, _ = _child_after_sigterm(tmp_path, install=True)
    assert rc == 130 and not alive


def test_without_the_handler_the_child_survives(tmp_path):
    """The control: the failure this guards against, reproduced."""
    _, alive, _ = _child_after_sigterm(tmp_path, install=False)
    assert alive


def test_ctrl_c_in_a_single_threaded_step_stops_the_child(tmp_path):
    """The same unwinding hit Ctrl-C: run_logged dropped the child from the live set before stop_children ran."""
    rc, alive, _ = _child_after_sigterm(tmp_path, install=False, sig=signal.SIGINT)
    assert rc == 130 and not alive


def test_started_through_ssh_the_children_stop_promptly(tmp_path):
    """Started with SIGINT ignored, as `ssh host 'cmd &'` starts it, the children inherited the ignore: stop_children's
    SIGINT reached nothing and each level sat out its grace (20 s here) before SIGTERM. The child here is a Python
    process, like quant.py and convert.py, which keeps whatever SIGINT disposition it starts with."""
    rc, alive, took = _child_after_sigterm(tmp_path, install=True, ignore_sigint=True,
                                           child="exec python3 -c 'import time; time.sleep(300)'")
    assert rc == 130 and not alive and took < 10, took
