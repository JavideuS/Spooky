"""
The per-run timeout in quantum.benchmark.sweep_runner.

A sweep is a long unattended job, so a run that overruns must be recorded and
skipped rather than blocking everything behind it. These tests are fast: they
use sub-second deadlines and busy-wait in Python, never a solver.
"""

import signal
import time

import pytest

from quantum.benchmark.sweep_runner import RunTimeout, _run_timeout

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "setitimer"), reason="POSIX itimer required"
)


def _spin(seconds):
    """Busy-wait in interpretable Python so a signal can actually be delivered."""
    end = time.time() + seconds
    while time.time() < end:
        pass


def test_overrunning_run_is_aborted():
    started = time.time()
    with pytest.raises(RunTimeout, match="exceeded run_timeout_sec"):
        with _run_timeout(0.2):
            _spin(5)
    assert time.time() - started < 3, "did not abort near the deadline"


def test_a_run_inside_the_budget_is_untouched():
    with _run_timeout(5):
        _spin(0.05)


def test_a_swallowed_delivery_is_retried():
    """The failure this exists for: SIGALRM was delivered inside a garbage
    collection callback, where Python discards exceptions, so the single
    one-shot alarm was the only attempt and the run continued past its
    deadline. The timer repeats now, so a lost delivery is not the end of it.
    """
    import quantum.benchmark.sweep_runner as sr

    deliveries = []
    original = sr._TIMEOUT_RETRY_SECONDS
    sr._TIMEOUT_RETRY_SECONDS = 0.2
    try:
        with pytest.raises(RunTimeout):
            with _run_timeout(0.2):
                for _ in range(10):
                    try:
                        _spin(5)
                    except RunTimeout:
                        # stand in for a context that eats the exception
                        deliveries.append(1)
                        if len(deliveries) >= 3:
                            raise
    finally:
        sr._TIMEOUT_RETRY_SECONDS = original

    assert len(deliveries) >= 3, "timer did not re-arm after a swallowed delivery"


def test_timer_is_disarmed_afterwards():
    """A leaked itimer would fire during an unrelated later run and abort it."""
    with _run_timeout(10):
        pass
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    assert remaining == 0 and interval == 0


def test_handler_is_restored_afterwards():
    sentinel = signal.getsignal(signal.SIGALRM)
    with _run_timeout(10):
        pass
    assert signal.getsignal(signal.SIGALRM) is sentinel


def test_no_timeout_configured_is_a_passthrough():
    for disabled in (None, 0):
        with _run_timeout(disabled):
            _spin(0.05)
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
        assert remaining == 0
