"""
Manages a qiskit_ibm_runtime.Session across a PennylaneSolver.solve() call's
whole windowed loop (device="qiskit.remote" only), so only the first
window's job waits in IBM's public queue — subsequent jobs within the
session are prioritized instead of re-queuing.

A new IBMHardwareDevice (see quantum.hardware.ibm_device) is constructed per
window, since wires/backend can change window to window — but the
underlying Session should outlive individual devices and windows, reused
until the backend changes. That mismatched lifetime is why session state
lives here as its own object owned by PennylaneSolver, rather than on
IBMHardwareDevice or as loose instance attributes on the solver itself.
"""

from quantum.utils.logger import get_logger


class IBMSessionManager:
    """
    Args:
        use_session: Whether to open a Session at all. If False, get_session()
            always returns None (today's per-job public-queue behavior).
        session_max_time: Forwarded to qiskit_ibm_runtime.Session's max_time
            (seconds, or a string like "2h 30m"). None uses IBM's own
            default (900s).
    """

    def __init__(self, use_session=True, session_max_time=None):
        self.use_session = use_session
        self.session_max_time = session_max_time
        self.logger = get_logger()
        self._session = None
        self._session_backend_name = None
        self._unavailable = False  # sticky: don't retry every window once denied

    def get_session(self, backend):
        """
        Return an open Session for backend, reusing it if one is already
        open on the same backend.

        Requires a plan that supports Sessions (IBM's Open plan doesn't;
        Pay-As-You-Go and above do). If Session(...) raises for any reason —
        wrong plan, network issue, whatever — this logs once and returns
        None for the rest of this manager's lifetime, so the caller falls
        back to per-job public queuing rather than crashing the run.
        """
        if not self.use_session:
            if not self._unavailable:  # log once, not every window
                self.logger.standard(
                    "🌐 Session: DISABLED (use_session=False) — per-job public queuing"
                )
                self._unavailable = True  # reuse the sticky flag to avoid re-logging
            return None
        if self._unavailable:
            return None
        if self._session is not None and self._session_backend_name == backend.name:
            self.logger.standard(f"🔐 Session: ACTIVE — reusing session on {backend.name}")
            return self._session

        # Backend changed mid-solve (a later window needed more qubits than
        # the session's backend has) — close the old reservation first.
        self.close()

        from qiskit_ibm_runtime import Session

        try:
            session_kwargs = {"backend": backend}
            if self.session_max_time is not None:
                session_kwargs["max_time"] = self.session_max_time
            self._session = Session(**session_kwargs)
            self._session_backend_name = backend.name
            self.logger.standard(
                f"🔐 Session: ACTIVE — opened new IBM Runtime Session on {backend.name}"
            )
        except Exception as exc:
            self.logger.minimal(
                f"🌐 Session: UNAVAILABLE — could not open an IBM Runtime Session "
                f"({exc}) — falling back to per-job public queuing for the rest "
                "of this run. This is expected on IBM's Open plan; Sessions "
                "need Pay-As-You-Go or above."
            )
            self._session = None
            self._unavailable = True
        return self._session

    def close(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception as exc:
                self.logger.standard(f"(non-fatal) Session close failed: {exc}")
            self._session = None
            self._session_backend_name = None
