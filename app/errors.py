"""Application errors raised by deterministic pipeline stages."""

from __future__ import annotations


class IntakeError(ValueError):
    """Raised when a PDF cannot be accepted for intake (validation / I/O)."""

    def __init__(self, message: str, *, code: str = "intake_error") -> None:
        super().__init__(message)
        self.code = code
