"""
auth.py — Authentication Manager Module
Maintains two parallel, isolated session objects for User A and User B.
Supports Bearer Token, Cookie-based, and Basic authentication.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger("bola.auth")


class AuthType(Enum):
    BEARER = "bearer"
    COOKIE = "cookie"
    BASIC = "basic"


@dataclass
class AuthConfig:
    raw: str  # e.g. "Bearer xyz" | "session=abc" | "user:pass"

    def detect_type(self) -> AuthType:
        lower = self.raw.lower()
        if lower.startswith("bearer "):
            return AuthType.BEARER
        if "=" in self.raw and not ":" in self.raw.split("=")[0]:
            return AuthType.COOKIE
        return AuthType.BASIC

    def apply_to_session(self, session: requests.Session) -> None:
        auth_type = self.detect_type()
        if auth_type == AuthType.BEARER:
            token = self.raw.split(" ", 1)[1]
            session.headers.update({"Authorization": f"Bearer {token}"})
            logger.debug("Applied Bearer token to session.")
        elif auth_type == AuthType.COOKIE:
            for pair in self.raw.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    name, value = pair.split("=", 1)
                    session.cookies.set(name.strip(), value.strip())
            logger.debug("Applied Cookie auth to session.")
        elif auth_type == AuthType.BASIC:
            if ":" in self.raw:
                user, password = self.raw.split(":", 1)
                session.auth = (user.strip(), password.strip())
                logger.debug("Applied Basic auth to session.")
            else:
                raise ValueError(
                    f"Basic auth format must be 'username:password', got: {self.raw!r}"
                )


class AuthManager:
    """
    Manages two fully authenticated, isolated requests.Session objects.
    Call authenticate() before passing sessions to the Request Engine.
    """

    def __init__(self, auth_a: str, auth_b: str, verify_ssl: bool = True):
        self.auth_a_config = AuthConfig(auth_a)
        self.auth_b_config = AuthConfig(auth_b)
        self.verify_ssl = verify_ssl
        self._session_a: Optional[requests.Session] = None
        self._session_b: Optional[requests.Session] = None

    def authenticate(self) -> None:
        self._session_a = self._build_session(self.auth_a_config, "A")
        self._session_b = self._build_session(self.auth_b_config, "B")
        logger.info("Both sessions authenticated successfully.")

    def _build_session(self, config: AuthConfig, label: str) -> requests.Session:
        session = requests.Session()
        session.verify = self.verify_ssl
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "BOLA-Framework/1.0",
        })
        try:
            config.apply_to_session(session)
        except Exception as exc:
            raise RuntimeError(f"Failed to configure session {label}: {exc}") from exc
        return session

    @property
    def session_a(self) -> requests.Session:
        if self._session_a is None:
            raise RuntimeError("Sessions not initialized. Call authenticate() first.")
        return self._session_a

    @property
    def session_b(self) -> requests.Session:
        if self._session_b is None:
            raise RuntimeError("Sessions not initialized. Call authenticate() first.")
        return self._session_b

    def close(self) -> None:
        if self._session_a:
            self._session_a.close()
        if self._session_b:
            self._session_b.close()
        logger.debug("Auth sessions closed.")