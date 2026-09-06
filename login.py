"""
login.py — Automatic Login Module
Handles form-based and JSON login flows automatically.
Extracts Bearer tokens, session cookies, and basic auth.
Feeds credentials directly into AuthManager.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger("bola.login")


@dataclass
class LoginResult:
    """Result of an automatic login attempt."""
    success: bool
    auth_string: str        # Ready to pass to --auth-a / --auth-b
    user_label: str
    error: Optional[str] = None


class AutoLogin:
    """
    Attempts automatic login against common API login endpoints.
    Supports:
      - JSON body login (most REST APIs: Juice Shop, crAPI)
      - Form-based login (DVWA)
      - Extracts Bearer token from response body
      - Extracts session cookie from Set-Cookie header
    """

    # Common login endpoint paths to try in order
    JSON_LOGIN_PATHS = (
        "/api/v1/auth/login",
        "/api/auth/login",
        "/rest/user/login",
        "/api/login",
        "/auth/login",
        "/login",
        "/api/v1/login",
        "/api/token",
        "/api/v1/token",
        "/graphql",
    )

    FORM_LOGIN_PATHS = (
        "/login.php",
        "/login",
        "/signin",
    )

    # Common JSON field names for tokens in responses
    TOKEN_FIELDS = (
        "token", "access_token", "accessToken",
        "jwt", "id_token", "auth_token", "key",
    )

    # Common JSON field names for username/password in request body
    USERNAME_FIELDS = ("email", "username", "user", "login", "identifier")
    PASSWORD_FIELDS = ("password", "pass", "passwd", "secret")

    def __init__(self, base_url: str, verify_ssl: bool = True, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def _try_json_login(
        self,
        path: str,
        username: str,
        password: str,
    ) -> Optional[LoginResult]:
        url = urljoin(self.base_url + "/", path.lstrip("/"))

        # Try each combination of username/password field names
        for u_field in self.USERNAME_FIELDS:
            for p_field in self.PASSWORD_FIELDS:
                payload = {u_field: username, p_field: password}
                try:
                    resp = requests.post(
                        url,
                        json=payload,
                        timeout=self.timeout,
                        verify=self.verify_ssl,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                    )
                    logger.debug(
                        "JSON login attempt: POST %s {%s, %s} -> %d",
                        url, u_field, p_field, resp.status_code,
                    )

                    if resp.status_code not in (200, 201):
                        continue

                    # Try to extract Bearer token from JSON response body
                    token = self._extract_token_from_body(resp)
                    if token:
                        logger.info("Bearer token extracted from %s", url)
                        return LoginResult(
                            success=True,
                            auth_string=f"Bearer {token}",
                            user_label="",
                        )

                    # Try to extract session cookie
                    cookie = self._extract_session_cookie(resp)
                    if cookie:
                        logger.info("Session cookie extracted from %s", url)
                        return LoginResult(
                            success=True,
                            auth_string=cookie,
                            user_label="",
                        )

                except requests.RequestException as exc:
                    logger.debug("JSON login failed at %s: %s", url, exc)
                    continue

        return None

    def _try_graphql_login(
        self,
        username: str,
        password: str,
    ) -> Optional[LoginResult]:
        """
        Attempt GraphQL mutation-based login.
        Tries common mutation names used in apps like crAPI.
        """
        url = urljoin(self.base_url + "/", "graphql")
        mutations = [
            f'mutation {{ login(email: "{username}", password: "{password}") {{ token }} }}',
            f'mutation {{ userLogin(email: "{username}", password: "{password}") {{ token access_token }} }}',
            f'mutation {{ signIn(username: "{username}", password: "{password}") {{ token }} }}',
        ]

        for mutation in mutations:
            try:
                resp = requests.post(
                    url,
                    json={"query": mutation},
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                if "errors" in data:
                    continue

                token = self._deep_find_token(data.get("data", {}))
                if token:
                    logger.info("GraphQL login token extracted.")
                    return LoginResult(
                        success=True,
                        auth_string=f"Bearer {token}",
                        user_label="",
                    )
            except Exception as exc:
                logger.debug("GraphQL login attempt failed: %s", exc)

        return None

    def _try_form_login(
        self,
        path: str,
        username: str,
        password: str,
    ) -> Optional[LoginResult]:
        """
        Handle form-based login (e.g. DVWA with PHPSESSID + user_token).
        First GET the login page to extract CSRF token, then POST.
        """
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        session = requests.Session()
        session.verify = self.verify_ssl

        try:
            get_resp = session.get(url, timeout=self.timeout)
            if get_resp.status_code != 200:
                return None

            # Extract CSRF/user_token from hidden input fields
            csrf_match = re.search(
                r'<input[^>]+name=["\']user_token["\'][^>]+value=["\']([^"\']+)["\']',
                get_resp.text,
                re.IGNORECASE,
            )
            csrf_token = csrf_match.group(1) if csrf_match else None

            form_data: dict = {
                "username": username,
                "password": password,
                "Login": "Login",
            }
            if csrf_token:
                form_data["user_token"] = csrf_token
                logger.debug("CSRF token found: %s", csrf_token)

            post_resp = session.post(
                url,
                data=form_data,
                timeout=self.timeout,
                allow_redirects=True,
            )
            logger.debug("Form login POST %s -> %d", url, post_resp.status_code)

            # DVWA returns 200 with a page — check for login failure markers
            if "Login failed" in post_resp.text or "incorrect" in post_resp.text.lower():
                return None

            # Extract session cookie
            cookie_str = "; ".join(
                f"{name}={value}"
                for name, value in session.cookies.items()
            )
            if cookie_str:
                logger.info("Form login success. Cookies: %s", cookie_str)
                return LoginResult(
                    success=True,
                    auth_string=cookie_str,
                    user_label="",
                )

        except requests.RequestException as exc:
            logger.debug("Form login failed at %s: %s", url, exc)

        return None

    def _extract_token_from_body(self, resp: requests.Response) -> Optional[str]:
        try:
            body = resp.json()
        except Exception:
            return None
        return self._deep_find_token(body)

    def _deep_find_token(self, obj: object, depth: int = 0) -> Optional[str]:
        if depth > 5:
            return None
        if isinstance(obj, dict):
            for field in self.TOKEN_FIELDS:
                if field in obj and isinstance(obj[field], str) and len(obj[field]) > 10:
                    return obj[field]
            for value in obj.values():
                result = self._deep_find_token(value, depth + 1)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = self._deep_find_token(item, depth + 1)
                if result:
                    return result
        return None

    def _extract_session_cookie(self, resp: requests.Response) -> Optional[str]:
        cookies = resp.cookies
        if not cookies:
            return None
        return "; ".join(f"{name}={value}" for name, value in cookies.items())

    def login(self, username: str, password: str, user_label: str) -> LoginResult:
        """
        Attempt login using all supported strategies in order:
        1. JSON login across common REST paths
        2. GraphQL mutation login
        3. Form-based login
        """
        logger.info("Auto-login: attempting login for User %s (%s)", user_label, username)

        # Try JSON login paths
        for path in self.JSON_LOGIN_PATHS:
            result = self._try_json_login(path, username, password)
            if result:
                result.user_label = user_label
                return result

        # Try GraphQL mutation login
        result = self._try_graphql_login(username, password)
        if result:
            result.user_label = user_label
            return result

        # Try form login paths
        for path in self.FORM_LOGIN_PATHS:
            result = self._try_form_login(path, username, password)
            if result:
                result.user_label = user_label
                return result

        return LoginResult(
            success=False,
            auth_string="",
            user_label=user_label,
            error=f"All login strategies failed for user: {username}",
        )