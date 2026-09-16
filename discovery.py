"""
discovery.py — Automatic Endpoint Discovery
Crawls a target web app and discovers API endpoints automatically.
No spec file required. Uses passive crawling, JS parsing,
and common path fuzzing.
"""

import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawler import Operation

logger = logging.getLogger("bola.discovery")

# Common API path patterns to probe
COMMON_API_PATHS = [
    "/api/users", "/api/user", "/api/users/{id}",
    "/api/orders", "/api/order", "/api/orders/{id}",
    "/api/products", "/api/product", "/api/products/{id}",
    "/api/baskets", "/api/basket", "/api/basket/{id}",
    "/api/cards", "/api/card", "/api/cards/{id}",
    "/api/addresses", "/api/address", "/api/addresses/{id}",
    "/api/accounts", "/api/account", "/api/accounts/{id}",
    "/api/profile", "/api/profiles", "/api/profiles/{id}",
    "/api/payments", "/api/payment", "/api/payments/{id}",
    "/api/v1/users", "/api/v1/users/{id}",
    "/api/v1/orders", "/api/v1/orders/{id}",
    "/api/v1/profile", "/api/v1/profiles/{id}",
    "/api/v2/users", "/api/v2/users/{id}",
    "/api/v2/orders", "/api/v2/orders/{id}",
    "/rest/user/whoami", "/rest/basket/{id}",
    "/rest/order-history", "/rest/orders/{id}",
    "/rest/products/{id}", "/rest/user/{id}",
    "/users", "/users/{id}", "/orders", "/orders/{id}",
    "/profile", "/profile/{id}", "/account", "/account/{id}",
    "/v1/users", "/v1/users/{id}", "/v1/orders", "/v1/orders/{id}",
    "/v2/users", "/v2/users/{id}", "/v2/orders", "/v2/orders/{id}",
]

# JS bundle patterns that reveal API routes
JS_ROUTE_PATTERNS = [
    r'["\']/(api|rest|v\d+)/[\w/{}]+["\']',
    r'axios\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
    r'fetch\(["\']([^"\']+)["\']',
    r'\.get\(["\']([^"\']+)["\']',
    r'\.post\(["\']([^"\']+)["\']',
    r'path:\s*["\']([^"\']+)["\']',
    r'url:\s*["\']([^"\']+)["\']',
    r'endpoint:\s*["\']([^"\']+)["\']',
]

HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


class EndpointDiscovery:
    """
    Automatically discovers API endpoints from a target web application.
    Strategy:
      1. Parse HTML for links and forms
      2. Extract routes from JavaScript bundles
      3. Probe common API path patterns
      4. Detect path parameters and build Operation objects
    """

    def __init__(
        self,
        base_url: str,
        session: Optional[requests.Session] = None,
        timeout: int = 10,
        probe_methods: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.probe_methods = probe_methods
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "BOLA-Framework/1.0",
            "Accept": "application/json, text/html, */*",
        })
        self.discovered_paths: set[str] = set()
        self.operations: list[Operation] = []

    def _get(self, url: str) -> Optional[requests.Response]:
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            return resp
        except requests.RequestException as exc:
            logger.debug("GET failed %s: %s", url, exc)
            return None

    def _probe(self, path: str, method: str = "GET") -> Optional[requests.Response]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                allow_redirects=False,
                headers={"Accept": "application/json"},
            )
            return resp
        except requests.RequestException:
            return None

    def _is_api_response(self, resp: requests.Response) -> bool:
        if resp is None:
            return False
        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            return True
        if resp.status_code in (200, 201, 401, 403):
            try:
                resp.json()
                return True
            except Exception:
                pass
        return False

    def _extract_js_urls(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        js_urls = []
        for script in soup.find_all("script", src=True):
            src = script["src"]
            if not src.startswith("http"):
                src = urljoin(self.base_url, src)
            js_urls.append(src)
        return js_urls

    def _extract_routes_from_js(self, js_content: str) -> set[str]:
        routes: set[str] = set()
        for pattern in JS_ROUTE_PATTERNS:
            matches = re.findall(pattern, js_content)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m.startswith("/") and len(m) > 2:
                            routes.add(m.split("?")[0])
                elif isinstance(match, str):
                    if match.startswith("/") and len(match) > 2:
                        routes.add(match.split("?")[0])
        # Filter to API-looking paths only
        api_routes = {
            r for r in routes
            if any(seg in r for seg in [
                "/api/", "/rest/", "/v1/", "/v2/", "/v3/",
                "/user", "/order", "/basket", "/product",
                "/account", "/profile", "/payment",
            ])
        }
        return api_routes

    def _has_path_param(self, path: str) -> bool:
        return bool(re.search(r"/\{[^}]+\}|/:\w+|/\d+$", path))

    def _normalize_path(self, path: str) -> str:
        # Convert :id style to {id} style
        path = re.sub(r"/:(\w+)", r"/{\1}", path)
        # Convert trailing numeric IDs to {id}
        path = re.sub(r"/(\d+)(/|$)", r"/{id}\2", path)
        return path

    def _build_operation(self, path: str, method: str, source: str) -> Operation:
        normalized = self._normalize_path(path)
        has_id = self._has_path_param(normalized)
        op_id = f"{method}:{normalized}"

        params = []
        id_rule = None
        if has_id:
            params.append({"name": "id", "in": "path", "required": True})
            id_rule = {"location": "path", "name": "id"}

        return Operation(
            operation_id=op_id,
            api_type="rest",
            method=method,
            path=normalized,
            parameters=params,
            object_identifier_extraction_rule=id_rule,
            tags=[source],
        )

    def _discover_from_html(self) -> set[str]:
        paths: set[str] = set()
        resp = self._get(self.base_url)
        if not resp:
            return paths

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract links
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith("/"):
                paths.add(href.split("?")[0])

        # Extract form actions
        for form in soup.find_all("form", action=True):
            action = form["action"]
            if action.startswith("/"):
                paths.add(action.split("?")[0])

        return paths

    def _discover_from_js(self) -> set[str]:
        paths: set[str] = set()
        resp = self._get(self.base_url)
        if not resp:
            return paths

        js_urls = self._extract_js_urls(resp.text)
        logger.info("Found %d JS bundles to scan.", len(js_urls))

        for js_url in js_urls:
            logger.debug("Scanning JS bundle: %s", js_url)
            js_resp = self._get(js_url)
            if js_resp and js_resp.status_code == 200:
                routes = self._extract_routes_from_js(js_resp.text)
                logger.debug("Found %d routes in %s", len(routes), js_url)
                paths.update(routes)

        return paths

    def _probe_common_paths(self) -> set[str]:
        paths: set[str] = set()
        logger.info("Probing %d common API paths...", len(COMMON_API_PATHS))

        for path in COMMON_API_PATHS:
            probe_path = re.sub(r"\{[^}]+\}", "1", path)
            resp = self._probe(probe_path, "GET")
            if resp and self._is_api_response(resp):
                logger.debug("Live endpoint found: GET %s -> %d", probe_path, resp.status_code)
                paths.add(path)

        return paths

    def _probe_methods_for_path(self, path: str) -> list[str]:
        live_methods: list[str] = []
        probe_path = re.sub(r"\{[^}]+\}", "1", path)

        for method in HTTP_METHODS:
            resp = self._probe(probe_path, method)
            if resp and resp.status_code not in (404, 405, 501):
                if self._is_api_response(resp) or resp.status_code in (200, 201, 401, 403):
                    live_methods.append(method)
                    logger.debug("Live: %s %s -> %d", method, probe_path, resp.status_code)

        return live_methods or ["GET"]

    def discover(self) -> list[Operation]:
        logger.info("Starting endpoint discovery on %s", self.base_url)
        all_paths: set[str] = set()

        # Step 1: HTML crawl
        logger.info("Step 1: HTML crawl...")
        html_paths = self._discover_from_html()
        logger.info("HTML crawl found %d paths.", len(html_paths))
        all_paths.update(html_paths)

        # Step 2: JS bundle route extraction
        logger.info("Step 2: JS bundle scanning...")
        js_paths = self._discover_from_js()
        logger.info("JS scan found %d API routes.", len(js_paths))
        all_paths.update(js_paths)

        # Step 3: Common path probing
        logger.info("Step 3: Common path probing...")
        probed_paths = self._probe_common_paths()
        logger.info("Probing found %d live endpoints.", len(probed_paths))
        all_paths.update(probed_paths)

        # Filter to API-relevant paths only
        api_paths = {
            p for p in all_paths
            if any(seg in p for seg in [
                "/api", "/rest", "/v1", "/v2", "/v3",
                "/user", "/order", "/basket", "/product",
                "/account", "/profile", "/payment", "/card",
            ])
        }

        logger.info("Total unique API paths to test: %d", len(api_paths))

        # Build operations
        seen: set[str] = set()
        for path in api_paths:
            normalized = self._normalize_path(path)
            if self.probe_methods:
                methods = self._probe_methods_for_path(normalized)
            else:
                methods = ["GET"]

            for method in methods:
                key = f"{method}:{normalized}"
                if key not in seen:
                    seen.add(key)
                    op = self._build_operation(normalized, method, "auto-discovery")
                    self.operations.append(op)

        logger.info("Discovery complete: %d operations found.", len(self.operations))
        return self.operations