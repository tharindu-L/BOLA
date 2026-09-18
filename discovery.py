"""
discovery.py — Automatic Endpoint Discovery
Crawls a target web app and discovers API operations automatically, with
no OpenAPI/GraphQL spec supplied. Feeds the same unified Operation model
used by OpenAPI and GraphQL introspection (crawler.py), so BOLA detection
logic is identical across all three modes.

Discovery strategy is generic and protocol/convention-driven — it never
hardcodes a literal endpoint list scraped from known test applications.
It combines:
  1. Well-known API-documentation endpoints (OpenAPI/Swagger conventions)
     — if a spec is actually published, hand it straight to RESTCrawler
     instead of guessing.
  2. HTML crawl (links/forms) + sitemap.xml/robots.txt (standard web
     discovery surfaces).
  3. JS bundle scanning for route-like string literals.
  4. ID-variant expansion: every discovered path is also tried with an
     identifier segment appended/substituted, so BOLA testing can reach
     the "/resource/{id}" shape even when only "/resource" was linked.
"""

import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawler import Operation, RESTCrawler

logger = logging.getLogger("bola.discovery")

# Generic, convention-based roots — not application-specific endpoints.
# These are industry-standard locations for API documentation and the
# top-level API namespace prefix, used only to bootstrap discovery.
API_DOC_CANDIDATES = [
    "/openapi.json", "/openapi.yaml", "/openapi.yml",
    "/swagger.json", "/swagger.yaml",
    "/v2/api-docs", "/v3/api-docs",
    "/api-docs", "/api/openapi.json", "/api/swagger.json",
]

API_ROOT_PREFIXES = ("/api", "/rest", "/v1", "/v2", "/v3", "/graphql")

JS_ROUTE_PATTERNS = [
    # Index 0: "bare literal" — no call context, so it's restricted to
    # strings already shaped like an API root path (see _extract_routes_from_js).
    r'["\']/?(api|rest|v\d+)/[\w/{}\-]+["\']',
    # Everything below is scoped to an explicit HTTP-call/config context
    # (fetch/axios/.get(/.post(/url:/path:/endpoint:), or — as a last
    # resort for minified bundles where the call site itself is mangled —
    # any quoted relative multi-segment path literal. All of these are
    # only accepted as candidates; a live JSON probe decides the rest.
    r'axios\.(get|post|put|delete|patch)\(["\'`]([^"\'`]+)["\'`]',
    r'fetch\(["\'`]([^"\'`]+)["\'`]',
    r'\.get\(["\'`]([^"\'`]+)["\'`]',
    r'\.post\(["\'`]([^"\'`]+)["\'`]',
    r'path:\s*["\'`]([^"\'`]+)["\'`]',
    r'url:\s*["\'`]([^"\'`]+)["\'`]',
    r'endpoint:\s*["\'`]([^"\'`]+)["\'`]',
    r'["\']([a-zA-Z][a-zA-Z0-9_\-]{1,30}/[a-zA-Z0-9_\-/{}]{1,60})["\']',
]

HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]

_UUID_SEGMENT_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_NUMERIC_SEGMENT_RE = re.compile(r"^\d+$")


class EndpointDiscovery:

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
            return self.session.get(url, timeout=self.timeout, allow_redirects=True)
        except requests.RequestException as exc:
            logger.debug("GET failed %s: %s", url, exc)
            return None

    def _probe(self, path: str, method: str = "GET") -> Optional[requests.Response]:
        url = f"{self.base_url}{path}"
        try:
            return self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                allow_redirects=False,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException:
            return None

    def _is_api_response(self, resp: Optional[requests.Response]) -> bool:
        if resp is None:
            return False
        ct = resp.headers.get("Content-Type", "")
        if "html" in ct:
            return False
        if "json" in ct:
            return True
        if resp.status_code in (401, 403):
            try:
                resp.json()
                return True
            except Exception:
                return False
        if resp.status_code in (200, 201, 400):
            if len(resp.content) > 50000:
                return False
            try:
                resp.json()
                return True
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------
    # 1. Published API documentation (OpenAPI/Swagger convention paths)
    # ------------------------------------------------------------------

    def try_discover_openapi_spec(self) -> list[Operation]:
        """If the target actually publishes an OpenAPI/Swagger document at
        a conventional location, parse it with RESTCrawler directly rather
        than guessing endpoints — far higher fidelity than path fuzzing."""
        for doc_path in API_DOC_CANDIDATES:
            resp = self._get(f"{self.base_url}{doc_path}")
            if resp is None or resp.status_code != 200:
                continue
            try:
                spec = resp.json()
            except Exception:
                continue
            if "paths" not in spec or ("openapi" not in spec and "swagger" not in spec):
                continue
            logger.info("Discovered published OpenAPI/Swagger spec at %s", doc_path)
            return self._operations_from_spec_dict(spec)
        return []

    def _operations_from_spec_dict(self, spec: dict) -> list[Operation]:
        crawler = RESTCrawler.__new__(RESTCrawler)  # bypass file-path constructor
        crawler.spec_path = None
        crawler.spec = spec
        operations: list[Operation] = []
        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            shared_params = path_item.get("parameters", [])
            for method in ("get", "post", "put", "patch", "delete"):
                op_data = path_item.get(method)
                if not op_data:
                    continue
                op_data = crawler._resolve_ref(op_data)
                op_params = shared_params + op_data.get("parameters", [])
                resolved_params = [crawler._resolve_ref(p) for p in op_params]
                request_body = op_data.get("requestBody")
                body_schema = None
                sample_payload = None
                if request_body:
                    rb = crawler._resolve_ref(request_body)
                    content = rb.get("content", {})
                    json_content = content.get("application/json", {})
                    body_schema = crawler._resolve_ref(json_content.get("schema"))
                    sample_payload = crawler._build_sample_payload(body_schema)
                op_id = op_data.get("operationId") or f"{method.upper()}:{path}"
                param_rules = crawler._extract_param_id_rules(resolved_params)
                body_rules = crawler._extract_body_id_rules(sample_payload)
                all_rules = param_rules + body_rules
                primary = next((r for r in all_rules if r["location"] == "path"), None) \
                    or (all_rules[0] if all_rules else None)
                operations.append(Operation(
                    operation_id=op_id,
                    api_type="rest",
                    method=method.upper(),
                    path=path,
                    parameters=resolved_params,
                    request_body_schema=body_schema,
                    sample_payload=sample_payload,
                    object_identifier_extraction_rule=primary,
                    object_identifier_rules=all_rules,
                    tags=op_data.get("tags", []) + ["auto-discovered-spec"],
                    is_write=method.upper() in {"POST", "PUT", "PATCH", "DELETE"},
                ))
        return operations

    # ------------------------------------------------------------------
    # 2. Standard web discovery surfaces
    # ------------------------------------------------------------------

    def _extract_js_urls(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        js_urls = []
        for script in soup.find_all("script", src=True):
            src = script["src"]
            if not src.startswith("http"):
                src = urljoin(self.base_url, src)
            js_urls.append(src)
        return js_urls

    def _normalize_js_route_candidate(self, raw: str) -> Optional[str]:
        """
        Bundled SPA code frequently builds request URLs by concatenating a
        base host with a *relative* path literal (e.g. Angular's
        `environment.hostServer + 'rest/user/login'`), so the string found
        in the bundle often has no leading slash at all. Normalize both
        shapes to an absolute path so the API_ROOT_PREFIXES filter below
        can recognize them either way, instead of silently dropping every
        relative literal (which is the common case for SPA frameworks).
        """
        candidate = raw.split("?")[0].split("#")[0]
        if candidate.startswith(("http://", "https://")):
            parsed = urlparse(candidate)
            candidate = parsed.path
        if not candidate or len(candidate) < 3:
            return None
        if not candidate.startswith("/"):
            candidate = "/" + candidate
        # Reject anything that isn't a plausible URL path (no spaces, no
        # obvious non-path characters like '{' from JS object literals
        # unless it's a route placeholder such as '/{id}').
        if re.search(r"[\s<>\"'`]", candidate):
            return None
        # Reject obvious static-asset paths — a generic extension check,
        # not an application-specific signature, that keeps the live-probe
        # candidate set from being swamped by image/font/stylesheet URLs
        # that happen to match the bare relative-path pattern.
        if re.search(r"\.(png|jpe?g|gif|svg|ico|css|woff2?|ttf|eot|map|mp4|webp)$", candidate, re.IGNORECASE):
            return None
        return candidate

    def _extract_routes_from_js(self, js_content: str) -> set[str]:
        # The first pattern is a "bare literal" scan with no surrounding
        # call context, so it's restricted to strings that already look
        # like an API root path. The rest of the patterns only match
        # inside an explicit HTTP-call context (fetch/axios/.get(/.post(/
        # url:/path:/endpoint:), which is itself strong evidence the
        # string is a request path even without a leading "/api"/"/rest" —
        # SPA frameworks routinely concatenate a relative literal like
        # 'basket/' onto a base-URL constant defined elsewhere in the
        # bundle. Anything gathered here still has to pass a live JSON
        # probe later, so over-collecting here is safe.
        prefix_gated_routes: set[str] = set()
        call_context_routes: set[str] = set()

        gated_matches = re.findall(JS_ROUTE_PATTERNS[0], js_content)
        for match in gated_matches:
            candidates = match if isinstance(match, tuple) else (match,)
            for m in candidates:
                normalized = self._normalize_js_route_candidate(m)
                if normalized:
                    prefix_gated_routes.add(normalized)

        for pattern in JS_ROUTE_PATTERNS[1:]:
            matches = re.findall(pattern, js_content)
            for match in matches:
                candidates = match if isinstance(match, tuple) else (match,)
                for m in candidates:
                    normalized = self._normalize_js_route_candidate(m)
                    if normalized and normalized.count("/") >= 1:
                        call_context_routes.add(normalized)

        api_routes = {r for r in prefix_gated_routes if r.startswith(API_ROOT_PREFIXES)}
        api_routes |= call_context_routes
        return api_routes

    def _discover_from_html(self) -> set[str]:
        paths: set[str] = set()
        resp = self._get(self.base_url)
        if not resp:
            return paths
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith("/"):
                paths.add(href.split("?")[0])
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
            js_resp = self._get(js_url)
            if js_resp and js_resp.status_code == 200:
                routes = self._extract_routes_from_js(js_resp.text)
                logger.debug("Found %d routes in %s", len(routes), js_url)
                paths.update(routes)
        return paths

    def _discover_from_sitemap(self) -> set[str]:
        paths: set[str] = set()
        for sitemap_path in ("/sitemap.xml", "/robots.txt"):
            resp = self._get(f"{self.base_url}{sitemap_path}")
            if not resp or resp.status_code != 200:
                continue
            for match in re.findall(r"https?://[^\s<>\"]+", resp.text):
                parsed = urlparse(match)
                if parsed.netloc == urlparse(self.base_url).netloc and parsed.path:
                    paths.add(parsed.path)
            for match in re.findall(r"(?im)^disallow:\s*(\S+)", resp.text):
                paths.add(match)
        return paths

    # ------------------------------------------------------------------
    # 3. ID-variant expansion — turn discovered "collection" paths into
    #    testable "member" paths without guessing new resource names.
    # ------------------------------------------------------------------

    def _has_path_param(self, path: str) -> bool:
        return bool(re.search(r"/\{[^}]+\}", path))

    def _normalize_path(self, path: str) -> str:
        segments = path.split("/")
        normalized = []
        for seg in segments:
            if not seg:
                normalized.append(seg)
                continue
            clean = seg.rstrip("/")
            if _NUMERIC_SEGMENT_RE.match(clean) or _UUID_SEGMENT_RE.match(clean):
                normalized.append("{id}")
            elif re.match(r"^:\w+$", clean):
                normalized.append("{" + clean[1:] + "}")
            else:
                normalized.append(seg)
        return "/".join(normalized)

    def _expand_with_id_variant(self, path: str) -> set[str]:
        """Given a discovered collection-shaped path (no id segment yet),
        propose a member-shaped variant by appending a generic {id}
        placeholder — this lets BOLA testing reach 'GET /resource/{id}'
        even though only 'GET /resource' was ever linked in the UI."""
        variants = {path}
        if not self._has_path_param(path) and not path.rstrip("/").endswith(tuple(API_ROOT_PREFIXES)):
            variants.add(path.rstrip("/") + "/{id}")
        return variants

    def _build_operation(self, path: str, method: str, source: str) -> Operation:
        normalized = self._normalize_path(path)
        has_id = self._has_path_param(normalized)
        params = []
        id_rule = None
        rules = []
        if has_id:
            for placeholder in re.findall(r"\{([^}]+)\}", normalized):
                params.append({"name": placeholder, "in": "path", "required": True})
                rules.append({"location": "path", "name": placeholder})
            id_rule = rules[0] if rules else None
        return Operation(
            operation_id=f"{method}:{normalized}",
            api_type="rest",
            method=method,
            path=normalized,
            parameters=params,
            object_identifier_extraction_rule=id_rule,
            object_identifier_rules=rules,
            tags=[source],
            is_write=method in {"POST", "PUT", "PATCH", "DELETE"},
        )

    def _probe_methods_for_path(self, path: str) -> list[str]:
        live_methods: list[str] = []
        probe_path = re.sub(r"\{[^}]+\}", "1", path)
        for method in HTTP_METHODS:
            resp = self._probe(probe_path, method)
            if resp and resp.status_code not in (404, 405, 501):
                if self._is_api_response(resp):
                    live_methods.append(method)
                    logger.debug("Live: %s %s -> %d", method, probe_path, resp.status_code)
        return live_methods or ["GET"]

    def discover(self) -> list[Operation]:
        logger.info("Starting endpoint discovery on %s", self.base_url)

        logger.info("Step 0: checking for a published OpenAPI/Swagger document...")
        spec_ops = self.try_discover_openapi_spec()
        if spec_ops:
            logger.info("Using published spec: %d operations found, skipping fuzzy discovery.", len(spec_ops))
            self.operations = spec_ops
            return self.operations

        all_paths: set[str] = set()

        logger.info("Step 1: HTML crawl...")
        html_paths = self._discover_from_html()
        logger.info("HTML crawl found %d paths.", len(html_paths))
        all_paths.update(html_paths)

        logger.info("Step 2: JS bundle scanning...")
        js_paths = self._discover_from_js()
        logger.info("JS scan found %d API routes.", len(js_paths))
        all_paths.update(js_paths)

        logger.info("Step 3: sitemap/robots.txt scanning...")
        sitemap_paths = self._discover_from_sitemap()
        logger.info("Sitemap/robots scan found %d paths.", len(sitemap_paths))
        all_paths.update(sitemap_paths)

        # Every candidate still has to pass a live JSON probe below before
        # it becomes an Operation, so we don't pre-filter discovered paths
        # by prefix here — a relative literal pulled from a JS bundle
        # (e.g. "basket/") is exactly as valid a candidate as "/api/basket"
        # and live probing is what actually separates real endpoints from
        # noise, not a naming convention.
        api_paths = set(all_paths)
        # If nothing at all was linked from the app shell, fall back to the
        # generic API root prefixes themselves as bootstrap probes — this
        # is a convention-level fallback, not an app-specific endpoint list.
        if not api_paths:
            api_paths = set(API_ROOT_PREFIXES)
        # Bound the candidate set so a large bundle's worth of loosely
        # matched relative-path literals can't turn discovery into an
        # unbounded number of live probes; prefer shorter/api-shaped paths.
        if len(api_paths) > 300:
            api_paths = set(
                sorted(api_paths, key=lambda p: (not p.startswith(API_ROOT_PREFIXES), len(p)))[:300]
            )

        logger.info("Step 4: expanding %d discovered paths with id-variants...", len(api_paths))
        expanded_paths: set[str] = set()
        for p in api_paths:
            expanded_paths.update(self._expand_with_id_variant(p))

        logger.info("Total unique candidate paths to test: %d", len(expanded_paths))

        seen: set[str] = set()
        for path in expanded_paths:
            normalized = self._normalize_path(path)
            probe_target = re.sub(r"\{[^}]+\}", "1", normalized)
            probe_resp = self._probe(probe_target, "GET")
            if not self._is_api_response(probe_resp):
                continue
            methods = self._probe_methods_for_path(normalized) if self.probe_methods else ["GET"]
            for method in methods:
                key = f"{method}:{normalized}"
                if key not in seen:
                    seen.add(key)
                    self.operations.append(self._build_operation(normalized, method, "auto-discovery"))

        logger.info("Discovery complete: %d operations found.", len(self.operations))
        return self.operations
