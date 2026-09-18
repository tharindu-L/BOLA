"""
engine.py — Request Engine Module
Issues controlled HTTP requests for REST and GraphQL operations
through both User A and User B sessions. Captures full request/response
pairs into a structured corpus. Parses GraphQL responses via AST.

Identifier substitution (path, query, body, nested body, GraphQL
variables and nested GraphQL input objects) is driven entirely by the
generic rules attached to each Operation by crawler.py / discovery.py —
no per-application special-casing lives here.

Write operations (POST/PUT/PATCH/DELETE/MUTATION) get an optional
follow-up verification read so the analyzer can distinguish "server
returned 200" from "the object was actually changed" (see analyzer.py).
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from graphql import parse as gql_parse, DocumentNode

from crawler import Operation
from identifiers import find_body_identifiers, substitute_body_path

logger = logging.getLogger("bola.engine")

DEFAULT_TIMEOUT = 20


@dataclass
class ResponseRecord:
    """Full capture of a single HTTP request/response pair."""
    operation_id: str
    user_label: str           # "A" or "B"
    method: str
    url: str
    request_headers: dict
    request_body: Optional[Any]
    status_code: int
    response_headers: dict
    response_body: Any        # Parsed JSON or raw string
    elapsed_ms: float
    error: Optional[str] = None
    gql_ast: Optional[DocumentNode] = None   # Parsed AST for GraphQL queries


@dataclass
class OperationResult:
    """Pair of responses for a single operation — one per user, plus an
    optional post-write verification read (state-change confirmation)."""
    operation: Operation
    response_a: Optional[ResponseRecord] = None
    response_b: Optional[ResponseRecord] = None
    # Read-back of the target object after a write, issued through User B's
    # own (legitimate) session, used to confirm whether A's write actually
    # took effect — i.e. whether this is a *confirmed* vs *potential* BOLA.
    verification_b: Optional[ResponseRecord] = None


class RequestEngine:
    """
    Executes operations through both authenticated sessions.
    For REST: uses requests directly with full path/query/body parameter
    substitution. For GraphQL: substitutes top-level and nested-input
    variables, and parses the query into AST.
    """

    def __init__(
        self,
        base_url: str,
        session_a: requests.Session,
        session_b: requests.Session,
        object_id_a: str,
        object_id_b: str,
        timeout: int = DEFAULT_TIMEOUT,
        verify_writes: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_a = session_a
        self.session_b = session_b
        self.object_id_a = str(object_id_a)
        self.object_id_b = str(object_id_b)
        self.timeout = timeout
        self.verify_writes = verify_writes

    # ------------------------------------------------------------------
    # REST substitution helpers
    # ------------------------------------------------------------------

    def _path_param_names(self, path: str) -> list[str]:
        return re.findall(r"\{([^}]+)\}", path)

    def _substitute_path_params(self, op: Operation, path: str, obj_id: str) -> str:
        """
        Replace {param} placeholders. Placeholders whose name matches an
        identifier rule get the test object id; any other placeholder
        (rare — e.g. a second, unrelated path segment) falls back to the
        same id rather than being left unresolved, since a literal
        unresolved {token} would break routing entirely.
        """
        rule_names = {r["name"] for r in op.object_identifier_rules if r["location"] == "path"}

        def _repl(match: "re.Match") -> str:
            name = match.group(1)
            return obj_id if (not rule_names or name in rule_names) else obj_id

        return re.sub(r"\{([^}]+)\}", _repl, path)

    def _build_rest_url(self, op: Operation, obj_id: str) -> str:
        substituted = self._substitute_path_params(op, op.path, obj_id)
        return f"{self.base_url}{substituted}"

    def _extract_query_params(self, op: Operation, obj_id: str) -> dict:
        params: dict = {}
        for r in op.object_identifier_rules:
            if r["location"] == "query":
                params[r["name"]] = obj_id
        return params

    def _substitute_body(self, op: Operation, obj_id: str) -> Optional[dict]:
        """Substitute every identifier-shaped field found in the sample
        payload (including nested/relationship fields) with the target
        object id, preserving each field's original JSON type."""
        body = op.sample_payload
        if not body:
            return body
        refs = find_body_identifiers(body)
        if not refs:
            return body
        new_body = body
        for ref in refs:
            new_body = substitute_body_path(new_body, ref.path, obj_id)
        return new_body

    def _issue_rest_request(
        self,
        op: Operation,
        session: requests.Session,
        obj_id: str,
        user_label: str,
    ) -> ResponseRecord:
        url = self._build_rest_url(op, obj_id)
        query_params = self._extract_query_params(op, obj_id)
        body = self._substitute_body(op, obj_id)

        try:
            start = time.monotonic()
            resp = session.request(
                method=op.method,
                url=url,
                params=query_params if query_params else None,
                json=body if body else None,
                timeout=self.timeout,
            )
            elapsed = (time.monotonic() - start) * 1000

            try:
                response_body = resp.json()
            except Exception:
                response_body = resp.text

            return ResponseRecord(
                operation_id=op.operation_id,
                user_label=user_label,
                method=op.method,
                url=resp.url,
                request_headers=dict(session.headers),
                request_body=body,
                status_code=resp.status_code,
                response_headers=dict(resp.headers),
                response_body=response_body,
                elapsed_ms=round(elapsed, 2),
            )
        except requests.RequestException as exc:
            logger.warning("REST request failed [%s] %s: %s", user_label, url, exc)
            return ResponseRecord(
                operation_id=op.operation_id,
                user_label=user_label,
                method=op.method,
                url=url,
                request_headers=dict(session.headers),
                request_body=body,
                status_code=0,
                response_headers={},
                response_body=None,
                elapsed_ms=0.0,
                error=str(exc),
            )

    def _issue_verification_read(self, op: Operation, obj_id: str) -> Optional[ResponseRecord]:
        """
        For a write operation on a path-identified object, issue a GET to
        the same resource through User B's own (legitimate) session to see
        whether the object's state actually changed. Only attempted when
        we can safely form a read (same path, GET) and never for DELETE's
        own path (a GET after delete naturally 404s, which is itself
        useful confirmation, not a hazard).
        """
        if not op.path:
            return None
        try:
            url = self._build_rest_url(op, obj_id)
            start = time.monotonic()
            resp = self.session_b.get(url, timeout=self.timeout)
            elapsed = (time.monotonic() - start) * 1000
            try:
                response_body = resp.json()
            except Exception:
                response_body = resp.text
            return ResponseRecord(
                operation_id=op.operation_id,
                user_label="B",
                method="GET",
                url=resp.url,
                request_headers=dict(self.session_b.headers),
                request_body=None,
                status_code=resp.status_code,
                response_headers=dict(resp.headers),
                response_body=response_body,
                elapsed_ms=round(elapsed, 2),
            )
        except requests.RequestException as exc:
            logger.debug("Verification read failed for %s: %s", op.operation_id, exc)
            return None

    # ------------------------------------------------------------------
    # GraphQL substitution helpers
    # ------------------------------------------------------------------

    def _build_graphql_variables_for_user(self, op: Operation, obj_id: str) -> dict:
        variables = dict(op.graphql_variables or {})
        for rule in op.object_identifier_rules:
            if rule.get("location") == "variable":
                variables[rule["name"]] = obj_id
        for nested in op.graphql_nested_identifier_paths:
            var_name = nested["variable"]
            field_name = nested["field"]
            if isinstance(variables.get(var_name), dict):
                variables[var_name] = dict(variables[var_name])
                variables[var_name][field_name] = obj_id
        return variables

    def _issue_graphql_request(
        self,
        op: Operation,
        session: requests.Session,
        obj_id: str,
        user_label: str,
        endpoint: str,
    ) -> ResponseRecord:
        variables = self._build_graphql_variables_for_user(op, obj_id)
        payload = {"query": op.graphql_query_string, "variables": variables}
        gql_ast: Optional[DocumentNode] = None

        try:
            gql_ast = gql_parse(op.graphql_query_string)
        except Exception as exc:
            logger.warning("AST parse failed for op %s: %s", op.operation_id, exc)

        try:
            start = time.monotonic()
            resp = session.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            elapsed = (time.monotonic() - start) * 1000

            try:
                response_body = resp.json()
            except Exception:
                response_body = resp.text

            return ResponseRecord(
                operation_id=op.operation_id,
                user_label=user_label,
                method=op.method,
                url=endpoint,
                request_headers=dict(session.headers),
                request_body=payload,
                status_code=resp.status_code,
                response_headers=dict(resp.headers),
                response_body=response_body,
                elapsed_ms=round(elapsed, 2),
                gql_ast=gql_ast,
            )
        except requests.RequestException as exc:
            logger.warning("GraphQL request failed [%s] %s: %s", user_label, op.query_name, exc)
            return ResponseRecord(
                operation_id=op.operation_id,
                user_label=user_label,
                method=op.method,
                url=endpoint,
                request_headers=dict(session.headers),
                request_body=payload,
                status_code=0,
                response_headers={},
                response_body=None,
                elapsed_ms=0.0,
                error=str(exc),
                gql_ast=gql_ast,
            )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def execute(
        self,
        operations: list[Operation],
        graphql_endpoint: Optional[str] = None,
    ) -> list[OperationResult]:
        """
        Execute all operations through both sessions.
        BOLA oracle input: User A's session requests User B's object id
        (cross-user attempt). User B's own request against the same id is
        the authorized baseline. For write operations we additionally
        issue a same-object verification read through User B afterward so
        the analyzer can tell whether the write actually landed.
        """
        results: list[OperationResult] = []

        for op in operations:
            logger.info("Executing op: %s", op.operation_id)
            result = OperationResult(operation=op)

            if op.api_type == "rest":
                # User A attempts to access/modify User B's object — BOLA test
                result.response_a = self._issue_rest_request(
                    op, self.session_a, self.object_id_b, "A"
                )
                # User B accesses/modifies their own object — baseline
                result.response_b = self._issue_rest_request(
                    op, self.session_b, self.object_id_b, "B"
                )
                if self.verify_writes and op.is_write and op.method != "DELETE":
                    result.verification_b = self._issue_verification_read(op, self.object_id_b)
                elif self.verify_writes and op.method == "DELETE":
                    # A GET on the deleted resource tells us whether it still exists.
                    result.verification_b = self._issue_verification_read(op, self.object_id_b)

            elif op.api_type == "graphql":
                endpoint = graphql_endpoint or self.base_url
                result.response_a = self._issue_graphql_request(
                    op, self.session_a, self.object_id_b, "A", endpoint
                )
                result.response_b = self._issue_graphql_request(
                    op, self.session_b, self.object_id_b, "B", endpoint
                )

            results.append(result)

        logger.info("Request Engine complete: %d operations executed.", len(results))
        return results
