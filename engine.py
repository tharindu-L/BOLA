"""
engine.py — Request Engine Module
Issues controlled HTTP requests for REST and GraphQL operations
through both User A and User B sessions. Captures full request/response
pairs into a structured corpus. Parses GraphQL responses via AST.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from graphql import parse as gql_parse, DocumentNode

from crawler import Operation

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
    """Pair of responses for a single operation — one per user."""
    operation: Operation
    response_a: Optional[ResponseRecord] = None
    response_b: Optional[ResponseRecord] = None


class RequestEngine:
    """
    Executes operations through both authenticated sessions.
    For REST: uses requests directly with path parameter substitution.
    For GraphQL: uses requests with JSON body; parses query into AST.
    """

    def __init__(
        self,
        base_url: str,
        session_a: requests.Session,
        session_b: requests.Session,
        object_id_a: str,
        object_id_b: str,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_a = session_a
        self.session_b = session_b
        self.object_id_a = str(object_id_a)
        self.object_id_b = str(object_id_b)
        self.timeout = timeout

    def _substitute_path_params(self, path: str, obj_id: str) -> str:
        """Replace {param} placeholders in path with the given object ID."""
        import re
        return re.sub(r"\{[^}]+\}", obj_id, path)

    def _build_rest_url(self, path: str, obj_id: str) -> str:
        substituted = self._substitute_path_params(path, obj_id)
        return f"{self.base_url}{substituted}"

    def _extract_query_params(self, op: Operation, obj_id: str) -> dict:
        params: dict = {}
        for p in op.parameters:
            if p.get("in") == "query":
                from crawler import RESTCrawler
                if any(pat in p.get("name", "").lower() for pat in RESTCrawler.OBJECT_ID_PATTERNS):
                    params[p["name"]] = obj_id
        return params

    def _issue_rest_request(
        self,
        op: Operation,
        session: requests.Session,
        obj_id: str,
        user_label: str,
    ) -> ResponseRecord:
        url = self._build_rest_url(op.path, obj_id)
        query_params = self._extract_query_params(op, obj_id)
        body = op.sample_payload

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

    def _build_graphql_variables_for_user(self, op: Operation, obj_id: str) -> dict:
        variables = dict(op.graphql_variables or {})
        rule = op.object_identifier_extraction_rule
        if rule and rule.get("location") == "variable":
            variables[rule["name"]] = obj_id
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

    def execute(
        self,
        operations: list[Operation],
        graphql_endpoint: Optional[str] = None,
    ) -> list[OperationResult]:
        """
        Execute all operations through both sessions.
        BOLA oracle logic: User B's object ID is sent through User A's session.
        User B's own request is the baseline.
        """
        results: list[OperationResult] = []

        for op in operations:
            logger.info("Executing op: %s", op.operation_id)
            result = OperationResult(operation=op)

            if op.api_type == "rest":
                # User A attempts to access User B's object — BOLA test
                result.response_a = self._issue_rest_request(
                    op, self.session_a, self.object_id_b, "A"
                )
                # User B accesses their own object — baseline
                result.response_b = self._issue_rest_request(
                    op, self.session_b, self.object_id_b, "B"
                )

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