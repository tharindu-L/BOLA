"""
analyzer.py — BOLA Analyzer Module
Core oracle: deterministic cross-session response comparison.
REST: structural JSON similarity with configurable threshold.
GraphQL: field-level AST-guided diffing on nested responses.
Produces evidence transcripts for every finding.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from graphql import DocumentNode
from graphql.language import ast as gql_ast_types

from engine import OperationResult, ResponseRecord

logger = logging.getLogger("bola.analyzer")

SIMILARITY_THRESHOLD = 0.85

IGNORED_FIELDS = frozenset({
    "timestamp", "created_at", "updated_at", "date", "time",
    "expires_at", "last_login", "modified_at", "request_id",
})


@dataclass
class Finding:
    """A confirmed BOLA vulnerability finding with full evidence transcript."""
    operation_id: str
    severity: str             # "High" | "Medium"
    api_type: str
    method: str
    path_or_query: str
    evidence: dict            # Full transcript: requests, responses, diff
    reproduction_steps: list[str]
    similarity_score: Optional[float] = None
    field_leakage: list[str] = field(default_factory=list)


def _flatten_json(obj: Any, prefix: str = "") -> dict:
    """Recursively flatten a JSON structure into dot-notation keys."""
    items: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if k.lower() in IGNORED_FIELDS:
                continue
            items.update(_flatten_json(v, full_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(_flatten_json(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = obj
    return items


def _json_similarity(body_a: Any, body_b: Any) -> float:
    """
    Structural similarity between two JSON responses.
    Returns ratio of matching key-value pairs to total unique keys.
    Ignores timestamp/metadata fields.
    """
    if body_a is None or body_b is None:
        return 0.0
    flat_a = _flatten_json(body_a)
    flat_b = _flatten_json(body_b)
    all_keys = set(flat_a.keys()) | set(flat_b.keys())
    if not all_keys:
        return 1.0
    matching = sum(
        1 for k in all_keys
        if flat_a.get(k) == flat_b.get(k) and k in flat_a and k in flat_b
    )
    return matching / len(all_keys)


def _diff_flat(body_a: Any, body_b: Any) -> dict:
    """Return a dict of keys that differ between the two responses."""
    flat_a = _flatten_json(body_a)
    flat_b = _flatten_json(body_b)
    all_keys = set(flat_a.keys()) | set(flat_b.keys())
    diff: dict = {}
    for k in all_keys:
        va = flat_a.get(k, "<missing>")
        vb = flat_b.get(k, "<missing>")
        if va != vb:
            diff[k] = {"user_a": va, "user_b": vb}
    return diff


def _extract_gql_selected_fields(ast_node: Optional[DocumentNode]) -> set[str]:
    """
    Walk the GraphQL AST and collect all leaf field names in the selection set.
    Used for field-level comparison in nested queries.
    """
    fields: set[str] = set()
    if ast_node is None:
        return fields

    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, gql_ast_types.FieldNode):
            full = f"{prefix}.{node.name.value}" if prefix else node.name.value
            if node.selection_set:
                for child in node.selection_set.selections:
                    walk(child, full)
            else:
                fields.add(full)
        elif isinstance(node, gql_ast_types.OperationDefinitionNode):
            if node.selection_set:
                for child in node.selection_set.selections:
                    walk(child, prefix)

    for definition in ast_node.definitions:
        walk(definition)
    return fields


def _extract_gql_data(response_body: Any, query_name: Optional[str]) -> Any:
    """Extract the data payload from a GraphQL response."""
    if not isinstance(response_body, dict):
        return response_body
    data = response_body.get("data", {})
    if query_name and isinstance(data, dict):
        return data.get(query_name, data)
    return data


def _has_gql_errors(response_body: Any) -> bool:
    if not isinstance(response_body, dict):
        return False
    errors = response_body.get("errors")
    return bool(errors)


class BOLAAnalyzer:
    """
    Core BOLA detection oracle.
    For each OperationResult, determines whether User A received
    data belonging to User B without authorization.
    """

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = similarity_threshold
        self.findings: list[Finding] = []

    def _build_reproduction_steps(
        self,
        result: OperationResult,
        rec_a: ResponseRecord,
        rec_b: ResponseRecord,
    ) -> list[str]:
        op = result.operation
        steps = [
            f"1. Authenticate as User B and obtain their object identifier.",
            f"2. Authenticate as User A (attacker session).",
        ]
        if op.api_type == "rest":
            steps.append(
                f"3. Issue {rec_a.method} request to: {rec_a.url}"
            )
            steps.append(
                "4. Include User A's authentication credentials in the Authorization header."
            )
            steps.append(
                f"5. Observe HTTP {rec_a.status_code} response — data belonging to User B is returned."
            )
        else:
            payload = rec_a.request_body or {}
            steps.append(
                f"3. Send GraphQL {op.method} '{op.query_name}' to {rec_a.url}"
            )
            steps.append(
                f"4. Use payload: {json.dumps(payload, indent=2)}"
            )
            steps.append(
                f"5. Observe HTTP {rec_a.status_code} — GraphQL data node '{op.query_name}' "
                f"returns User B's object fields."
            )
        steps.append("6. Compare response to User B's own authenticated response — data is substantially identical.")
        steps.append("7. Confirm BOLA: User A accessed User B's resource without authorization.")
        return steps

    def _analyze_rest(self, result: OperationResult) -> Optional[Finding]:
        rec_a = result.response_a
        rec_b = result.response_b
        op = result.operation

        if rec_a is None or rec_b is None:
            return None
        if rec_a.error or rec_a.status_code == 0:
            return None
        if rec_a.status_code != 200:
            logger.debug("REST op %s: User A got %d, skipping.", op.operation_id, rec_a.status_code)
            return None
        if rec_b.status_code != 200:
            return None

        score = _json_similarity(rec_a.response_body, rec_b.response_body)
        logger.debug("REST similarity for %s: %.2f", op.operation_id, score)

        if score < self.threshold:
            return None

        diff = _diff_flat(rec_a.response_body, rec_b.response_body)
        severity = "High" if score >= 0.95 else "Medium"

        finding = Finding(
            operation_id=op.operation_id,
            severity=severity,
            api_type="rest",
            method=op.method,
            path_or_query=op.path or "",
            similarity_score=round(score, 4),
            evidence={
                "request_a": {
                    "method": rec_a.method,
                    "url": rec_a.url,
                    "headers": rec_a.request_headers,
                    "body": rec_a.request_body,
                },
                "response_a": {
                    "status_code": rec_a.status_code,
                    "body": rec_a.response_body,
                    "elapsed_ms": rec_a.elapsed_ms,
                },
                "request_b": {
                    "method": rec_b.method,
                    "url": rec_b.url,
                    "headers": rec_b.request_headers,
                    "body": rec_b.request_body,
                },
                "response_b": {
                    "status_code": rec_b.status_code,
                    "body": rec_b.response_body,
                    "elapsed_ms": rec_b.elapsed_ms,
                },
                "diff": diff,
            },
            reproduction_steps=self._build_reproduction_steps(result, rec_a, rec_b),
        )
        return finding

    def _analyze_graphql(self, result: OperationResult) -> Optional[Finding]:
        rec_a = result.response_a
        rec_b = result.response_b
        op = result.operation

        if rec_a is None or rec_b is None:
            return None
        if rec_a.error or rec_a.status_code == 0:
            return None
        if rec_a.status_code != 200:
            return None
        if _has_gql_errors(rec_a.response_body):
            logger.debug("GraphQL op %s: User A response has errors, skipping.", op.operation_id)
            return None
        if _has_gql_errors(rec_b.response_body):
            return None

        data_a = _extract_gql_data(rec_a.response_body, op.query_name)
        data_b = _extract_gql_data(rec_b.response_body, op.query_name)

        if data_a is None or data_b is None:
            return None

        # Field-level diffing guided by AST
        selected_fields = _extract_gql_selected_fields(rec_a.gql_ast)
        flat_a = _flatten_json(data_a)
        flat_b = _flatten_json(data_b)

        leaked_fields: list[str] = []
        for fld in selected_fields:
            # Check if the field key exists in both (with any prefix path)
            matches_a = {k: v for k, v in flat_a.items() if k.endswith(fld) or fld in k}
            matches_b = {k: v for k, v in flat_b.items() if k.endswith(fld) or fld in k}
            for key in set(matches_a) & set(matches_b):
                if matches_a[key] == matches_b[key] and matches_b[key] not in (None, "", [], {}):
                    leaked_fields.append(key)

        score = _json_similarity(data_a, data_b)

        if score < self.threshold and not leaked_fields:
            return None

        diff = _diff_flat(data_a, data_b)
        severity = "High" if leaked_fields or score >= 0.95 else "Medium"

        finding = Finding(
            operation_id=op.operation_id,
            severity=severity,
            api_type="graphql",
            method=op.method,
            path_or_query=op.query_name or "",
            similarity_score=round(score, 4),
            field_leakage=leaked_fields,
            evidence={
                "query_name": op.query_name,
                "query_string": op.graphql_query_string,
                "request_a": {
                    "url": rec_a.url,
                    "body": rec_a.request_body,
                },
                "response_a": {
                    "status_code": rec_a.status_code,
                    "data": data_a,
                    "elapsed_ms": rec_a.elapsed_ms,
                },
                "request_b": {
                    "url": rec_b.url,
                    "body": rec_b.request_body,
                },
                "response_b": {
                    "status_code": rec_b.status_code,
                    "data": data_b,
                    "elapsed_ms": rec_b.elapsed_ms,
                },
                "diff": diff,
                "leaked_fields": leaked_fields,
                "ast_selected_fields": list(selected_fields),
            },
            reproduction_steps=self._build_reproduction_steps(result, rec_a, rec_b),
        )
        return finding

    def analyze(self, results: list[OperationResult]) -> list[Finding]:
        self.findings = []
        for result in results:
            op = result.operation
            try:
                if op.api_type == "rest":
                    finding = self._analyze_rest(result)
                elif op.api_type == "graphql":
                    finding = self._analyze_graphql(result)
                else:
                    logger.warning("Unknown api_type: %s", op.api_type)
                    finding = None

                if finding:
                    self.findings.append(finding)
                    logger.info(
                        "[%s] BOLA CONFIRMED: %s %s (similarity=%.2f)",
                        finding.severity,
                        finding.method,
                        finding.path_or_query,
                        finding.similarity_score or 0.0,
                    )
            except Exception as exc:
                logger.error("Analysis error on op %s: %s", op.operation_id, exc, exc_info=True)

        logger.info("Analysis complete: %d finding(s).", len(self.findings))
        return self.findings