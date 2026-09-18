"""
analyzer.py — BOLA Analyzer Module
Core oracle: deterministic cross-session response comparison.
REST: structural JSON similarity with configurable threshold.
GraphQL: field-level AST-guided diffing on nested responses.
Write operations: state-change verification via a follow-up read.
Produces evidence transcripts for every finding, with secrets redacted.

REST and GraphQL go through the same gating pipeline (identifier-rule
requirement + ownership-signal requirement) so GraphQL no longer flags
every list/enumeration query as a false positive, and every finding is
classified Confirmed vs Potential based on the strength of the
available evidence rather than a single HTTP-200 signal.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from graphql import DocumentNode
from graphql.language import ast as gql_ast_types

from engine import OperationResult, ResponseRecord
from identifiers import (
    name_matches_ownership,
    redact_headers,
    redact_body,
)

logger = logging.getLogger("bola.analyzer")

SIMILARITY_THRESHOLD = 0.85

IGNORED_FIELDS = frozenset({
    "timestamp", "created_at", "updated_at", "date", "time",
    "expires_at", "last_login", "modified_at", "request_id",
})

# Generic PII / account-data field-name fragments used as an ownership
# signal: a response is only flagged as leaking *another user's* data if
# it actually contains fields shaped like personal/account data, which
# rules out endpoints that legitimately return the same public/shared
# payload to everyone. These are broad category terms (email, phone,
# card number, role, ...), not literal field names from any one app.
OWNERSHIP_SIGNAL_FRAGMENTS = (
    "email", "username", "password", "phone", "address",
    "customer", "order", "basket", "cart",
    "card", "cvv", "iban", "account",
    "dob", "dateofbirth", "ssn", "passport",
    "role", "token", "secret",
)

VERDICT_CONFIRMED = "Confirmed"
VERDICT_POTENTIAL = "Potential"


@dataclass
class Finding:
    operation_id: str
    severity: str
    api_type: str
    method: str
    path_or_query: str
    evidence: dict
    reproduction_steps: list[str]
    verdict: str = VERDICT_POTENTIAL
    similarity_score: Optional[float] = None
    field_leakage: list[str] = field(default_factory=list)


def _flatten_json(obj: Any, prefix: str = "") -> dict:
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


def _redact_request(rec: ResponseRecord) -> dict:
    return {
        "method": rec.method,
        "url": rec.url,
        "headers": redact_headers(rec.request_headers),
        "body": redact_body(rec.request_body),
    }


def _redact_response(rec: ResponseRecord) -> dict:
    return {
        "status_code": rec.status_code,
        "body": redact_body(rec.response_body),
        "elapsed_ms": rec.elapsed_ms,
    }


class BOLAAnalyzer:
    """
    Core BOLA detection oracle.
    For each OperationResult, determines whether User A received (read
    ops) or actually altered (write ops) an object belonging to User B
    without authorization, and classifies the finding as Confirmed
    (strong evidence) or Potential (suspicious but unverified).
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
            "1. Authenticate as User B and obtain their object identifier.",
            "2. Authenticate as User A (attacker session).",
        ]
        if op.api_type == "rest":
            steps.append(f"3. Issue {rec_a.method} request to: {rec_a.url}")
            steps.append("4. Include User A's authentication credentials in the Authorization header.")
            steps.append(f"5. Observe HTTP {rec_a.status_code} response — data belonging to User B is returned.")
        else:
            steps.append(f"3. Send GraphQL {op.method} '{op.query_name}' to {rec_a.url}")
            steps.append("4. Use User A's authentication credentials with User B's object identifier as a variable.")
            steps.append(
                f"5. Observe HTTP {rec_a.status_code} — GraphQL data node '{op.query_name}' "
                f"returns User B's object fields."
            )
        steps.append("6. Compare response to User B's own authenticated response — data is substantially identical.")
        steps.append("7. Confirm BOLA: User A accessed/altered User B's resource without authorization.")
        return steps

    def _has_ownership_signals(self, body: Any) -> bool:
        flat = _flatten_json(body)
        keys_lower = {k.lower().split(".")[-1] for k in flat.keys()}
        if keys_lower & set(OWNERSHIP_SIGNAL_FRAGMENTS):
            return True
        # Structural fallback: any key that fragment-matches an ownership
        # pattern (owner/user/account/...) also counts as a signal.
        return any(name_matches_ownership(k) for k in keys_lower)

    # ------------------------------------------------------------------
    # Read-operation oracle (GET-shaped: does A receive B's data?)
    # ------------------------------------------------------------------

    def _analyze_rest_read(self, result: OperationResult) -> Optional[Finding]:
        rec_a = result.response_a
        rec_b = result.response_b
        op = result.operation

        if rec_a is None or rec_b is None:
            return None
        if rec_a.error or rec_a.status_code == 0:
            return None
        if rec_a.status_code != 200 or rec_b.status_code != 200:
            logger.debug("REST op %s: non-200 response, skipping.", op.operation_id)
            return None
        if not op.object_identifier_rules:
            logger.debug("Skipping %s — no object identifier.", op.operation_id)
            return None

        score = _json_similarity(rec_a.response_body, rec_b.response_body)
        logger.debug("REST similarity for %s: %.2f", op.operation_id, score)

        if score < self.threshold:
            return None
        if not self._has_ownership_signals(rec_b.response_body):
            logger.debug(
                "Skipping %s — response has no user-ownership signals, likely public.",
                op.operation_id,
            )
            return None

        diff = _diff_flat(rec_a.response_body, rec_b.response_body)
        verdict = VERDICT_CONFIRMED if score >= 0.95 else VERDICT_POTENTIAL
        severity = "High" if verdict == VERDICT_CONFIRMED else "Medium"

        return Finding(
            operation_id=op.operation_id,
            severity=severity,
            verdict=verdict,
            api_type="rest",
            method=op.method,
            path_or_query=op.path or "",
            similarity_score=round(score, 4),
            evidence={
                "request_a": _redact_request(rec_a),
                "response_a": _redact_response(rec_a),
                "request_b": _redact_request(rec_b),
                "response_b": _redact_response(rec_b),
                "diff": diff,
            },
            reproduction_steps=self._build_reproduction_steps(result, rec_a, rec_b),
        )

    # ------------------------------------------------------------------
    # Write-operation oracle (PUT/PATCH/POST/DELETE: did A's request
    # actually change B's object, not merely receive HTTP 200?)
    # ------------------------------------------------------------------

    def _write_injected_values(self, op) -> set[str]:
        """String-ified values from the write payload A submitted (every
        leaf field, not only identifier-shaped ones — the attacker's
        write can alter *any* field of B's object), used to check whether
        the verification read reflects A's write actually taking effect."""
        values: set[str] = set()
        if op.sample_payload:
            for v in _flatten_json(op.sample_payload).values():
                values.add(str(v))
        return values

    def _analyze_rest_write(self, result: OperationResult) -> Optional[Finding]:
        rec_a = result.response_a
        rec_b = result.response_b
        op = result.operation

        if rec_a is None or rec_b is None:
            return None
        if rec_a.error or rec_a.status_code == 0:
            return None
        # Any 2xx (200/201/202/204) is treated as "server accepted the request";
        # by itself this is only Potential evidence — actual confirmation
        # requires the verification read.
        if not (200 <= rec_a.status_code < 300):
            return None
        if not op.object_identifier_rules:
            return None

        verification = result.verification_b
        verdict = VERDICT_POTENTIAL
        confirmation_note = "Server returned success status; state change not independently verified."

        if op.method == "DELETE":
            if verification is not None and verification.status_code in (404, 410):
                verdict = VERDICT_CONFIRMED
                confirmation_note = (
                    f"Verification read after User A's DELETE returned HTTP "
                    f"{verification.status_code} — User B's object no longer exists."
                )
            elif verification is not None and verification.status_code == 200:
                # Object still readable and unchanged — the delete had no
                # effect, so this is not exploitable; do not report.
                logger.debug("DELETE op %s: object still present after delete, not BOLA.", op.operation_id)
                return None
        else:
            # PUT/PATCH/POST state change
            if verification is not None and verification.status_code == 200:
                flat_v = _flatten_json(verification.response_body)
                v_values = {str(v) for v in flat_v.values()}
                if v_values & self._write_injected_values(op):
                    verdict = VERDICT_CONFIRMED
                    confirmation_note = (
                        "Verification read after User A's write reflects the values "
                        "User A submitted — User B's object was actually modified."
                    )
                else:
                    # Verified the object exists but content doesn't match
                    # what A tried to write — could not confirm the change.
                    return None

        severity = "High" if verdict == VERDICT_CONFIRMED else "Medium"
        evidence = {
            "request_a": _redact_request(rec_a),
            "response_a": _redact_response(rec_a),
            "request_b": _redact_request(rec_b),
            "response_b": _redact_response(rec_b),
            "confirmation": confirmation_note,
        }
        if verification is not None:
            evidence["verification_read"] = _redact_response(verification)

        steps = [
            "1. Authenticate as User B and obtain their object identifier.",
            "2. Authenticate as User A (attacker session).",
            f"3. Issue {op.method} request to: {rec_a.url} using User A's credentials "
            f"and User B's object identifier.",
            f"4. Observe HTTP {rec_a.status_code} accepted response.",
            "5. Re-read the object through User B's own session to verify the change.",
            f"6. {confirmation_note}",
        ]

        return Finding(
            operation_id=op.operation_id,
            severity=severity,
            verdict=verdict,
            api_type="rest",
            method=op.method,
            path_or_query=op.path or "",
            similarity_score=None,
            evidence=evidence,
            reproduction_steps=steps,
        )

    def _analyze_rest(self, result: OperationResult) -> Optional[Finding]:
        if result.operation.is_write:
            return self._analyze_rest_write(result)
        return self._analyze_rest_read(result)

    # ------------------------------------------------------------------
    # GraphQL oracle — same gating discipline as REST (identifier rule
    # required + ownership-signal required) so parameterless/public list
    # queries are not automatically flagged.
    # ------------------------------------------------------------------

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
        if _has_gql_errors(rec_a.response_body) or _has_gql_errors(rec_b.response_body):
            return None

        # A GraphQL operation is only testable as a cross-user BOLA case if
        # it actually accepts an object identifier (arg or nested input
        # field) — otherwise User A and User B are, by construction,
        # issuing the identical request and any similarity is meaningless.
        has_identifier = bool(op.object_identifier_rules or op.graphql_nested_identifier_paths)
        if not has_identifier:
            logger.debug("Skipping %s — no identifier argument, not a cross-user test.", op.operation_id)
            return None

        data_a = _extract_gql_data(rec_a.response_body, op.query_name)
        data_b = _extract_gql_data(rec_b.response_body, op.query_name)
        if data_a is None or data_b is None:
            return None

        if not self._has_ownership_signals(data_b):
            logger.debug("Skipping %s — no ownership/PII signals, likely public data.", op.operation_id)
            return None

        selected_fields = _extract_gql_selected_fields(rec_a.gql_ast)
        flat_a = _flatten_json(data_a)
        flat_b = _flatten_json(data_b)

        leaked_fields: list[str] = []
        for fld in selected_fields:
            matches_a = {k: v for k, v in flat_a.items() if k.endswith(fld) or fld in k}
            matches_b = {k: v for k, v in flat_b.items() if k.endswith(fld) or fld in k}
            for key in set(matches_a) & set(matches_b):
                if matches_a[key] == matches_b[key] and matches_b[key] not in (None, "", [], {}):
                    leaked_fields.append(key)

        score = _json_similarity(data_a, data_b)
        if score < self.threshold and not leaked_fields:
            return None

        diff = _diff_flat(data_a, data_b)
        verdict = VERDICT_CONFIRMED if (leaked_fields or score >= 0.95) else VERDICT_POTENTIAL
        severity = "High" if verdict == VERDICT_CONFIRMED else "Medium"

        return Finding(
            operation_id=op.operation_id,
            severity=severity,
            verdict=verdict,
            api_type="graphql",
            method=op.method,
            path_or_query=op.query_name or "",
            similarity_score=round(score, 4),
            field_leakage=leaked_fields,
            evidence={
                "query_name": op.query_name,
                "query_string": op.graphql_query_string,
                "request_a": {"url": rec_a.url, "body": redact_body(rec_a.request_body)},
                "response_a": {"status_code": rec_a.status_code, "data": redact_body(data_a), "elapsed_ms": rec_a.elapsed_ms},
                "request_b": {"url": rec_b.url, "body": redact_body(rec_b.request_body)},
                "response_b": {"status_code": rec_b.status_code, "data": redact_body(data_b), "elapsed_ms": rec_b.elapsed_ms},
                "diff": diff,
                "leaked_fields": leaked_fields,
                "ast_selected_fields": list(selected_fields),
            },
            reproduction_steps=self._build_reproduction_steps(result, rec_a, rec_b),
        )

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
                        "[%s/%s] BOLA: %s %s",
                        finding.verdict,
                        finding.severity,
                        finding.method,
                        finding.path_or_query,
                    )
            except Exception as exc:
                logger.error("Analysis error on op %s: %s", op.operation_id, exc, exc_info=True)

        logger.info("Analysis complete: %d finding(s).", len(self.findings))
        return self.findings
