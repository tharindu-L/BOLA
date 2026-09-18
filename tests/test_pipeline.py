"""
End-to-end-ish tests for the BOLA detection pipeline that don't require a
live target: they build Operation/ResponseRecord/OperationResult objects
directly and drive engine substitution + analyzer gating logic.
"""

import json

from crawler import Operation, RESTCrawler
from discovery import EndpointDiscovery
from engine import RequestEngine, ResponseRecord, OperationResult
from analyzer import BOLAAnalyzer, VERDICT_CONFIRMED, VERDICT_POTENTIAL


def _rec(**kwargs) -> ResponseRecord:
    defaults = dict(
        request_headers={}, request_body=None, response_headers={}, elapsed_ms=0.0,
    )
    defaults.update(kwargs)
    return ResponseRecord(**defaults)


# ---------------------------------------------------------------------------
# Crawler: path-parameter identifier detection (username/opaque, not just "id")
# ---------------------------------------------------------------------------

def _write_spec(tmp_path, spec: dict) -> str:
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return str(p)


def test_rest_crawler_detects_non_id_named_path_param(tmp_path):
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "t", "version": "1"},
        "paths": {
            "/users/v1/{username}": {
                "get": {
                    "operationId": "getUser",
                    "parameters": [
                        {"name": "username", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                }
            }
        },
    }
    spec_path = _write_spec(tmp_path, spec)
    ops = RESTCrawler(spec_path).crawl()
    assert len(ops) == 1
    op = ops[0]
    assert op.object_identifier_extraction_rule == {"location": "path", "name": "username"}


def test_rest_crawler_finds_nested_body_identifier(tmp_path):
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "t", "version": "1"},
        "paths": {
            "/orders": {
                "post": {
                    "operationId": "createOrder",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "data": {
                                            "type": "object",
                                            "properties": {
                                                "object_id": {"type": "integer"},
                                            },
                                        }
                                    },
                                }
                            }
                        }
                    },
                }
            }
        },
    }
    spec_path = _write_spec(tmp_path, spec)
    ops = RESTCrawler(spec_path).crawl()
    op = ops[0]
    body_rules = [r for r in op.object_identifier_rules if r["location"] == "body"]
    assert any(r["name"] == "object_id" for r in body_rules)


# ---------------------------------------------------------------------------
# Engine: path/body substitution
# ---------------------------------------------------------------------------

def _fake_engine():
    class FakeSession:
        headers = {}
    return RequestEngine(
        base_url="http://target",
        session_a=FakeSession(),
        session_b=FakeSession(),
        object_id_a="1",
        object_id_b="2",
    )


def test_engine_substitutes_nested_body_identifier():
    engine = _fake_engine()
    op = Operation(
        operation_id="PUT:/orders",
        api_type="rest",
        method="PUT",
        path="/orders",
        sample_payload={"data": {"object": {"id": 111}}},
        object_identifier_rules=[{"location": "body", "name": "id", "path": ("data", "object", "id")}],
    )
    substituted = engine._substitute_body(op, "999")
    assert substituted["data"]["object"]["id"] == 999


def test_engine_substitutes_all_path_placeholders():
    engine = _fake_engine()
    op = Operation(
        operation_id="GET:/x/{id}",
        api_type="rest",
        method="GET",
        path="/x/{id}",
        object_identifier_rules=[{"location": "path", "name": "id"}],
    )
    url = engine._build_rest_url(op, "42")
    assert url == "http://target/x/42"


# ---------------------------------------------------------------------------
# Analyzer: GraphQL gating — parameterless/public list queries must NOT be
# flagged just because both users get an identical response.
# ---------------------------------------------------------------------------

def _gql_result(query_name, id_rule, body_a, body_b):
    op = Operation(
        operation_id=f"QUERY:{query_name}",
        api_type="graphql",
        method="QUERY",
        query_name=query_name,
        graphql_query_string=f"query {{ {query_name} {{ id }} }}",
        object_identifier_rules=[id_rule] if id_rule else [],
    )
    rec_a = _rec(
        operation_id=op.operation_id, user_label="A", method="QUERY", url="http://x",
        status_code=200, response_body={"data": {query_name: body_a}},
    )
    rec_b = _rec(
        operation_id=op.operation_id, user_label="B", method="QUERY", url="http://x",
        status_code=200, response_body={"data": {query_name: body_b}},
    )
    return OperationResult(operation=op, response_a=rec_a, response_b=rec_b)


def test_graphql_public_list_without_id_arg_not_flagged():
    public_list = [{"id": 1, "title": "hello"}, {"id": 2, "title": "world"}]
    result = _gql_result("posts", id_rule=None, body_a=public_list, body_b=public_list)
    findings = BOLAAnalyzer().analyze([result])
    assert findings == []


def test_graphql_object_with_id_arg_and_ownership_signal_is_flagged():
    leaked = {"id": 5, "content": "secret note", "owner": {"email": "victim@test.com"}}
    result = _gql_result(
        "paste",
        id_rule={"location": "variable", "name": "id"},
        body_a=leaked,
        body_b=leaked,
    )
    findings = BOLAAnalyzer().analyze([result])
    assert len(findings) == 1
    assert findings[0].verdict in (VERDICT_CONFIRMED, VERDICT_POTENTIAL)


# ---------------------------------------------------------------------------
# Analyzer: write-operation verification (PUT/DELETE confirmed vs not)
# ---------------------------------------------------------------------------

def _rest_write_result(method, verification_status, verification_body):
    op = Operation(
        operation_id=f"{method}:/api/user/{{id}}",
        api_type="rest",
        method=method,
        path="/api/user/{id}",
        sample_payload={"email": "attacker@test.com"} if method != "DELETE" else None,
        object_identifier_rules=[{"location": "path", "name": "id"}],
        is_write=True,
    )
    rec_a = _rec(
        operation_id=op.operation_id, user_label="A", method=method, url="http://x/api/user/2",
        request_body=op.sample_payload, status_code=200, response_body={"status": "ok"},
    )
    rec_b = _rec(
        operation_id=op.operation_id, user_label="B", method=method, url="http://x/api/user/2",
        request_body=op.sample_payload, status_code=200, response_body={"status": "ok"},
    )
    verification = None
    if verification_status is not None:
        verification = _rec(
            operation_id=op.operation_id, user_label="B", method="GET", url="http://x/api/user/2",
            status_code=verification_status, response_body=verification_body,
        )
    return OperationResult(operation=op, response_a=rec_a, response_b=rec_b, verification_b=verification)


def test_delete_confirmed_when_object_gone():
    result = _rest_write_result("DELETE", verification_status=404, verification_body=None)
    findings = BOLAAnalyzer().analyze([result])
    assert len(findings) == 1
    assert findings[0].verdict == VERDICT_CONFIRMED


def test_delete_not_flagged_when_object_still_present():
    result = _rest_write_result("DELETE", verification_status=200, verification_body={"id": 2})
    findings = BOLAAnalyzer().analyze([result])
    assert findings == []


def test_put_confirmed_when_verification_reflects_injected_value():
    result = _rest_write_result(
        "PUT",
        verification_status=200,
        verification_body={"id": 2, "email": "attacker@test.com"},
    )
    findings = BOLAAnalyzer().analyze([result])
    assert len(findings) == 1
    assert findings[0].verdict == VERDICT_CONFIRMED


def test_put_not_flagged_when_verification_shows_no_change():
    result = _rest_write_result(
        "PUT",
        verification_status=200,
        verification_body={"id": 2, "email": "original-owner@test.com"},
    )
    findings = BOLAAnalyzer().analyze([result])
    assert findings == []


def test_put_potential_when_verification_unavailable():
    result = _rest_write_result("PUT", verification_status=None, verification_body=None)
    findings = BOLAAnalyzer().analyze([result])
    assert len(findings) == 1
    assert findings[0].verdict == VERDICT_POTENTIAL


# ---------------------------------------------------------------------------
# Analyzer: secret redaction in evidence
# ---------------------------------------------------------------------------

def test_evidence_redacts_authorization_header():
    op = Operation(
        operation_id="GET:/api/Users/{id}",
        api_type="rest",
        method="GET",
        path="/api/Users/{id}",
        object_identifier_rules=[{"location": "path", "name": "id"}],
    )
    shared_body = {"id": 2, "email": "victim@test.com", "username": "victim"}
    rec_a = _rec(
        operation_id=op.operation_id, user_label="A", method="GET", url="http://x/api/Users/2",
        request_headers={"Authorization": "Bearer super-secret-jwt"},
        status_code=200, response_body=shared_body,
    )
    rec_b = _rec(
        operation_id=op.operation_id, user_label="B", method="GET", url="http://x/api/Users/2",
        request_headers={"Authorization": "Bearer other-secret-jwt"},
        status_code=200, response_body=shared_body,
    )
    result = OperationResult(operation=op, response_a=rec_a, response_b=rec_b)
    findings = BOLAAnalyzer().analyze([result])
    assert len(findings) == 1
    assert findings[0].evidence["request_a"]["headers"]["Authorization"] == "***REDACTED***"
    assert "super-secret-jwt" not in json.dumps(findings[0].evidence)


# ---------------------------------------------------------------------------
# Discovery: JS bundle route extraction must find *relative* path literals
# (no leading slash), which is how SPA frameworks like Angular typically
# build request URLs by concatenating a base-host constant with a bare
# relative string (e.g. `environment.hostServer + "rest/user/login"`).
# Regression test for a real scan against Juice Shop finding 0 routes.
# ---------------------------------------------------------------------------

def test_js_route_extraction_finds_relative_literals_in_call_context():
    discoverer = EndpointDiscovery(base_url="http://target")
    js_content = (
        'login(t){return this.http.post(environment.hostServer+"rest/user/login",t)}'
        'getUser(id){return this.http.get("api/Users/"+id)}'
    )
    routes = discoverer._extract_routes_from_js(js_content)
    assert "/rest/user/login" in routes
    assert "/api/Users/" in routes


def test_js_route_extraction_ignores_static_assets():
    discoverer = EndpointDiscovery(base_url="http://target")
    js_content = 'fetch("assets/public/images/logo.png")'
    routes = discoverer._extract_routes_from_js(js_content)
    assert not any(r.endswith(".png") for r in routes)
