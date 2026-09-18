"""Unit tests for the generic identifier/ownership engine (identifiers.py)."""

from identifiers import (
    name_matches_identifier,
    name_matches_ownership,
    value_looks_like_identifier,
    find_body_identifiers,
    substitute_body_path,
    extract_ownership_signals,
    decode_jwt_payload,
    extract_self_id_from_claims,
    redact_headers,
    redact_body,
)


def test_name_matches_identifier_generic_shapes():
    assert name_matches_identifier("id")
    assert name_matches_identifier("userId")
    assert name_matches_identifier("vehicleId")
    assert name_matches_identifier("username")
    assert name_matches_identifier("order_id")
    assert name_matches_identifier("uuid")
    # Names with no identifier-shaped fragment at all should not match.
    assert name_matches_identifier("description") is False


def test_value_looks_like_identifier():
    assert value_looks_like_identifier(123)
    assert value_looks_like_identifier("123")
    assert value_looks_like_identifier("550e8400-e29b-41d4-a716-446655440000")
    assert value_looks_like_identifier("john_doe99")
    assert value_looks_like_identifier(True) is False
    assert value_looks_like_identifier("a very long free text description " * 5) is False
    # JWTs are not object identifiers even though they're opaque strings
    assert value_looks_like_identifier("aaaa.bbbb.cccc") is False


def test_find_body_identifiers_nested():
    body = {
        "title": "hello",
        "object_id": 123,
        "data": {
            "object": {
                "id": 456,
                "owner_id": 10,
            },
            "notes": "not an id",
        },
    }
    refs = find_body_identifiers(body)
    names_paths = {(r.name, r.path) for r in refs}
    assert ("object_id", ("object_id",)) in names_paths
    assert ("id", ("data", "object", "id")) in names_paths
    assert ("owner_id", ("data", "object", "owner_id")) in names_paths


def test_substitute_body_path_preserves_type():
    body = {"data": {"object": {"id": 123}}}
    out = substitute_body_path(body, ("data", "object", "id"), "999")
    assert out["data"]["object"]["id"] == 999
    assert isinstance(out["data"]["object"]["id"], int)
    # original untouched
    assert body["data"]["object"]["id"] == 123


def test_extract_ownership_signals():
    body = {"id": 1, "owner_id": 10, "content": "text"}
    signals = extract_ownership_signals(body)
    assert signals.get("owner_id") == 10
    assert "id" not in signals  # 'id' alone is not an ownership fragment


def test_jwt_decode_and_self_id():
    import base64
    import json

    payload = {"data": {"id": 42, "email": "a@test.com"}}
    b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    token = f"header.{b64}.sig"
    claims = decode_jwt_payload(token)
    assert claims == payload
    assert extract_self_id_from_claims(claims) == "42"


def test_redact_headers_and_body():
    headers = {"Authorization": "Bearer abc123", "Accept": "application/json"}
    redacted = redact_headers(headers)
    assert redacted["Authorization"] == "***REDACTED***"
    assert redacted["Accept"] == "application/json"

    body = {"password": "hunter2", "email": "a@test.com"}
    redacted_body = redact_body(body)
    assert redacted_body["password"] == "***REDACTED***"
    assert redacted_body["email"] == "a@test.com"
