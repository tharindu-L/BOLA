"""
identifiers.py — Generic Object Identifier & Ownership Engine
Shared by crawler.py, engine.py and analyzer.py so that REST (OpenAPI),
GraphQL (introspection) and automatic discovery all reason about object
identifiers, ownership and substitution the same way.

Nothing in this module encodes application-specific endpoint names or
field names — every rule here is a structural/generic pattern (name
shape, value shape, nesting), not a literal signature for a known app.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Generic identifier NAME patterns.
# These are broad, structural fragments (not full literal field names from
# any specific application) used to score how "identifier-shaped" a
# parameter/property name is. A name only needs to match one fragment.
# ---------------------------------------------------------------------------
IDENTIFIER_NAME_FRAGMENTS = (
    "id", "uuid", "guid", "slug", "key", "ref", "reference",
    "username", "user_name", "login", "handle",
    "number", "num", "code", "token",
)

# Fragments that indicate an *ownership* relationship rather than the
# object's own identifier — used to correlate "who owns this object".
OWNERSHIP_NAME_FRAGMENTS = (
    "user", "owner", "author", "account", "customer",
    "created_by", "createdby", "created_by_id", "createdbyid",
    "buyer", "seller", "member", "profile",
)

# Structural (not app-specific) patterns for recognizing identifier *values*.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_INT_RE = re.compile(r"^-?\d+$")
_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{2,63}$")  # opaque token / username / slug shape
_JWT_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*$")

SECRET_HEADER_NAMES = frozenset({
    "authorization", "cookie", "set-cookie", "x-api-key", "x-auth-token",
    "proxy-authorization", "x-access-token",
})

SECRET_BODY_FRAGMENTS = (
    "password", "passwd", "secret", "token", "apikey", "api_key",
    "access_token", "refresh_token", "totp", "cvv", "authorization",
)


def name_matches_identifier(name: str) -> bool:
    """Structural check: does a parameter/property name look like it holds
    an object identifier? Matches on fragments, not full literal names."""
    lname = name.lower()
    return any(frag in lname for frag in IDENTIFIER_NAME_FRAGMENTS)


def name_matches_ownership(name: str) -> bool:
    """Structural check: does a field name look like it expresses ownership
    of an object (as opposed to being the object's own id)?"""
    lname = name.lower()
    return any(frag in lname for frag in OWNERSHIP_NAME_FRAGMENTS)


def value_looks_like_identifier(value: Any) -> bool:
    """Structural value-shape check, type-agnostic: integers, UUIDs,
    opaque alnum tokens/usernames/slugs all qualify. Free text and large
    blobs do not."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v or len(v) > 128:
        return False
    if _JWT_RE.match(v):
        return False  # tokens are not object identifiers
    if _INT_RE.match(v):
        return True
    if _UUID_RE.match(v):
        return True
    if _SLUG_RE.match(v) and " " not in v:
        return True
    return False


def classify_identifier_type(value: Any) -> str:
    if isinstance(value, int) or (isinstance(value, str) and _INT_RE.match(value)):
        return "integer"
    if isinstance(value, str) and _UUID_RE.match(value):
        return "uuid"
    return "opaque"


@dataclass
class BodyIdentifierRef:
    """A single identifier-shaped location inside a JSON body."""
    path: tuple           # sequence of dict-keys / list-indices to reach the value
    name: str              # the terminal key name
    value: Any
    value_type: str        # "integer" | "uuid" | "opaque"
    is_ownership: bool = False


def find_body_identifiers(obj: Any, _path: tuple = ()) -> list:
    """
    Recursively walk a JSON-like structure (dict/list/scalars) and return
    every location whose key name looks identifier-shaped AND whose value
    is identifier-shaped. Handles arbitrary nesting (nested body / REST
    resource relationships).
    """
    found: list[BodyIdentifierRef] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = _path + (k,)
            if isinstance(v, (dict, list)):
                found.extend(find_body_identifiers(v, new_path))
            else:
                if name_matches_identifier(k) and value_looks_like_identifier(v):
                    found.append(BodyIdentifierRef(
                        path=new_path,
                        name=k,
                        value=v,
                        value_type=classify_identifier_type(v),
                        is_ownership=name_matches_ownership(k),
                    ))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = _path + (i,)
            if isinstance(v, (dict, list)):
                found.extend(find_body_identifiers(v, new_path))
    return found


def substitute_body_path(body: Any, path: tuple, new_value: Any) -> Any:
    """Return a deep-copied body with the value at `path` replaced,
    preserving the original value's type (int stays int, str stays str)."""
    new_body = copy.deepcopy(body)
    if not path:
        return new_body
    cursor = new_body
    for key in path[:-1]:
        cursor = cursor[key]
    last = path[-1]
    original = cursor[last]
    if isinstance(original, int) and not isinstance(original, bool):
        try:
            cursor[last] = int(new_value)
        except (TypeError, ValueError):
            cursor[last] = new_value
    else:
        cursor[last] = str(new_value)
    return new_body


def extract_ownership_signals(body: Any) -> dict:
    """
    Flatten a response body and return {field_path: value} for every field
    whose name looks like an ownership/self field (owner/user/account/...).
    Used to correlate which authenticated identity an object belongs to.
    """
    signals: dict = {}

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                full = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    walk(v, full)
                elif name_matches_ownership(k) and value_looks_like_identifier(v):
                    signals[full] = v
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")

    walk(body)
    return signals


def decode_jwt_payload(token: str) -> Optional[dict]:
    """Best-effort, non-cryptographic decode of a JWT payload segment so we
    can harvest the authenticated user's own object id (e.g. 'id', 'sub',
    'user_id'). We never validate the signature — this is read-only
    introspection of a token the caller already legitimately holds."""
    import base64
    import json as _json

    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_b64 + padding)
        return _json.loads(raw)
    except Exception:
        return None


SELF_ID_FIELD_CANDIDATES = ("id", "sub", "user_id", "userid", "uid", "userId")


def extract_self_id_from_claims(claims: dict) -> Optional[str]:
    """Given decoded JWT claims or a login-response body, find the most
    likely field representing the authenticated user's own object id."""
    if not isinstance(claims, dict):
        return None
    # common wrapper: {"data": {...}}
    candidates = [claims]
    if isinstance(claims.get("data"), dict):
        candidates.append(claims["data"])
    for source in candidates:
        for field_name in SELF_ID_FIELD_CANDIDATES:
            if field_name in source and value_looks_like_identifier(source[field_name]):
                return str(source[field_name])
    return None


def redact_headers(headers: dict) -> dict:
    """Return a copy of headers with secret-bearing values masked before
    they are ever written into a report or evidence transcript."""
    redacted = {}
    for k, v in (headers or {}).items():
        if k.lower() in SECRET_HEADER_NAMES:
            redacted[k] = "***REDACTED***"
        else:
            redacted[k] = v
    return redacted


def redact_body(body: Any) -> Any:
    """Recursively mask fields whose name looks like a secret (password,
    token, cvv, ...) before a body is embedded in a report."""
    if isinstance(body, dict):
        out = {}
        for k, v in body.items():
            if any(frag in k.lower() for frag in SECRET_BODY_FRAGMENTS):
                out[k] = "***REDACTED***"
            else:
                out[k] = redact_body(v)
        return out
    if isinstance(body, list):
        return [redact_body(v) for v in body]
    return body
