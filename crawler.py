"""
crawler.py — API Crawler Module
Parses OpenAPI 3.x specs (REST) and GraphQL introspection schemas.
Produces a unified list of Operation objects.

Object-identifier detection uses the generic, structural engine in
identifiers.py (name-shape + value-shape), not literal per-application
field names, so it generalizes to unseen APIs.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from graphql import build_client_schema, get_introspection_query, parse as gql_parse
from graphql.type import GraphQLObjectType, GraphQLNonNull, GraphQLList, GraphQLScalarType, GraphQLEnumType
from openapi_spec_validator import validate

from identifiers import (
    name_matches_identifier,
    find_body_identifiers,
)

logger = logging.getLogger("bola.crawler")


@dataclass
class Operation:
    """
    Unified internal representation of an API operation.
    Abstracts REST and GraphQL into a single structure consumed
    by the Request Engine and BOLA Analyzer.
    """
    operation_id: str
    api_type: str                          # "rest" | "graphql"
    method: str                            # GET/POST/PUT/PATCH/DELETE/MUTATION/QUERY
    path: Optional[str] = None            # REST path e.g. /api/users/{id}
    query_name: Optional[str] = None      # GraphQL field name
    parameters: list[dict] = field(default_factory=list)
    request_body_schema: Optional[dict] = None
    sample_payload: Optional[dict] = None
    # Primary/legacy single rule, kept for simple callers: {"location": "path|query|body", "name": "id"}
    object_identifier_extraction_rule: Optional[dict] = None
    # Full set of identifier-shaped locations found for this operation, including
    # nested body paths: [{"location": "path|query|body", "name": str, "path": tuple|None}]
    object_identifier_rules: list[dict] = field(default_factory=list)
    graphql_query_string: Optional[str] = None
    graphql_variables: Optional[dict] = None
    # GraphQL: identifier-shaped locations nested inside input-object variables,
    # as (variable_name, path_within_variable_value)
    graphql_nested_identifier_paths: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # True for operations that mutate/delete state (used to gate write verification).
    is_write: bool = False


WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE", "MUTATION"})


class RESTCrawler:
    """
    Parses an OpenAPI 3.x specification file (JSON or YAML).
    Extracts all endpoints as Operation objects.
    """

    def __init__(self, spec_path: str):
        self.spec_path = Path(spec_path)
        self.spec: dict = {}

    def _load_spec(self) -> dict:
        raw = self.spec_path.read_text(encoding="utf-8")
        if self.spec_path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)
        try:
            validate(data)
            logger.debug("OpenAPI spec validation passed.")
        except Exception as exc:
            logger.warning("OpenAPI spec validation warning: %s", exc)
        return data

    def _resolve_ref(self, obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "$ref" in obj:
            ref_path = obj["$ref"].lstrip("#/").split("/")
            resolved = self.spec
            for part in ref_path:
                resolved = resolved.get(part, {})
            return resolved
        return obj

    def _extract_param_id_rules(self, parameters: list[dict]) -> list[dict]:
        """
        Path parameters are, by REST convention, almost always the object
        identifier for that resource position regardless of what the
        author happened to name them (/books/v1/{book_title},
        /users/v1/{username}, /resource/{id} are all structurally the
        same shape) — so every path parameter is treated as a candidate.
        Query parameters are far more often unrelated filters/pagination,
        so those still require an identifier-shaped name to qualify.
        """
        rules = []
        for param in parameters:
            p = self._resolve_ref(param)
            name = p.get("name", "")
            location = p.get("in", "")
            if location == "path":
                rules.append({"location": location, "name": name})
            elif location == "query" and name_matches_identifier(name):
                rules.append({"location": location, "name": name})
        return rules

    def _build_sample_payload(self, schema: Optional[dict]) -> Optional[dict]:
        if not schema:
            return None
        schema = self._resolve_ref(schema)
        props = schema.get("properties", {})
        sample: dict = {}
        for prop_name, prop_schema in props.items():
            prop_schema = self._resolve_ref(prop_schema)
            ptype = prop_schema.get("type", "string")
            example = prop_schema.get("example")
            if example is not None:
                sample[prop_name] = example
            elif ptype == "string":
                # Identifier-shaped properties get an id-like placeholder so
                # the request engine has something structurally valid to
                # substitute; everything else gets a generic filler value.
                sample[prop_name] = "1" if name_matches_identifier(prop_name) else "test_value"
            elif ptype == "integer":
                sample[prop_name] = 1
            elif ptype == "boolean":
                sample[prop_name] = True
            elif ptype == "array":
                sample[prop_name] = []
            elif ptype == "object":
                nested = self._build_sample_payload(prop_schema)
                sample[prop_name] = nested or {}
            else:
                sample[prop_name] = None
        return sample or None

    def _extract_body_id_rules(self, sample_payload: Optional[dict]) -> list[dict]:
        if not sample_payload:
            return []
        refs = find_body_identifiers(sample_payload)
        return [{"location": "body", "name": r.name, "path": r.path} for r in refs]

    def crawl(self) -> list[Operation]:
        self.spec = self._load_spec()
        operations: list[Operation] = []
        paths = self.spec.get("paths", {})

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            shared_params = path_item.get("parameters", [])

            for method in ("get", "post", "put", "patch", "delete"):
                op_data = path_item.get(method)
                if not op_data:
                    continue

                op_data = self._resolve_ref(op_data)
                op_params = shared_params + op_data.get("parameters", [])
                resolved_params = [self._resolve_ref(p) for p in op_params]
                request_body = op_data.get("requestBody")
                body_schema = None
                sample_payload = None

                if request_body:
                    rb = self._resolve_ref(request_body)
                    content = rb.get("content", {})
                    json_content = content.get("application/json", {})
                    body_schema = self._resolve_ref(json_content.get("schema"))
                    sample_payload = self._build_sample_payload(body_schema)

                op_id = op_data.get("operationId") or f"{method.upper()}:{path}"
                param_rules = self._extract_param_id_rules(resolved_params)
                body_rules = self._extract_body_id_rules(sample_payload)
                all_rules = param_rules + body_rules
                # Prefer a path-located rule as the "primary" one (most common
                # BOLA shape) for callers that only care about a single rule.
                primary = next((r for r in all_rules if r["location"] == "path"), None) \
                    or (all_rules[0] if all_rules else None)

                op = Operation(
                    operation_id=op_id,
                    api_type="rest",
                    method=method.upper(),
                    path=path,
                    parameters=resolved_params,
                    request_body_schema=body_schema,
                    sample_payload=sample_payload,
                    object_identifier_extraction_rule=primary,
                    object_identifier_rules=all_rules,
                    tags=op_data.get("tags", []),
                    is_write=method.upper() in WRITE_METHODS,
                )
                operations.append(op)
                logger.debug("REST op: %s %s (rules=%d)", method.upper(), path, len(all_rules))

        logger.info("REST crawl complete: %d operations found.", len(operations))
        return operations


class GraphQLCrawler:
    """
    Issues an introspection query against a GraphQL endpoint.
    Parses the schema and produces Operation objects for each
    Query and Mutation field, including nested selection sets.
    """

    def __init__(self, endpoint_url: str, headers: Optional[dict] = None):
        self.endpoint_url = endpoint_url
        self.headers = headers or {}

    def _run_introspection(self) -> dict:
        query = get_introspection_query()
        endpoint = self.endpoint_url
        if not endpoint.endswith("/graphql"):
            endpoint = endpoint.rstrip("/") + "/graphql"

        logger.debug("Introspection endpoint: %s", endpoint)

        resp = requests.post(
            endpoint,
            json={"query": query},
            headers={
                **self.headers,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30,
        )

        logger.debug("Introspection status: %d", resp.status_code)
        logger.debug("Introspection response length: %d", len(resp.text))

        resp.raise_for_status()

        if not resp.text or not resp.text.strip():
            raise RuntimeError(
                f"Introspection returned empty response from {endpoint}. "
                f"Status: {resp.status_code}. "
                f"Check that GraphQL introspection is enabled on the target."
            )

        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"Introspection errors: {data['errors']}")
        return data["data"]

    def _get_named_type(self, type_ref: Any) -> Any:
        while isinstance(type_ref, (GraphQLNonNull, GraphQLList)):
            type_ref = type_ref.of_type
        return type_ref

    def _build_selection_set(self, gql_type: Any, depth: int = 0, max_depth: int = 3) -> str:
        if depth >= max_depth:
            return ""
        named = self._get_named_type(gql_type)
        if not isinstance(named, GraphQLObjectType):
            return ""
        fields_str_parts = []
        for fname, fdef in named.fields.items():
            inner = self._get_named_type(fdef.type)
            if isinstance(inner, GraphQLObjectType):
                nested = self._build_selection_set(inner, depth + 1, max_depth)
                if nested:
                    fields_str_parts.append(f"{fname} {{ {nested} }}")
            elif isinstance(inner, (GraphQLScalarType, GraphQLEnumType)):
                fields_str_parts.append(fname)
        return " ".join(fields_str_parts)

    def _extract_arg_id_rules(self, args: dict) -> list[dict]:
        """Every scalar argument that is identifier-shaped by name."""
        rules = []
        for arg_name, arg_def in args.items():
            named = self._get_named_type(arg_def.type)
            if isinstance(named, GraphQLObjectType):
                continue  # handled by _extract_nested_input_id_paths
            if name_matches_identifier(arg_name):
                rules.append({"location": "variable", "name": arg_name})
        return rules

    def _extract_nested_input_id_paths(self, args: dict) -> list[dict]:
        """Scan input-object arguments (e.g. mutation(input: {id, ...})) for
        identifier-shaped nested fields, so mutations that wrap the object
        id inside an input type are still substitutable."""
        nested: list[dict] = []
        for arg_name, arg_def in args.items():
            named = self._get_named_type(arg_def.type)
            fields_attr = getattr(named, "fields", None)
            if not fields_attr:
                continue
            for field_name, field_def in fields_attr.items():
                inner = self._get_named_type(field_def.type)
                if isinstance(inner, GraphQLObjectType):
                    continue
                if name_matches_identifier(field_name):
                    nested.append({"variable": arg_name, "field": field_name})
        return nested

    def _build_gql_variables(self, args: dict) -> dict:
        variables: dict = {}
        for arg_name, arg_def in args.items():
            named = self._get_named_type(arg_def.type)
            fields_attr = getattr(named, "fields", None)
            if fields_attr:
                # Input object type — build a minimal nested sample so
                # identifier-shaped nested fields exist to be substituted.
                inner_val: dict = {}
                for fname, fdef in fields_attr.items():
                    inner_named = self._get_named_type(fdef.type)
                    inner_type_name = getattr(inner_named, "name", "String")
                    if "Int" in inner_type_name:
                        inner_val[fname] = 1
                    elif "Boolean" in inner_type_name:
                        inner_val[fname] = True
                    elif name_matches_identifier(fname):
                        inner_val[fname] = "1"
                    else:
                        inner_val[fname] = "test_value"
                variables[arg_name] = inner_val
                continue
            type_name = getattr(named, "name", "String")
            if "Int" in type_name:
                variables[arg_name] = 1
            elif "Boolean" in type_name:
                variables[arg_name] = True
            else:
                variables[arg_name] = "test_value"
        return variables

    def crawl(self) -> list[Operation]:
        logger.info("Running GraphQL introspection on %s", self.endpoint_url)
        intro_data = self._run_introspection()
        schema = build_client_schema(intro_data)
        operations: list[Operation] = []

        type_map = {
            "query": (schema.query_type, "QUERY"),
            "mutation": (schema.mutation_type, "MUTATION"),
        }

        for op_kind, (root_type, method_label) in type_map.items():
            if not root_type:
                continue
            for field_name, field_def in root_type.fields.items():
                args = field_def.args or {}
                selection = self._build_selection_set(field_def.type)
                if not selection:
                    selection = "__typename"

                arg_defs_str = ""
                var_defs_str = ""
                if args:
                    arg_pairs = ", ".join(f"${k}: {self._get_named_type(v.type).name}" for k, v in args.items())
                    var_defs_str = f"({arg_pairs})"
                    arg_use = ", ".join(f"{k}: ${k}" for k in args)
                    arg_defs_str = f"({arg_use})"

                query_string = (
                    f"{op_kind} BolaScan{var_defs_str} {{\n"
                    f"  {field_name}{arg_defs_str} {{\n"
                    f"    {selection}\n"
                    f"  }}\n"
                    f"}}"
                )
                variables = self._build_gql_variables(args)
                arg_rules = self._extract_arg_id_rules(args)
                nested_rules = self._extract_nested_input_id_paths(args)
                primary = arg_rules[0] if arg_rules else None
                op_id = f"{method_label}:{field_name}"

                op = Operation(
                    operation_id=op_id,
                    api_type="graphql",
                    method=method_label,
                    query_name=field_name,
                    parameters=[{"name": k, "in": "variable"} for k in args],
                    graphql_query_string=query_string,
                    graphql_variables=variables,
                    object_identifier_extraction_rule=primary,
                    object_identifier_rules=arg_rules,
                    graphql_nested_identifier_paths=nested_rules,
                    is_write=method_label == "MUTATION",
                )
                operations.append(op)
                logger.debug(
                    "GraphQL op: %s %s (arg_rules=%d nested_rules=%d)",
                    method_label, field_name, len(arg_rules), len(nested_rules),
                )

        logger.info("GraphQL crawl complete: %d operations found.", len(operations))
        return operations
