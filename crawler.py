"""
crawler.py — API Crawler Module
Parses OpenAPI 3.x specs (REST) and GraphQL introspection schemas.
Produces a unified list of Operation objects.
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
    object_identifier_extraction_rule: Optional[dict] = None  # {"location": "path|query|body", "name": "id"}
    graphql_query_string: Optional[str] = None
    graphql_variables: Optional[dict] = None
    tags: list[str] = field(default_factory=list)


class RESTCrawler:
    """
    Parses an OpenAPI 3.x specification file (JSON or YAML).
    Extracts all endpoints as Operation objects.
    """

    OBJECT_ID_PATTERNS = (
        "id", "user_id", "order_id", "basket_id", "account_id",
        "product_id", "item_id", "record_id", "uuid", "slug",
    )

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

    def _extract_id_rule(self, parameters: list[dict]) -> Optional[dict]:
        for param in parameters:
            p = self._resolve_ref(param)
            name = p.get("name", "").lower()
            location = p.get("in", "")
            if any(pat in name for pat in self.OBJECT_ID_PATTERNS):
                return {"location": location, "name": p.get("name")}
        return None

    def _build_sample_payload(self, schema: Optional[dict]) -> Optional[dict]:
        if not schema:
            return None
        schema = self._resolve_ref(schema)
        props = schema.get("properties", {})
        sample: dict = {}
        for prop_name, prop_schema in props.items():
            prop_schema = self._resolve_ref(prop_schema)
            ptype = prop_schema.get("type", "string")
            if ptype == "string":
                sample[prop_name] = "test_value"
            elif ptype == "integer":
                sample[prop_name] = 1
            elif ptype == "boolean":
                sample[prop_name] = True
            elif ptype == "array":
                sample[prop_name] = []
            else:
                sample[prop_name] = None
        return sample or None

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
                id_rule = self._extract_id_rule(resolved_params)

                op = Operation(
                    operation_id=op_id,
                    api_type="rest",
                    method=method.upper(),
                    path=path,
                    parameters=resolved_params,
                    request_body_schema=body_schema,
                    sample_payload=sample_payload,
                    object_identifier_extraction_rule=id_rule,
                    tags=op_data.get("tags", []),
                )
                operations.append(op)
                logger.debug("REST op: %s %s", method.upper(), path)

        logger.info("REST crawl complete: %d operations found.", len(operations))
        return operations


class GraphQLCrawler:
    """
    Issues an introspection query against a GraphQL endpoint.
    Parses the schema and produces Operation objects for each
    Query and Mutation field, including nested selection sets.
    """

    OBJECT_ID_ARG_PATTERNS = ("id", "userId", "orderId", "vehicleId", "uuid")

    def __init__(self, endpoint_url: str, headers: Optional[dict] = None):
        self.endpoint_url = endpoint_url
        self.headers = headers or {}

    def _run_introspection(self) -> dict:
        query = get_introspection_query()
        resp = requests.post(
            self.endpoint_url,
            json={"query": query},
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
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

    def _extract_id_rule(self, args: dict) -> Optional[dict]:
        for arg_name in args:
            if any(pat.lower() in arg_name.lower() for pat in self.OBJECT_ID_ARG_PATTERNS):
                return {"location": "variable", "name": arg_name}
        return None

    def _build_gql_variables(self, args: dict) -> dict:
        variables: dict = {}
        for arg_name, arg_def in args.items():
            named = self._get_named_type(arg_def.type)
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
                id_rule = self._extract_id_rule(args)
                op_id = f"{method_label}:{field_name}"

                op = Operation(
                    operation_id=op_id,
                    api_type="graphql",
                    method=method_label,
                    query_name=field_name,
                    parameters=[{"name": k, "in": "variable"} for k in args],
                    graphql_query_string=query_string,
                    graphql_variables=variables,
                    object_identifier_extraction_rule=id_rule,
                )
                operations.append(op)
                logger.debug("GraphQL op: %s %s", method_label, field_name)

        logger.info("GraphQL crawl complete: %d operations found.", len(operations))
        return operations