"""
cli.py — Command-Line Interface
Entry point: bola-framework scan
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from login import AutoLogin

import click
from rich.console import Console
from rich.table import Table
from rich import box

from auth import AuthManager
from analyzer import BOLAAnalyzer, Finding
from crawler import RESTCrawler, GraphQLCrawler, Operation
from engine import RequestEngine

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _is_graphql_spec(spec: str) -> bool:
    return spec.lower() in ("graphql", "gql") or spec.startswith("http")


def _print_findings_table(findings: list[Finding]) -> None:
    if not findings:
        console.print("\n[bold green]No BOLA vulnerabilities detected.[/bold green]\n")
        return

    table = Table(
        title=f"\n[bold red]BOLA Findings — {len(findings)} vulnerability(s)[/bold red]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Severity", style="bold", width=8)
    table.add_column("API Type", width=8)
    table.add_column("Method", width=10)
    table.add_column("Path / Query", min_width=25)
    table.add_column("Similarity", width=10)
    table.add_column("Operation ID", min_width=20)

    for f in findings:
        color = "red" if f.severity == "High" else "yellow"
        table.add_row(
            f"[{color}]{f.severity}[/{color}]",
            f.api_type.upper(),
            f.method,
            f.path_or_query,
            f"{f.similarity_score:.2%}" if f.similarity_score is not None else "N/A",
            f.operation_id,
        )

    console.print(table)


def _build_json_report(
    findings: list[Finding],
    target: str,
    spec: str,
    elapsed_sec: float,
) -> dict:
    return {
        "meta": {
            "tool": "BOLA Detection Framework",
            "version": "1.0.0",
            "target": target,
            "spec": spec,
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed_sec, 2),
            "total_findings": len(findings),
        },
        "findings": [
            {
                "operation_id": f.operation_id,
                "severity": f.severity,
                "api_type": f.api_type,
                "method": f.method,
                "path_or_query": f.path_or_query,
                "similarity_score": f.similarity_score,
                "field_leakage": f.field_leakage,
                "reproduction_steps": f.reproduction_steps,
                "evidence": f.evidence,
            }
            for f in findings
        ],
    }


@click.group()
def cli() -> None:
    """BOLA Detection Framework — Automated black-box BOLA scanner for REST and GraphQL APIs."""
    pass


@cli.command("scan")
@click.option("--target", "-t", required=True, help="Base URL of the target API.")
@click.option(
    "--spec", "-s", required=True,
    help="Path to OpenAPI spec file (JSON/YAML) or 'graphql' for introspection."
)
@click.option("--auth-a", required=True, help="Auth credentials for User A. e.g. 'Bearer token123'")
@click.option("--auth-b", required=True, help="Auth credentials for User B. e.g. 'Bearer token456'")
@click.option("--id-a", default="1", show_default=True, help="Object ID belonging to User A.")
@click.option("--id-b", default="2", show_default=True, help="Object ID belonging to User B.")
@click.option(
    "--output", "-o", default=None,
    help="Path to write JSON report. Prints to stdout if omitted."
)
@click.option("--threshold", default=0.85, show_default=True, help="Similarity threshold (0.0–1.0).")
@click.option("--no-ssl-verify", is_flag=True, default=False, help="Disable SSL certificate verification.")
@click.option("--timeout", default=20, show_default=True, help="Request timeout in seconds.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable debug logging.")
@click.option("--user-a", default=None, help="Username/email for User A (auto-login).")
@click.option("--pass-a", default=None, help="Password for User A (auto-login).")
@click.option("--user-b", default=None, help="Username/email for User B (auto-login).")
@click.option("--pass-b", default=None, help="Password for User B (auto-login).")
def scan(
    target: str,
    spec: str,
    auth_a: str,
    auth_b: str,
    id_a: str,
    id_b: str,
    output: Optional[str],
    threshold: float,
    no_ssl_verify: bool,
    timeout: int,
    verbose: bool,
    user_a: Optional[str],
    pass_a: Optional[str],
    user_b: Optional[str],
    pass_b: Optional[str],
) -> None:
    """
    Run a BOLA scan against a REST or GraphQL API.

    Examples:\n
      # REST scan\n
      bola-framework scan --target http://localhost:3000 --spec ./openapi.json \\
        --auth-a "Bearer tokenA" --auth-b "Bearer tokenB" --id-a 1 --id-b 2\n

      # GraphQL scan\n
      bola-framework scan --target http://localhost:8888 --spec graphql \\
        --auth-a "Bearer tokenA" --auth-b "Bearer tokenB" --id-a user1 --id-b user2
    """
    import time as _time
    _configure_logging(verbose)
    logger = logging.getLogger("bola.cli")
    start_time = _time.monotonic()

    console.print("\n[bold cyan]BOLA Detection Framework[/bold cyan] — Starting scan...\n")
    console.print(f"  Target : [yellow]{target}[/yellow]")
    console.print(f"  Spec   : [yellow]{spec}[/yellow]")
    console.print(f"  ID-A   : [yellow]{id_a}[/yellow]  |  ID-B : [yellow]{id_b}[/yellow]\n")

    # --- Phase 1: Crawl ---
    operations: list[Operation] = []
    graphql_endpoint: Optional[str] = None

    try:
        if _is_graphql_spec(spec):
            graphql_endpoint = spec if spec.startswith("http") else target
            if spec.lower() in ("graphql", "gql"):
                graphql_endpoint = target
            console.print("[*] Running GraphQL introspection...")
            crawler = GraphQLCrawler(endpoint_url=graphql_endpoint)
            operations = crawler.crawl()
        else:
            console.print("[*] Parsing OpenAPI specification...")
            crawler = RESTCrawler(spec_path=spec)
            operations = crawler.crawl()
    except Exception as exc:
        console.print(f"[bold red]Crawl failed:[/bold red] {exc}")
        logger.error("Crawl failed: %s", exc, exc_info=verbose)
        sys.exit(1)

    if not operations:
        console.print("[bold yellow]No operations found. Check your spec or target.[/bold yellow]")
        sys.exit(0)

    console.print(f"[green]Found {len(operations)} operation(s).[/green]\n")

        # --- Phase 2: Auth (auto-login if credentials provided) ---
    try:
        # Auto-login takes priority over manual tokens if credentials supplied
        if user_a and pass_a:
            console.print("[*] Auto-login: authenticating User A...")
            auto = AutoLogin(base_url=target, verify_ssl=not no_ssl_verify)
            result_a = auto.login(user_a, pass_a, "A")
            if not result_a.success:
                console.print(f"[bold red]Auto-login failed for User A:[/bold red] {result_a.error}")
                sys.exit(1)
            auth_a = result_a.auth_string
            console.print(f"[green]User A authenticated:[/green] {auth_a[:40]}...")

        if user_b and pass_b:
            console.print("[*] Auto-login: authenticating User B...")
            auto = AutoLogin(base_url=target, verify_ssl=not no_ssl_verify)
            result_b = auto.login(user_b, pass_b, "B")
            if not result_b.success:
                console.print(f"[bold red]Auto-login failed for User B:[/bold red] {result_b.error}")
                sys.exit(1)
            auth_b = result_b.auth_string
            console.print(f"[green]User B authenticated:[/green] {auth_b[:40]}...")

        console.print("[*] Initializing sessions...")
        auth_mgr = AuthManager(auth_a=auth_a, auth_b=auth_b, verify_ssl=not no_ssl_verify)
        auth_mgr.authenticate()
    except Exception as exc:
        console.print(f"[bold red]Authentication failed:[/bold red] {exc}")
        logger.error("Auth error: %s", exc, exc_info=verbose)
        sys.exit(1)

    # --- Phase 3: Execute ---
    try:
        console.print("[*] Executing requests through both sessions...")
        engine = RequestEngine(
            base_url=target,
            session_a=auth_mgr.session_a,
            session_b=auth_mgr.session_b,
            object_id_a=id_a,
            object_id_b=id_b,
            timeout=timeout,
        )
        results = engine.execute(operations, graphql_endpoint=graphql_endpoint)
    except Exception as exc:
        console.print(f"[bold red]Request engine error:[/bold red] {exc}")
        logger.error("Engine error: %s", exc, exc_info=verbose)
        auth_mgr.close()
        sys.exit(1)

    # --- Phase 4: Analyze ---
    try:
        console.print("[*] Analyzing responses for BOLA...")
        analyzer = BOLAAnalyzer(similarity_threshold=threshold)
        findings = analyzer.analyze(results)
    except Exception as exc:
        console.print(f"[bold red]Analyzer error:[/bold red] {exc}")
        logger.error("Analyzer error: %s", exc, exc_info=verbose)
        auth_mgr.close()
        sys.exit(1)

    auth_mgr.close()
    elapsed = _time.monotonic() - start_time

    # --- Output ---
    _print_findings_table(findings)

    report = _build_json_report(findings, target, spec, elapsed)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"\n[green]Report written to:[/green] {out_path.resolve()}")
    else:
        console.print("\n[bold]JSON Report:[/bold]")
        console.print_json(json.dumps(report, indent=2))

    console.print(f"\nScan completed in [cyan]{elapsed:.2f}s[/cyan].")
    exit_code = 1 if findings else 0
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()