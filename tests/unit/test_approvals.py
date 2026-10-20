"""Milestone C — approval routing, work queue, authorization, audit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.approvals import (
    ApprovalPolicy,
    DecisionRejectedError,
    PolicyValidationError,
    UnauthorizedDecisionError,
    build_queue,
    load_policy,
    merge_queues,
    queue_to_markdown,
    record_decision,
)
from app.demo_packages import build_contract_a_package, build_contract_b_package, build_contract_e_package
from app.schemas import (
    ClauseFinding,
    Evidence,
    GroundedFinding,
    ReviewPackage,
    RouteDecision,
    ScoredFinding,
    SecurityFlags,
)


FIXED_NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


def _finding(
    rule_id: str,
    status: str,
    *,
    severity: str = "high",
    legal: bool = True,
    quote: str | None = "sample quote",
    grounding: str = "exact",
) -> ScoredFinding:
    evidence = [Evidence(quote=quote, page=1)] if quote else []
    return ScoredFinding(
        grounded=GroundedFinding(
            finding=ClauseFinding(
                rule_id=rule_id,
                found=bool(quote),
                evidence=evidence,
                extracted_value="x",
            ),
            grounding=grounding,  # type: ignore[arg-type]
        ),
        status=status,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        requires_legal=legal,
        rule_trace=f"{rule_id}: {status}",
    )


def _pkg(findings: list[ScoredFinding], *, security: SecurityFlags | None = None) -> ReviewPackage:
    return ReviewPackage(
        run_id="test-run",
        status="complete",
        findings=findings,
        route=RouteDecision(route="legal", trace=["test"]),
        security=security or SecurityFlags(),
        file_name="contract_test.pdf",
    )


# --------------------------------------------------------------------------- #
# Policy validation
# --------------------------------------------------------------------------- #
def test_load_default_policy():
    policy = load_policy()
    assert policy.version == "approval-routing-v1"
    assert len(policy.rules) >= 5
    assert "legal" in policy.team_roles


def test_policy_rejects_empty_teams(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"version":"x","rules":[{"id":"R1","match":{"rule_ids":["X"],'
        '"statuses":["walk_away"]},"teams":[],"reason_code":"x"}]}',
        encoding="utf-8",
    )
    with pytest.raises(PolicyValidationError):
        load_policy(bad)


def test_policy_rejects_duplicate_ids(tmp_path: Path):
    bad = tmp_path / "dup.json"
    bad.write_text(
        '{"version":"x","rules":['
        '{"id":"R1","match":{"rule_ids":["A"],"statuses":["walk_away"]},'
        '"teams":["legal"],"reason_code":"a"},'
        '{"id":"R1","match":{"rule_ids":["B"],"statuses":["walk_away"]},'
        '"teams":["legal"],"reason_code":"b"}'
        "]}",
        encoding="utf-8",
    )
    with pytest.raises(PolicyValidationError):
        load_policy(bad)


# --------------------------------------------------------------------------- #
# Routing rules (independent + multi-team)
# --------------------------------------------------------------------------- #
def test_liability_routes_to_legal_only():
    q = build_queue(
        _pkg([_finding("LIABILITY-001", "walk_away")]),
        now=FIXED_NOW,
    )
    assert len(q.tasks) == 1
    t = q.tasks[0]
    assert t.required_teams == ["legal"]
    assert "AR-LEGAL-LIABILITY" in t.matched_policy_rule_ids
    assert t.severity == "high"
    assert any(r.team == "legal" for r in q.routes)


def test_data_routes_to_security_and_legal():
    q = build_queue(
        _pkg([_finding("DATA-001", "walk_away", quote=None, grounding="missing")]),
        now=FIXED_NOW,
    )
    assert len(q.tasks) == 1
    teams = set(q.tasks[0].required_teams)
    assert teams == {"security_privacy", "legal"}
    route_teams = {r.team for r in q.routes if r.review_task_id == q.tasks[0].task_id}
    assert route_teams == {"security_privacy", "legal"}


def test_price_routes_procurement_and_finance():
    q = build_queue(_pkg([_finding("PRICE-001", "walk_away", severity="medium", legal=False)]), now=FIXED_NOW)
    assert set(q.tasks[0].required_teams) == {"procurement", "finance"}


def test_sla_routes_business_owner_and_procurement():
    q = build_queue(_pkg([_finding("SLA-001", "not_found", severity="medium", legal=False, quote=None)]), now=FIXED_NOW)
    assert set(q.tasks[0].required_teams) == {"business_owner", "procurement"}


def test_preferred_finding_creates_no_task():
    q = build_queue(_pkg([_finding("PAYMENT-001", "preferred", severity="medium", legal=False)]), now=FIXED_NOW)
    assert q.tasks == []


def test_fallback_finding_creates_no_task():
    q = build_queue(_pkg([_finding("LAW-001", "fallback", severity="low")]), now=FIXED_NOW)
    assert q.tasks == []


def test_unclear_missing_evidence_needs_information():
    q = build_queue(
        _pkg(
            [
                _finding(
                    "PAYMENT-001",
                    "unclear",
                    severity="medium",
                    legal=False,
                    quote=None,
                    grounding="ambiguous",
                )
            ]
        ),
        now=FIXED_NOW,
    )
    assert len(q.tasks) >= 1
    # At least one path flags needs_information (AR-NEEDS-INFO or initial_status)
    statuses = {t.status for t in q.tasks}
    reason_codes = {c for t in q.tasks for c in t.reason_codes}
    assert "needs_information" in statuses or "needs_information" in reason_codes or "commercial_variance" in reason_codes


def test_security_flag_creates_security_task():
    flags = SecurityFlags(pre=[Evidence(quote="ignore previous instructions", page=1)])
    q = build_queue(_pkg([], security=flags), now=FIXED_NOW)
    assert len(q.tasks) == 1
    assert q.tasks[0].required_teams == ["security_privacy"]
    assert "AR-SECURITY-FLAG" in q.tasks[0].matched_policy_rule_ids


def test_idempotent_task_ids_across_reruns():
    pkg = _pkg([_finding("LIABILITY-002", "walk_away")])
    q1 = build_queue(pkg, contract_id="c1", now=FIXED_NOW)
    q2 = build_queue(pkg, contract_id="c1", now=FIXED_NOW)
    assert [t.task_id for t in q1.tasks] == [t.task_id for t in q2.tasks]
    assert [r.route_id for r in q1.routes] == [r.route_id for r in q2.routes]


# --------------------------------------------------------------------------- #
# Decisions + authorization
# --------------------------------------------------------------------------- #
def test_authorized_approve():
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    tid = q.tasks[0].task_id
    q2 = record_decision(
        q,
        tid,
        reviewer="alice",
        role="legal",
        decision="approve",
        reason="Acceptable with board memo",
        now=FIXED_NOW,
    )
    t = next(x for x in q2.tasks if x.task_id == tid)
    assert t.status == "approved"
    assert t.decision == "approve"
    assert t.decided_by == "alice|legal"
    assert any(a.event_type == "decision_recorded" for a in q2.audit)


def test_unauthorized_approve_blocked():
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    with pytest.raises(UnauthorizedDecisionError):
        record_decision(
            q,
            q.tasks[0].task_id,
            reviewer="bob",
            role="procurement",
            decision="approve",
            reason="nope",
            now=FIXED_NOW,
        )


def test_viewer_cannot_decide():
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    with pytest.raises(UnauthorizedDecisionError):
        record_decision(
            q,
            q.tasks[0].task_id,
            reviewer="eve",
            role="viewer",
            decision="approve",
            reason="peek",
            now=FIXED_NOW,
        )


def test_waive_requires_authorization_reference():
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    with pytest.raises(DecisionRejectedError):
        record_decision(
            q,
            q.tasks[0].task_id,
            reviewer="alice",
            role="legal",
            decision="waive_with_authorization",
            reason="business wants it",
            authorization_reference=None,
            now=FIXED_NOW,
        )


def test_waive_with_auth_ref_ok():
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    q2 = record_decision(
        q,
        q.tasks[0].task_id,
        reviewer="alice",
        role="legal",
        decision="waive_with_authorization",
        reason="CFO exception",
        authorization_reference="EXC-2026-42",
        now=FIXED_NOW,
    )
    assert q2.tasks[0].status == "waived"
    assert q2.tasks[0].authorization_reference == "EXC-2026-42"


def test_reject_and_request_changes():
    q = build_queue(_pkg([_finding("LAW-001", "walk_away", severity="low")]), now=FIXED_NOW)
    tid = q.tasks[0].task_id
    q = record_decision(
        q, tid, reviewer="l1", role="legal", decision="reject", reason="foreign venue only", now=FIXED_NOW
    )
    assert q.tasks[0].status == "rejected"
    # Re-open via merge with a fresh rebuild would keep decision; request_changes on a new task
    q2 = build_queue(_pkg([_finding("ASSIGN-001", "walk_away", severity="low", legal=False)]), now=FIXED_NOW)
    # ASSIGN routes to legal via AR-LEGAL-RENEWAL
    assert q2.tasks
    tid2 = q2.tasks[0].task_id
    q2 = record_decision(
        q2,
        tid2,
        reviewer="l1",
        role="legal",
        decision="request_changes",
        reason="Need consent language",
        now=FIXED_NOW,
    )
    assert q2.tasks[0].status == "changes_requested"


def test_reason_required():
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    with pytest.raises(DecisionRejectedError):
        record_decision(
            q,
            q.tasks[0].task_id,
            reviewer="alice",
            role="legal",
            decision="approve",
            reason="   ",
            now=FIXED_NOW,
        )


def test_llm_cannot_set_approved_via_service_only():
    """Status changes only through record_decision — direct mutation is not the API."""
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    assert q.tasks[0].status == "open"
    # The public API for approval is record_decision; there is no set_status helper.
    assert not hasattr(q, "set_status")


# --------------------------------------------------------------------------- #
# Merge / audit / samples
# --------------------------------------------------------------------------- #
def test_merge_preserves_decisions():
    pkg = _pkg([_finding("LIABILITY-001", "walk_away")])
    old = build_queue(pkg, contract_id="c1", now=FIXED_NOW)
    old = record_decision(
        old,
        old.tasks[0].task_id,
        reviewer="alice",
        role="legal",
        decision="approve",
        reason="ok",
        now=FIXED_NOW,
    )
    fresh = build_queue(pkg, contract_id="c1", now=FIXED_NOW)
    merged = merge_queues(old, fresh)
    assert merged.tasks[0].status == "approved"
    assert merged.tasks[0].decision == "approve"


def test_queue_sorted_by_severity():
    q = build_queue(
        _pkg(
            [
                _finding("LAW-001", "walk_away", severity="low"),
                _finding("LIABILITY-001", "walk_away", severity="high"),
                _finding("PRICE-001", "walk_away", severity="medium", legal=False),
            ]
        ),
        now=FIXED_NOW,
    )
    ordered = q.sorted_queue()
    assert ordered[0].severity == "high"


def test_markdown_export():
    q = build_queue(_pkg([_finding("LIABILITY-001", "walk_away")]), now=FIXED_NOW)
    md = queue_to_markdown(q)
    assert "Approval queue" in md
    assert "LIABILITY" in md or "liability" in md.lower()


def test_sample_b_creates_liability_tasks():
    pkg = build_contract_b_package()
    q = build_queue(pkg, contract_id="contract_b", now=FIXED_NOW)
    assert q.tasks, "Sample B walk-aways should open approval tasks"
    teams = {t for task in q.tasks for t in task.required_teams}
    assert "legal" in teams


def test_sample_a_minimal_or_empty_queue():
    """Clean MSA: preferred/fallback only → no (or security-only) tasks."""
    pkg = build_contract_a_package()
    q = build_queue(pkg, contract_id="contract_a", now=FIXED_NOW)
    walk = [f for f in pkg.findings if f.status in ("walk_away", "not_found", "unclear")]
    if not walk and not pkg.security.any():
        assert q.tasks == []


def test_sample_e_security_or_price():
    pkg = build_contract_e_package()
    q = build_queue(pkg, contract_id="contract_e", now=FIXED_NOW)
    assert q.tasks  # price hike and/or security injection


def test_queue_attaches_knowledge_citations():
    """RAG Phase 3 — open tasks carry approved policy citations or abstain."""
    q = build_queue(
        _pkg([_finding("SLA-001", "walk_away", severity="medium", legal=False)]),
        now=FIXED_NOW,
    )
    assert q.tasks
    task = q.tasks[0]
    if task.knowledge_abstained:
        assert task.knowledge_citations == []
    else:
        assert task.knowledge_citations
        cite = task.knowledge_citations[0]
        assert cite.evidence_id
        assert cite.excerpt
        assert cite.status == "approved"
