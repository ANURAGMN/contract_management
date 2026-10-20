"""Milestone C (spec §21) — approval routing and exception work queue.

Routing is a deterministic policy decision. LLMs never set task status to
approved. Demo UI may pick a mock role; the service layer still enforces
authorization against the policy file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from app.schemas import ReviewPackage, ScoredFinding, SecurityFlags

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "data" / "playbook" / "approval-routing-v1.json"

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
Team = Literal[
    "procurement",
    "legal",
    "security_privacy",
    "finance",
    "business_owner",
]
RouteStatus = Literal[
    "open",
    "in_review",
    "needs_information",
    "approved",
    "rejected",
    "changes_requested",
    "waived",
    "cancelled",
]
TaskStatus = RouteStatus
DecisionKind = Literal[
    "approve",
    "reject",
    "request_changes",
    "need_information",
    "waive_with_authorization",
]
Severity = Literal["high", "medium", "low"]
ActorRole = Literal[
    "procurement",
    "legal",
    "security_privacy",
    "finance",
    "business_owner",
    "admin",
    "viewer",
]


# --------------------------------------------------------------------------- #
# Policy models (loaded from JSON)
# --------------------------------------------------------------------------- #
class MatchSpec(BaseModel):
    rule_ids: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    missing_evidence: bool = False
    security_flag: bool = False


class PolicyRule(BaseModel):
    id: str
    priority: int = 100
    match: MatchSpec
    teams: list[Team]
    severity: Severity = "medium"
    reason_code: str
    reason_template: str = ""
    mandatory_decision: bool = True
    required_inputs: list[str] = Field(default_factory=list)
    initial_status: RouteStatus = "open"


class ApprovalPolicy(BaseModel):
    version: str
    description: str = ""
    rules: list[PolicyRule]
    team_roles: dict[str, list[str]] = Field(default_factory=dict)

    def sorted_rules(self) -> list[PolicyRule]:
        return sorted(self.rules, key=lambda r: r.priority)


class PolicyValidationError(ValueError):
    """Raised when the approval-routing policy file is invalid."""


def load_policy(path: Path | str | None = None) -> ApprovalPolicy:
    p = Path(path) if path else DEFAULT_POLICY_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    try:
        policy = ApprovalPolicy.model_validate(raw)
    except ValidationError as exc:
        raise PolicyValidationError(str(exc)) from exc
    _validate_policy(policy)
    return policy


def _validate_policy(policy: ApprovalPolicy) -> None:
    if not policy.version:
        raise PolicyValidationError("policy.version is required")
    if not policy.rules:
        raise PolicyValidationError("policy.rules must not be empty")
    ids: set[str] = set()
    for rule in policy.rules:
        if rule.id in ids:
            raise PolicyValidationError(f"duplicate rule id: {rule.id}")
        ids.add(rule.id)
        if not rule.teams:
            raise PolicyValidationError(f"{rule.id}: teams must not be empty")
        if not rule.reason_code:
            raise PolicyValidationError(f"{rule.id}: reason_code is required")
        m = rule.match
        if not (m.rule_ids or m.security_flag or m.missing_evidence):
            raise PolicyValidationError(
                f"{rule.id}: match must specify rule_ids, security_flag, or missing_evidence"
            )


# --------------------------------------------------------------------------- #
# Runtime models
# --------------------------------------------------------------------------- #
class AuditEvent(BaseModel):
    event_id: str
    timestamp: datetime
    actor: str
    actor_role: ActorRole
    event_type: str
    task_id: Optional[str] = None
    route_id: Optional[str] = None
    detail: str = ""
    before_status: Optional[str] = None
    after_status: Optional[str] = None


class HumanDecision(BaseModel):
    reviewer: str
    role: ActorRole
    decision: DecisionKind
    reason: str
    timestamp: datetime
    authorization_reference: Optional[str] = None


class ApprovalRoute(BaseModel):
    route_id: str
    review_task_id: str
    team: Team
    reason_code: str
    reason: str
    matched_policy_rule_ids: list[str] = Field(default_factory=list)
    severity: Severity = "medium"
    source_evidence_ids: list[str] = Field(default_factory=list)
    status: RouteStatus = "open"
    required_inputs: list[str] = Field(default_factory=list)
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class KnowledgeCitation(BaseModel):
    """RAG evidence explaining why an approval task exists (normative policy)."""

    evidence_id: str
    chunk_id: str
    title: str
    excerpt: str
    version: Optional[str] = None
    status: Optional[str] = None
    document_type: str = ""
    relevance_score: Optional[float] = None


class ReviewTask(BaseModel):
    task_id: str
    contract_id: str
    related_finding_ids: list[str] = Field(default_factory=list)
    related_obligation_ids: list[str] = Field(default_factory=list)
    required_teams: list[Team] = Field(default_factory=list)
    status: TaskStatus = "open"
    severity: Severity = "medium"
    title: str = ""
    reason_codes: list[str] = Field(default_factory=list)
    matched_policy_rule_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    knowledge_citations: list[KnowledgeCitation] = Field(default_factory=list)
    knowledge_abstained: bool = False
    due_at: Optional[datetime] = None
    decision: Optional[DecisionKind] = None
    decision_reason: Optional[str] = None
    decided_by: Optional[str] = None
    decision_at: Optional[datetime] = None
    authorization_reference: Optional[str] = None
    audit_event_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ApprovalQueue(BaseModel):
    policy_version: str
    contract_id: str
    tasks: list[ReviewTask] = Field(default_factory=list)
    routes: list[ApprovalRoute] = Field(default_factory=list)
    audit: list[AuditEvent] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    def open_tasks(self) -> list[ReviewTask]:
        closed = {"approved", "rejected", "waived", "cancelled"}
        return [t for t in self.tasks if t.status not in closed]

    def sorted_queue(self) -> list[ReviewTask]:
        rank = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            self.open_tasks(),
            key=lambda t: (rank.get(t.severity, 9), t.due_at or datetime.max.replace(tzinfo=timezone.utc), t.task_id),
        )


class UnauthorizedDecisionError(PermissionError):
    """Raised when the caller's role is not authorized for the task's teams."""


class DecisionRejectedError(ValueError):
    """Raised when a decision is malformed (e.g. waive without auth ref)."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_id(*parts: str, prefix: str = "") -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{h}" if prefix else h


def _evidence_ids(finding: ScoredFinding) -> list[str]:
    ids: list[str] = []
    for i, ev in enumerate(finding.grounded.finding.evidence):
        ids.append(
            _stable_id(
                finding.grounded.finding.rule_id,
                str(ev.page),
                ev.quote[:80],
                prefix=f"ev{i}-",
            )
        )
    return ids


def _has_missing_evidence(finding: ScoredFinding) -> bool:
    f = finding.grounded
    if f.grounding in ("failed", "missing", "ambiguous"):
        return True
    if finding.status in ("unclear", "not_found") and not f.finding.evidence:
        return True
    return False


def _rule_matches(
    rule: PolicyRule,
    *,
    finding: Optional[ScoredFinding],
    security: SecurityFlags,
) -> bool:
    m = rule.match
    if m.security_flag:
        return security.any()
    if finding is None:
        return False
    rid = finding.grounded.finding.rule_id
    if m.rule_ids and "*" not in m.rule_ids and rid not in m.rule_ids:
        return False
    if m.statuses and finding.status not in m.statuses:
        return False
    if m.missing_evidence and not _has_missing_evidence(finding):
        return False
    return True


def _format_reason(template: str, finding: Optional[ScoredFinding], fallback: str) -> str:
    if not template:
        return fallback
    if finding is None:
        return template
    return template.format(
        rule_id=finding.grounded.finding.rule_id,
        status=finding.status,
    )


def _role_authorized(policy: ApprovalPolicy, role: ActorRole, teams: list[Team]) -> bool:
    if role == "admin":
        return True
    for team in teams:
        allowed = policy.team_roles.get(team, [team])
        if role in allowed:
            return True
    return False


_DECISION_TO_STATUS: dict[DecisionKind, TaskStatus] = {
    "approve": "approved",
    "reject": "rejected",
    "request_changes": "changes_requested",
    "need_information": "needs_information",
    "waive_with_authorization": "waived",
}


# --------------------------------------------------------------------------- #
# Queue builder
# --------------------------------------------------------------------------- #
def build_queue(
    pkg: ReviewPackage,
    *,
    contract_id: Optional[str] = None,
    policy: Optional[ApprovalPolicy] = None,
    related_obligation_ids: Optional[list[str]] = None,
    now: Optional[datetime] = None,
) -> ApprovalQueue:
    """Create approval tasks/routes from a ReviewPackage. Idempotent IDs."""
    policy = policy or load_policy()
    ts = now or _now()
    cid = contract_id or (pkg.file_name or pkg.run_id or "unknown").replace(".pdf", "")
    obl_ids = list(related_obligation_ids or [])

    tasks: list[ReviewTask] = []
    routes: list[ApprovalRoute] = []
    audit: list[AuditEvent] = []

    # Security-flag-only rules (no finding)
    for rule in policy.sorted_rules():
        if not rule.match.security_flag:
            continue
        if not _rule_matches(rule, finding=None, security=pkg.security):
            continue
        task_id = _stable_id(cid, "security", rule.id, prefix="task-")
        task, new_routes, evt = _make_task(
            task_id=task_id,
            contract_id=cid,
            finding_ids=["security_flag"],
            obligation_ids=obl_ids,
            rule=rule,
            finding=None,
            evidence_ids=[],
            ts=ts,
        )
        if not any(t.task_id == task_id for t in tasks):
            tasks.append(task)
            routes.extend(new_routes)
            audit.append(evt)

    # Finding-driven rules — preserve all matches (multi-team OK)
    for finding in pkg.findings:
        # Preferred / fallback preferred terms do not open approval tasks
        if finding.status in ("preferred", "fallback") and not _has_missing_evidence(finding):
            continue
        matched = [
            r
            for r in policy.sorted_rules()
            if not r.match.security_flag
            and _rule_matches(r, finding=finding, security=pkg.security)
        ]
        if not matched:
            continue

        rid = finding.grounded.finding.rule_id
        task_id = _stable_id(cid, rid, finding.status, prefix="task-")
        # Merge multi-rule matches into one task with union of teams
        teams: list[Team] = []
        policy_ids: list[str] = []
        reason_codes: list[str] = []
        required_inputs: list[str] = []
        severities: list[Severity] = []
        evidence = _evidence_ids(finding)
        reasons: list[str] = []
        initial: TaskStatus = "open"

        for rule in matched:
            for t in rule.teams:
                if t not in teams:
                    teams.append(t)
            policy_ids.append(rule.id)
            reason_codes.append(rule.reason_code)
            for inp in rule.required_inputs:
                if inp not in required_inputs:
                    required_inputs.append(inp)
            severities.append(rule.severity)
            reasons.append(
                _format_reason(rule.reason_template, finding, rule.reason_code)
            )
            if rule.initial_status == "needs_information":
                initial = "needs_information"

        sev: Severity = "high" if "high" in severities else (
            "medium" if "medium" in severities else "low"
        )
        title = reasons[0] if reasons else f"Review {rid}"
        task = ReviewTask(
            task_id=task_id,
            contract_id=cid,
            related_finding_ids=[rid],
            related_obligation_ids=obl_ids,
            required_teams=teams,
            status=initial,
            severity=sev,
            title=title,
            reason_codes=reason_codes,
            matched_policy_rule_ids=policy_ids,
            source_evidence_ids=evidence,
            created_at=ts,
            updated_at=ts,
        )
        tasks.append(task)

        for rule in matched:
            for team in rule.teams:
                route_id = _stable_id(task_id, rule.id, team, prefix="route-")
                routes.append(
                    ApprovalRoute(
                        route_id=route_id,
                        review_task_id=task_id,
                        team=team,
                        reason_code=rule.reason_code,
                        reason=_format_reason(
                            rule.reason_template, finding, rule.reason_code
                        ),
                        matched_policy_rule_ids=[rule.id],
                        severity=rule.severity,
                        source_evidence_ids=evidence,
                        status=initial,
                        required_inputs=list(rule.required_inputs),
                        created_at=ts,
                        updated_at=ts,
                    )
                )

        evt = AuditEvent(
            event_id=_stable_id(task_id, "created", ts.isoformat(), prefix="aud-"),
            timestamp=ts,
            actor="system",
            actor_role="admin",
            event_type="task_created",
            task_id=task_id,
            detail=title,
            after_status=initial,
        )
        task.audit_event_ids.append(evt.event_id)
        audit.append(evt)

    queue = ApprovalQueue(
        policy_version=policy.version,
        contract_id=cid,
        tasks=tasks,
        routes=routes,
        audit=audit,
        created_at=ts,
        updated_at=ts,
    )
    return attach_knowledge_to_queue(queue)


def attach_knowledge_to_queue(
    queue: ApprovalQueue,
    *,
    retriever=None,
) -> ApprovalQueue:
    """RAG Phase 3 — attach approved policy citations to each open approval task."""
    from app.rag import retrieve_for_approval_reason

    enriched: list[ReviewTask] = []
    for task in queue.tasks:
        reason = task.reason_codes[0] if task.reason_codes else "exception"
        try:
            hits = retrieve_for_approval_reason(
                reason, task.title, retriever=retriever, top_k=3
            )
        except Exception:
            hits = []
        if not hits:
            enriched.append(
                task.model_copy(update={"knowledge_abstained": True, "knowledge_citations": []})
            )
            continue
        cites = [
            KnowledgeCitation(
                evidence_id=e.evidence_id,
                chunk_id=e.chunk_id,
                title=e.title,
                excerpt=e.excerpt,
                version=e.version,
                status=e.status,
                document_type=e.document_type,
                relevance_score=e.relevance_score,
            )
            for e in hits
        ]
        enriched.append(
            task.model_copy(
                update={
                    "knowledge_citations": cites,
                    "knowledge_abstained": False,
                }
            )
        )
    return queue.model_copy(update={"tasks": enriched, "updated_at": _now()})


def _make_task(
    *,
    task_id: str,
    contract_id: str,
    finding_ids: list[str],
    obligation_ids: list[str],
    rule: PolicyRule,
    finding: Optional[ScoredFinding],
    evidence_ids: list[str],
    ts: datetime,
) -> tuple[ReviewTask, list[ApprovalRoute], AuditEvent]:
    reason = _format_reason(rule.reason_template, finding, rule.reason_code)
    status: TaskStatus = rule.initial_status
    task = ReviewTask(
        task_id=task_id,
        contract_id=contract_id,
        related_finding_ids=finding_ids,
        related_obligation_ids=obligation_ids,
        required_teams=list(rule.teams),
        status=status,
        severity=rule.severity,
        title=reason,
        reason_codes=[rule.reason_code],
        matched_policy_rule_ids=[rule.id],
        source_evidence_ids=evidence_ids,
        created_at=ts,
        updated_at=ts,
    )
    routes = [
        ApprovalRoute(
            route_id=_stable_id(task_id, rule.id, team, prefix="route-"),
            review_task_id=task_id,
            team=team,
            reason_code=rule.reason_code,
            reason=reason,
            matched_policy_rule_ids=[rule.id],
            severity=rule.severity,
            source_evidence_ids=evidence_ids,
            status=status,
            required_inputs=list(rule.required_inputs),
            created_at=ts,
            updated_at=ts,
        )
        for team in rule.teams
    ]
    evt = AuditEvent(
        event_id=_stable_id(task_id, "created", ts.isoformat(), prefix="aud-"),
        timestamp=ts,
        actor="system",
        actor_role="admin",
        event_type="task_created",
        task_id=task_id,
        detail=reason,
        after_status=status,
    )
    task.audit_event_ids.append(evt.event_id)
    return task, routes, evt


# --------------------------------------------------------------------------- #
# Decisions (service layer — authorization enforced here)
# --------------------------------------------------------------------------- #
def record_decision(
    queue: ApprovalQueue,
    task_id: str,
    *,
    reviewer: str,
    role: ActorRole,
    decision: DecisionKind,
    reason: str,
    authorization_reference: Optional[str] = None,
    policy: Optional[ApprovalPolicy] = None,
    now: Optional[datetime] = None,
) -> ApprovalQueue:
    """Apply a human decision. Mutates a deep copy; never called by an LLM path."""
    policy = policy or load_policy()
    ts = now or _now()
    task = next((t for t in queue.tasks if t.task_id == task_id), None)
    if task is None:
        raise KeyError(f"unknown task_id: {task_id}")

    if not _role_authorized(policy, role, task.required_teams):
        raise UnauthorizedDecisionError(
            f"role '{role}' is not authorized for teams {task.required_teams}"
        )

    if not reason or not reason.strip():
        raise DecisionRejectedError("decision reason is required")

    if decision == "waive_with_authorization":
        if not authorization_reference or not authorization_reference.strip():
            raise DecisionRejectedError(
                "waive_with_authorization requires an authorization_reference"
            )

    before = task.status
    after = _DECISION_TO_STATUS[decision]
    human = HumanDecision(
        reviewer=reviewer,
        role=role,
        decision=decision,
        reason=reason.strip(),
        timestamp=ts,
        authorization_reference=authorization_reference,
    )

    task.status = after
    task.decision = human.decision
    task.decision_reason = human.reason
    task.decided_by = f"{human.reviewer}|{human.role}"
    task.decision_at = ts
    task.authorization_reference = human.authorization_reference
    task.updated_at = ts

    for route in queue.routes:
        if route.review_task_id == task_id:
            route.status = after
            route.updated_at = ts
            route.assigned_to = reviewer

    evt = AuditEvent(
        event_id=_stable_id(task_id, decision, ts.isoformat(), reviewer, prefix="aud-"),
        timestamp=ts,
        actor=reviewer,
        actor_role=role,
        event_type="decision_recorded",
        task_id=task_id,
        detail=f"{decision}: {reason.strip()}",
        before_status=before,
        after_status=after,
    )
    task.audit_event_ids.append(evt.event_id)
    queue.audit.append(evt)
    queue.updated_at = ts
    return queue


def merge_queues(old: ApprovalQueue, new: ApprovalQueue) -> ApprovalQueue:
    """Preserve human decisions on re-run; refresh open system-created tasks."""
    decided = {
        t.task_id: t
        for t in old.tasks
        if t.decision is not None
    }
    merged_tasks: list[ReviewTask] = []
    seen: set[str] = set()
    for t in new.tasks:
        if t.task_id in decided:
            merged_tasks.append(decided[t.task_id])
        else:
            merged_tasks.append(t)
        seen.add(t.task_id)
    for tid, t in decided.items():
        if tid not in seen:
            merged_tasks.append(t)

    # Keep routes for decided tasks from old; otherwise use new
    decided_ids = set(decided)
    routes = [r for r in new.routes if r.review_task_id not in decided_ids]
    routes.extend(r for r in old.routes if r.review_task_id in decided_ids)

    return ApprovalQueue(
        policy_version=new.policy_version,
        contract_id=new.contract_id,
        tasks=merged_tasks,
        routes=routes,
        audit=list(old.audit) + [
            a for a in new.audit if a.event_type == "task_created"
            and a.task_id not in decided_ids
        ],
        created_at=old.created_at,
        updated_at=_now(),
    )


def queue_to_markdown(queue: ApprovalQueue) -> str:
    lines = [
        f"# Approval queue — {queue.contract_id}",
        f"Policy: `{queue.policy_version}`",
        f"Tasks: {len(queue.tasks)} · Open: {len(queue.open_tasks())}",
        "",
    ]
    for t in queue.sorted_queue() or queue.tasks:
        lines.append(f"## {t.task_id} — {t.title}")
        lines.append(f"- status: `{t.status}` · severity: `{t.severity}`")
        lines.append(f"- teams: {', '.join(t.required_teams)}")
        lines.append(f"- findings: {', '.join(t.related_finding_ids)}")
        lines.append(f"- policy rules: {', '.join(t.matched_policy_rule_ids)}")
        if t.decision:
            lines.append(
                f"- decision: `{t.decision}` by {t.decided_by} — {t.decision_reason}"
            )
        lines.append("")
    lines.append("## Audit")
    for a in queue.audit:
        lines.append(
            f"- `{a.timestamp.isoformat()}` [{a.event_type}] "
            f"{a.actor}/{a.actor_role}: {a.detail}"
        )
    return "\n".join(lines) + "\n"
