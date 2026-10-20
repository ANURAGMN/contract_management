"""Regression: Contract B's uncapped general indemnity (clause 12) sits outside
the 1x liability cap. Once the clause reviewer resolves the cross-reference, the
rule engine must score LIABILITY-002 as walk_away and the router must send B to
Legal. This locks in the documented failure-and-fix.
"""

from app import steps
from app.schemas import (
    ClauseFinding,
    Evidence,
    GroundedFinding,
    GroundedResult,
    Playbook,
    RouteInput,
    RulesInput,
    Preflight,
    SecurityFlags,
)


def _playbook():
    return steps.load_playbook("demo-v1")


def test_indemnity_outside_cap_routes_to_legal():
    pb = _playbook()

    indemnity = GroundedFinding(
        finding=ClauseFinding(
            rule_id="LIABILITY-002",
            found=True,
            extracted_value="uncapped general indemnity (clause 12 outside the cap)",
            evidence=[
                Evidence(
                    quote="The Supplier's obligations under this Clause 12 shall not be subject to any limitation of liability",
                    page=4,
                )
            ],
            cross_references=["clause 5"],
        ),
        grounding="exact",
    )

    grounded = GroundedResult(meta=_empty_meta(), findings=[indemnity], dropped=[])

    scored = steps.apply_rules(RulesInput(grounded=grounded, playbook=pb, pages=[]))
    liab = [f for f in scored.findings if f.grounded.finding.rule_id == "LIABILITY-002"][0]
    assert liab.status == "walk_away"
    assert liab.severity == "high"

    route = steps.route(
        RouteInput(scored=scored, preflight=Preflight(unreliable_extraction=False), flags=SecurityFlags())
    )
    assert route.route == "legal"


def _empty_meta():
    from app.schemas import ContractMetadata, Field_

    return ContractMetadata(**{k: Field_() for k in ContractMetadata.model_fields})
