"""RAG Phase 1 — ingestion, lexical retrieval, filters, citations, abstention."""

from __future__ import annotations

from app.rag import (
    ABSTAIN_MESSAGE,
    RetrievalFilter,
    ask_knowledge_base,
    build_default_retriever,
    compare_clause_to_knowledge,
    ingest_corpus,
    validate_citations,
)


def test_ingest_stable_ids_and_metadata():
    chunks = ingest_corpus()
    assert len(chunks) >= 10
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    again = ingest_corpus()
    assert {c.chunk_id for c in chunks} == {c.chunk_id for c in again}
    renewal = [c for c in chunks if c.document_id == "pol-renewal-v1"]
    assert renewal
    assert all(c.status == "approved" for c in renewal)
    assert all(c.version == "1.0" for c in renewal)


def test_corpus_status():
    _, status, docs = build_default_retriever()
    assert status.chunk_count > 0
    assert status.approved_policy_count >= 5
    assert "Synthetic" in status.label or "demo" in status.label.lower()
    assert any(d.status == "draft" for d in docs)
    assert any(d.status == "retired" for d in docs)


def test_renewal_query_ranks_renewal_policy():
    retriever, _, _ = build_default_retriever()
    ev, audit = retriever.retrieve(
        "auto-renewal notice period approved renewal policy",
        top_k=5,
        filters=RetrievalFilter(statuses=["approved"], exclude_precedent_from_normative=True),
    )
    assert ev
    assert audit.mode == "lexical"
    top_types = [e.document_type for e in ev[:3]]
    assert "renewal_and_exit_policy" in top_types or "approved_clauses" in top_types
    # Payment terms should not be the top hit
    assert ev[0].document_type != "payment_terms_policy"


def test_uptime_query_preserves_99_9():
    retriever, _, _ = build_default_retriever()
    ev, _ = retriever.retrieve(
        "minimum uptime 99.9% SLA service credits",
        top_k=5,
        filters=RetrievalFilter(
            statuses=["approved"],
            document_types=["sla_policy", "approved_clauses"],
            exclude_precedent_from_normative=True,
        ),
    )
    assert ev
    joined = " ".join(e.excerpt for e in ev)
    assert "99.9%" in joined
    assert ev[0].document_type in ("sla_policy", "approved_clauses")


def test_normative_excludes_draft_and_retired():
    retriever, _, _ = build_default_retriever()
    ev, _ = retriever.retrieve(
        "auto-renewal notice period preferred days",
        top_k=8,
        filters=RetrievalFilter(statuses=["approved"], exclude_precedent_from_normative=True),
    )
    assert ev
    assert all(e.status == "approved" for e in ev)
    # Draft prefers 45 days — must not appear
    assert not any("45 days" in e.excerpt and "draft" in (e.version or "").lower() for e in ev)
    assert not any(e.document_id == "pol-renewal-draft" for e in ev)
    assert not any(e.document_id == "pol-sla-retired" for e in ev)


def test_retired_only_when_status_filter_allows():
    from datetime import date

    retriever, _, _ = build_default_retriever()
    ev, _ = retriever.retrieve(
        "minimum uptime 95.0% retired",
        top_k=5,
        filters=RetrievalFilter(
            statuses=["retired"],
            exclude_precedent_from_normative=True,
            as_of=date(2025, 6, 1),  # within retired doc's effective window
        ),
    )
    assert any(e.document_id == "pol-sla-retired" for e in ev)


def test_historical_is_precedent_not_normative():
    retriever, _, _ = build_default_retriever()
    ev_norm, _ = retriever.retrieve(
        "uptime 99.5% TechnoNova",
        top_k=5,
        filters=RetrievalFilter(statuses=["approved"], exclude_precedent_from_normative=True),
    )
    assert all(not e.is_precedent for e in ev_norm)

    ev_hist, _ = retriever.retrieve(
        "uptime 99.5% TechnoNova historical",
        top_k=5,
        filters=RetrievalFilter(
            statuses=["approved"],
            document_types=["historical_contract"],
            exclude_precedent_from_normative=False,
            include_precedent=True,
        ),
    )
    assert ev_hist
    assert all(e.is_precedent for e in ev_hist)


def test_citation_validation_drops_unknown():
    retriever, _, _ = build_default_retriever()
    ev, _ = retriever.retrieve("liability cap indemnity", top_k=3)
    assert ev
    valid, invalid = validate_citations(
        [ev[0].evidence_id, ev[0].chunk_id, "invented-citation-xyz"],
        ev,
    )
    assert len(valid) >= 1
    assert "invented-citation-xyz" in invalid
    assert all(v.chunk_id in {e.chunk_id for e in ev} for v in valid)


def test_ask_abstains_when_no_evidence():
    retriever, _, _ = build_default_retriever()
    result = ask_knowledge_base(
        "What is our policy for lunar mineral royalties?",
        retriever,
        min_top_score=5.0,
    )
    assert result.abstained
    assert ABSTAIN_MESSAGE in result.answer
    assert result.evidence == []


def test_ask_auto_renewal_with_citation():
    retriever, _, _ = build_default_retriever()
    result = ask_knowledge_base(
        "What does our policy require for auto-renewal notice?",
        retriever,
    )
    assert not result.abstained
    assert result.evidence
    assert "60" in result.answer or any("60" in e.excerpt for e in result.evidence)
    assert all(e.status == "approved" for e in result.evidence)


def test_compare_renewal_90_day_deviation():
    retriever, _, _ = build_default_retriever()
    cmp_ = compare_clause_to_knowledge(
        "renewal",
        contract_quote="either party must give ninety (90) days' written notice of non-renewal",
        contract_page=2,
        rule_id="RENEWAL-001",
        retriever=retriever,
    )
    assert not cmp_.abstained
    assert cmp_.knowledge_evidence
    assert "60" in cmp_.assessment or "deviation" in cmp_.assessment.lower()
    assert cmp_.contract_page == 2


def test_compare_sla_99_5_below_minimum():
    retriever, _, _ = build_default_retriever()
    cmp_ = compare_clause_to_knowledge(
        "sla",
        contract_quote="Supplier commits to 99.5% platform availability per month",
        contract_page=3,
        rule_id="SLA-001",
        retriever=retriever,
    )
    assert not cmp_.abstained
    assert "99.9%" in cmp_.assessment or "99.9" in cmp_.assessment
    assert "99.5" in cmp_.assessment
    assert any("99.9%" in e.excerpt for e in cmp_.knowledge_evidence)


def test_retrieve_for_negotiation_returns_excerpts():
    from app.rag import retrieve_for_negotiation

    evidence, excerpts, audit = retrieve_for_negotiation()
    assert evidence
    assert excerpts
    assert all(e.status == "approved" for e in evidence)
    assert audit.mode == "lexical_negotiation"
    assert not any(e.is_precedent for e in evidence)


def test_retrieve_for_approval_reason_sla():
    from app.rag import retrieve_for_approval_reason

    hits = retrieve_for_approval_reason(
        "operational_sla",
        "SLA below preferred uptime",
    )
    assert hits
    assert all(h.status == "approved" for h in hits)
    joined = " ".join(h.excerpt for h in hits)
    assert "99.9" in joined or "uptime" in joined.lower() or "SLA" in joined


def test_ground_review_package_persists_comparisons():
    from app.demo_packages import build_contract_b_package
    from app.steps import attach_knowledge_groundings

    pkg = attach_knowledge_groundings(build_contract_b_package())
    assert pkg.knowledge_groundings
    intents = {g.get("intent") for g in pkg.knowledge_groundings}
    assert intents & {"renewal", "sla", "liability", "payment", "data", "termination"}
    assert pkg.stage_status.get("knowledge") == "ok"
