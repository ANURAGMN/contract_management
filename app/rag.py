"""RAG Phase 1 — local knowledge baseline (lexical retrieval).

Synthetic demo knowledge pack + deterministic lexical/BM25-style search.
Embeddings / vector stores can plug in later behind the same interfaces.
No LLM is used for retrieval or for inventing citation IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = ROOT / "data" / "knowledge"
MANIFEST_PATH = KNOWLEDGE_ROOT / "manifest.json"

DEMO_LABEL = "Synthetic demo data — not real company policy or legal advice"

DocumentType = Literal[
    "procurement_policy",
    "security_privacy_standard",
    "sla_policy",
    "payment_terms_policy",
    "renewal_and_exit_policy",
    "approved_clauses",
    "negotiation_playbook",
    "negotiation_outcomes",
    "historical_contract",
]
DocStatus = Literal["draft", "approved", "retired"]

# Intent → preferred document types + query hints (MVP: renewal + SLA first)
CLAUSE_INTENTS: dict[str, dict] = {
    "renewal": {
        "query": "auto-renewal notice period preferred 60 days renewal policy",
        "document_types": ["renewal_and_exit_policy", "approved_clauses", "negotiation_playbook"],
        "rule_ids": ["RENEWAL-001"],
    },
    "sla": {
        "query": "minimum uptime 99.9% SLA service credits platform availability",
        "document_types": ["sla_policy", "approved_clauses", "negotiation_playbook"],
        "rule_ids": ["SLA-001"],
    },
    "liability": {
        "query": "liability cap indemnity limitation of liability",
        "document_types": ["approved_clauses", "negotiation_playbook"],
        "rule_ids": ["LIABILITY-001", "LIABILITY-002"],
    },
    "payment": {
        "query": "payment terms Net 45 Net 30 invoicing",
        "document_types": ["payment_terms_policy", "procurement_policy"],
        "rule_ids": ["PAYMENT-001"],
    },
    "data": {
        "query": "data processing agreement DPA personal data privacy",
        "document_types": ["security_privacy_standard", "approved_clauses"],
        "rule_ids": ["DATA-001"],
    },
    "termination": {
        "query": "termination for convenience notice period",
        "document_types": ["approved_clauses", "renewal_and_exit_policy"],
        "rule_ids": ["TERMINATION-001"],
    },
}

RULE_TO_INTENT = {
    rid: intent
    for intent, conf in CLAUSE_INTENTS.items()
    for rid in conf["rule_ids"]
}

ABSTAIN_MESSAGE = (
    "No supporting approved policy found in the demo knowledge base"
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class DocumentMeta(BaseModel):
    document_id: str
    title: str
    document_type: str
    owner: str = ""
    version: str = "1.0"
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    status: DocStatus = "approved"
    jurisdiction: Optional[str] = None
    confidentiality: str = "internal"
    source_path: str


class KnowledgeChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    document_type: str
    text: str
    heading_path: list[str] = Field(default_factory=list)
    page_number: Optional[int] = None
    source_path: str
    version: Optional[str] = None
    status: Optional[str] = None
    effective_from: Optional[date] = None
    effective_to: Optional[date] = None
    jurisdiction: Optional[str] = None
    confidentiality: Optional[str] = None
    content_hash: str = ""
    is_precedent: bool = False  # historical contracts


class RetrievedEvidence(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    title: str
    excerpt: str
    relevance_score: Optional[float] = None
    page_number: Optional[int] = None
    heading_path: list[str] = Field(default_factory=list)
    version: Optional[str] = None
    status: Optional[str] = None
    source_path: str
    document_type: str = ""
    is_precedent: bool = False


class RetrievalFilter(BaseModel):
    statuses: list[str] = Field(default_factory=lambda: ["approved"])
    document_types: Optional[list[str]] = None
    exclude_precedent_from_normative: bool = True
    as_of: Optional[date] = None
    jurisdiction: Optional[str] = None
    include_precedent: bool = False  # when True, allow historical labelled as precedent


class RetrievalAudit(BaseModel):
    query: str
    filters: dict
    returned_chunk_ids: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    corpus_id: str = ""
    mode: str = "lexical"


class CorpusStatus(BaseModel):
    corpus_id: str
    label: str
    document_count: int
    approved_policy_count: int
    chunk_count: int
    last_indexed_at: datetime
    mode: str = "lexical"


class AskResult(BaseModel):
    question: str
    abstained: bool
    answer: str
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    audit: RetrievalAudit


class ClauseKnowledgeComparison(BaseModel):
    """Side-by-side contract quote vs retrieved policy for one clause intent."""

    intent: str
    rule_id: Optional[str] = None
    contract_quote: Optional[str] = None
    contract_page: Optional[int] = None
    assessment: str
    action: str
    knowledge_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    abstained: bool = False
    audit: Optional[RetrievalAudit] = None


# --------------------------------------------------------------------------- #
# Interfaces (swap backends later)
# --------------------------------------------------------------------------- #
class KnowledgeIndex(Protocol):
    def all_chunks(self) -> list[KnowledgeChunk]: ...

    def get(self, chunk_id: str) -> Optional[KnowledgeChunk]: ...


class KnowledgeRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
    ) -> tuple[list[RetrievedEvidence], RetrievalAudit]: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?|[₹$]?\d[\d,]*(?:\.\d+)?", re.I)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.M)


def _stable_chunk_id(document_id: str, heading_path: list[str], text: str) -> str:
    raw = "|".join([document_id, "/".join(heading_path), text.strip()[:200]])
    return "chk-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


# --------------------------------------------------------------------------- #
# Ingestion + chunking
# --------------------------------------------------------------------------- #
def load_manifest(path: Path | None = None) -> tuple[str, str, list[DocumentMeta]]:
    p = path or MANIFEST_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    docs = []
    for d in raw["documents"]:
        docs.append(
            DocumentMeta(
                document_id=d["document_id"],
                title=d["title"],
                document_type=d["document_type"],
                owner=d.get("owner", ""),
                version=d.get("version", "1.0"),
                effective_from=_parse_date(d.get("effective_from")),
                effective_to=_parse_date(d.get("effective_to")),
                status=d.get("status", "approved"),
                jurisdiction=d.get("jurisdiction"),
                confidentiality=d.get("confidentiality", "internal"),
                source_path=d["source_path"],
            )
        )
    return raw["corpus_id"], raw.get("label", DEMO_LABEL), docs


def _chunk_markdown(text: str, meta: DocumentMeta, root: Path) -> list[KnowledgeChunk]:
    # Split on ## / ### headings; keep H1 as document title context
    parts: list[tuple[list[str], str]] = []
    current_h: list[str] = [meta.title]
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            parts.append((list(current_h), body))

    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.+)$", line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                current_h = [title]
            elif level == 2:
                current_h = [current_h[0] if current_h else meta.title, title]
            else:
                base = current_h[:2] if len(current_h) >= 2 else (
                    [current_h[0], title] if current_h else [meta.title, title]
                )
                current_h = base[:2] + [title]
            buf = []
        else:
            buf.append(line)
    flush()

    if not parts and text.strip():
        parts = [([meta.title], text.strip())]

    is_precedent = meta.document_type == "historical_contract"
    chunks: list[KnowledgeChunk] = []
    for heading_path, body in parts:
        if len(body) < 20:
            continue
        cid = _stable_chunk_id(meta.document_id, heading_path, body)
        chunks.append(
            KnowledgeChunk(
                chunk_id=cid,
                document_id=meta.document_id,
                title=meta.title,
                document_type=meta.document_type,
                text=body,
                heading_path=heading_path,
                source_path=str((root / meta.source_path).as_posix()),
                version=meta.version,
                status=meta.status,
                effective_from=meta.effective_from,
                effective_to=meta.effective_to,
                jurisdiction=meta.jurisdiction,
                confidentiality=meta.confidentiality,
                content_hash=_content_hash(body),
                is_precedent=is_precedent,
            )
        )
    return chunks


def _chunk_csv(text: str, meta: DocumentMeta, root: Path) -> list[KnowledgeChunk]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0]
    rows = lines[1:] or [header]
    body = "Negotiation outcomes (synthetic):\n" + header + "\n" + "\n".join(rows)
    cid = _stable_chunk_id(meta.document_id, [meta.title, "outcomes"], body)
    return [
        KnowledgeChunk(
            chunk_id=cid,
            document_id=meta.document_id,
            title=meta.title,
            document_type=meta.document_type,
            text=body,
            heading_path=[meta.title, "outcomes"],
            source_path=str((root / meta.source_path).as_posix()),
            version=meta.version,
            status=meta.status,
            effective_from=meta.effective_from,
            effective_to=meta.effective_to,
            jurisdiction=meta.jurisdiction,
            confidentiality=meta.confidentiality,
            content_hash=_content_hash(body),
            is_precedent=False,
        )
    ]


def ingest_corpus(knowledge_root: Path | None = None) -> list[KnowledgeChunk]:
    root = knowledge_root or KNOWLEDGE_ROOT
    _, _, docs = load_manifest(root / "manifest.json")
    chunks: list[KnowledgeChunk] = []
    seen_ids: set[str] = set()
    for meta in docs:
        path = root / meta.source_path
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".csv":
            produced = _chunk_csv(raw, meta, root)
        else:
            produced = _chunk_markdown(raw, meta, root)
        for ch in produced:
            if ch.chunk_id in seen_ids:
                continue
            seen_ids.add(ch.chunk_id)
            chunks.append(ch)
    return chunks


# --------------------------------------------------------------------------- #
# In-memory lexical index
# --------------------------------------------------------------------------- #
class InMemoryKnowledgeIndex:
    def __init__(self, chunks: list[KnowledgeChunk]):
        self._chunks = {c.chunk_id: c for c in chunks}
        self._order = [c.chunk_id for c in chunks]

    def all_chunks(self) -> list[KnowledgeChunk]:
        return [self._chunks[i] for i in self._order]

    def get(self, chunk_id: str) -> Optional[KnowledgeChunk]:
        return self._chunks.get(chunk_id)


def _passes_filters(chunk: KnowledgeChunk, filters: RetrievalFilter) -> bool:
    if filters.statuses and (chunk.status or "") not in filters.statuses:
        return False
    if filters.document_types and chunk.document_type not in filters.document_types:
        return False
    if filters.exclude_precedent_from_normative and chunk.is_precedent:
        if not filters.include_precedent:
            return False
    if filters.jurisdiction and chunk.jurisdiction and chunk.jurisdiction != filters.jurisdiction:
        return False
    as_of = filters.as_of or date.today()
    if chunk.effective_from and chunk.effective_from > as_of:
        return False
    if chunk.effective_to and chunk.effective_to < as_of:
        return False
    return True


def _lexical_score(query: str, text: str) -> float:
    """Simple TF overlap + exact-phrase boosts for legal/numeric terms."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokenize(text)
    if not t_tokens:
        return 0.0
    tf: dict[str, int] = {}
    for t in t_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for qt in set(q_tokens):
        if qt in tf:
            score += 1.0 + math.log(1 + tf[qt])
    # Exact phrase / numeric boosts
    q_lower = query.lower()
    t_lower = text.lower()
    for phrase in (
        "99.9%",
        "99.5%",
        "60 days",
        "90 days",
        "30 days",
        "net 45",
        "net 30",
        "net 15",
        "auto-renewal",
        "service credits",
        "data processing agreement",
        "liability",
    ):
        if phrase in q_lower and phrase in t_lower:
            score += 3.0
        elif phrase.replace(" ", "") in q_lower.replace(" ", "") and phrase in t_lower:
            score += 2.0
    # Length normalization (mild)
    score = score / (1.0 + math.log(1 + len(t_tokens) / 40.0))
    return score


class LexicalKnowledgeRetriever:
    """Deterministic lexical retriever — no network, no embeddings."""

    def __init__(
        self,
        index: InMemoryKnowledgeIndex,
        *,
        corpus_id: str = "demo-knowledge-v1",
        min_score: float = 1.5,
    ):
        self.index = index
        self.corpus_id = corpus_id
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Optional[RetrievalFilter] = None,
    ) -> tuple[list[RetrievedEvidence], RetrievalAudit]:
        filters = filters or RetrievalFilter()
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self.index.all_chunks():
            if not _passes_filters(chunk, filters):
                continue
            s = _lexical_score(query, " ".join(chunk.heading_path) + "\n" + chunk.text)
            if s >= self.min_score:
                scored.append((s, chunk))
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        top = scored[:top_k]

        evidence: list[RetrievedEvidence] = []
        scores: dict[str, float] = {}
        for s, ch in top:
            eid = "ev-" + hashlib.sha1(f"{ch.chunk_id}|{query}".encode()).hexdigest()[:10]
            excerpt = ch.text.strip()
            if len(excerpt) > 420:
                excerpt = excerpt[:417] + "..."
            evidence.append(
                RetrievedEvidence(
                    evidence_id=eid,
                    chunk_id=ch.chunk_id,
                    document_id=ch.document_id,
                    title=ch.title,
                    excerpt=excerpt,
                    relevance_score=round(s, 3),
                    page_number=ch.page_number,
                    heading_path=list(ch.heading_path),
                    version=ch.version,
                    status=ch.status,
                    source_path=ch.source_path,
                    document_type=ch.document_type,
                    is_precedent=ch.is_precedent,
                )
            )
            scores[ch.chunk_id] = round(s, 3)

        audit = RetrievalAudit(
            query=query,
            filters=filters.model_dump(mode="json"),
            returned_chunk_ids=[e.chunk_id for e in evidence],
            scores=scores,
            selected_evidence_ids=[e.evidence_id for e in evidence],
            corpus_id=self.corpus_id,
            mode="lexical",
        )
        return evidence, audit


# --------------------------------------------------------------------------- #
# Citation validation + ask + clause comparison
# --------------------------------------------------------------------------- #
def validate_citations(
    cited_ids: list[str],
    evidence: list[RetrievedEvidence],
) -> tuple[list[RetrievedEvidence], list[str]]:
    """Keep only citations that resolve to retrieved evidence. Drop unknowns."""
    by_ev = {e.evidence_id: e for e in evidence}
    by_chunk = {e.chunk_id: e for e in evidence}
    valid: list[RetrievedEvidence] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for cid in cited_ids:
        hit = by_ev.get(cid) or by_chunk.get(cid)
        if hit is None:
            invalid.append(cid)
            continue
        key = hit.evidence_id
        if key in seen:
            continue
        seen.add(key)
        valid.append(hit)
    return valid, invalid


def build_default_retriever(
    knowledge_root: Path | None = None,
) -> tuple[LexicalKnowledgeRetriever, CorpusStatus, list[DocumentMeta]]:
    root = knowledge_root or KNOWLEDGE_ROOT
    corpus_id, label, docs = load_manifest(root / "manifest.json")
    chunks = ingest_corpus(root)
    index = InMemoryKnowledgeIndex(chunks)
    retriever = LexicalKnowledgeRetriever(index, corpus_id=corpus_id)
    approved = sum(
        1
        for d in docs
        if d.status == "approved" and d.document_type != "historical_contract"
    )
    status = CorpusStatus(
        corpus_id=corpus_id,
        label=label,
        document_count=len(docs),
        approved_policy_count=approved,
        chunk_count=len(chunks),
        last_indexed_at=datetime.now(timezone.utc),
        mode="lexical",
    )
    return retriever, status, docs


def ask_knowledge_base(
    question: str,
    retriever: LexicalKnowledgeRetriever,
    *,
    top_k: int = 4,
    filters: Optional[RetrievalFilter] = None,
    min_top_score: float = 2.0,
) -> AskResult:
    """Extractive ask — returns cited passages or abstains. No LLM."""
    filters = filters or RetrievalFilter(
        statuses=["approved"],
        exclude_precedent_from_normative=True,
    )
    evidence, audit = retriever.retrieve(question, top_k=top_k, filters=filters)
    if not evidence or (evidence[0].relevance_score or 0) < min_top_score:
        return AskResult(
            question=question,
            abstained=True,
            answer=ABSTAIN_MESSAGE,
            evidence=[],
            audit=audit,
        )
    # Build a short extractive answer from top excerpts
    lines = [
        "Based on approved demo knowledge (illustrative only):",
    ]
    for i, ev in enumerate(evidence[:3], 1):
        section = " > ".join(ev.heading_path[1:]) if len(ev.heading_path) > 1 else ev.title
        lines.append(f"{i}. [{ev.title} · {section} · v{ev.version} · {ev.status}]")
        lines.append(f"   {ev.excerpt[:280]}")
        lines.append(f"   citation: `{ev.evidence_id}` / chunk `{ev.chunk_id}`")
    return AskResult(
        question=question,
        abstained=False,
        answer="\n".join(lines),
        evidence=evidence,
        audit=audit,
    )


def compare_clause_to_knowledge(
    intent: str,
    *,
    contract_quote: Optional[str] = None,
    contract_page: Optional[int] = None,
    rule_id: Optional[str] = None,
    retriever: Optional[LexicalKnowledgeRetriever] = None,
    top_k: int = 4,
) -> ClauseKnowledgeComparison:
    """MVP: retrieve policy for renewal/SLA (and other mapped intents)."""
    if retriever is None:
        retriever, _, _ = build_default_retriever()
    conf = CLAUSE_INTENTS.get(intent)
    if conf is None:
        return ClauseKnowledgeComparison(
            intent=intent,
            rule_id=rule_id,
            contract_quote=contract_quote,
            contract_page=contract_page,
            assessment=ABSTAIN_MESSAGE,
            action="Manual review — unknown clause intent",
            abstained=True,
        )

    filters = RetrievalFilter(
        statuses=["approved"],
        document_types=list(conf["document_types"]),
        exclude_precedent_from_normative=True,
    )
    query = conf["query"]
    if contract_quote:
        query = f"{query}; {contract_quote[:160]}"
    evidence, audit = retriever.retrieve(query, top_k=top_k, filters=filters)

    if not evidence:
        return ClauseKnowledgeComparison(
            intent=intent,
            rule_id=rule_id,
            contract_quote=contract_quote,
            contract_page=contract_page,
            assessment=ABSTAIN_MESSAGE,
            action="Escalate — no approved policy evidence retrieved",
            abstained=True,
            audit=audit,
        )

    assessment, action = _assess_clause(intent, contract_quote, evidence)
    return ClauseKnowledgeComparison(
        intent=intent,
        rule_id=rule_id,
        contract_quote=contract_quote,
        contract_page=contract_page,
        assessment=assessment,
        action=action,
        knowledge_evidence=evidence,
        abstained=False,
        audit=audit,
    )


def _has_ticket_only(quote: str) -> bool:
    q = (quote or "").lower()
    ticket = any(w in q for w in ("ticket", "tickets", "answered on time", "support ticket"))
    uptime = any(w in q for w in ("uptime", "availability", "platform availability"))
    return ticket and not uptime


def _assess_clause(
    intent: str,
    contract_quote: Optional[str],
    evidence: list[RetrievedEvidence],
) -> tuple[str, str]:
    """Lightweight deterministic comparison for the two MVP examples."""
    quote = (contract_quote or "").lower()
    joined = " ".join(e.excerpt.lower() for e in evidence)

    if intent == "renewal":
        # Policy preferred: 60 days; 90 days = process deviation; <30 = walk-away
        if "90" in quote and "day" in quote:
            return (
                "Contract requires 90 days' notice; approved demo policy prefers 60 days. "
                "This is a procurement process deviation, not automatically a legal violation.",
                "Recommend aligning notice to 60 days (preferred) or accept 90 days with Procurement acknowledgment.",
            )
        if re.search(r"\b([12]?\d)\s*days?\b", quote):
            m = re.search(r"\b([12]?\d)\s*days?\b", quote)
            days = int(m.group(1)) if m else None
            if days is not None and days < 30:
                return (
                    f"Contract notice ({days} days) is below the 30-day walk-away floor in demo policy.",
                    "Route to Legal and Procurement.",
                )
        if "60 days" in joined:
            return (
                "Retrieved approved renewal policy (preferred 60-day notice).",
                "Compare extracted notice days against preferred 60 / walk-away <30.",
            )

    if intent == "sla":
        if _has_ticket_only(quote):
            return (
                "Contract states a support-ticket response percentage only — no platform "
                "uptime/availability commitment. Approved demo SLA policy requires 99.9% uptime.",
                "Treat as missing uptime SLA (walk-away); route to Business Owner and Procurement.",
            )
        if "99.5" in quote or "99.0" in quote or "98%" in quote or "98 " in quote:
            return (
                f"Contract offers uptime/availability below the approved demo policy "
                f"minimum of 99.9% (contract excerpt mentions "
                f"{'99.5' if '99.5' in quote else 'a lower %'}).",
                "Flag operational exception; route to Business Owner and Procurement.",
            )
        if "99.9" in quote:
            return (
                "Contract uptime appears aligned with the 99.9% minimum in approved SLA policy.",
                "Confirm service-credit formula against preferred 5%/20% credits.",
            )
        if "99.9%" in joined:
            return (
                "Retrieved approved SLA policy requiring 99.9% minimum uptime.",
                "Compare extracted uptime against 99.9% and route if below.",
            )

    return (
        "Retrieved approved knowledge for this clause category (illustrative demo policy).",
        "Review side-by-side evidence; apply deterministic playbook rules for severity/routing.",
    )


def comparisons_for_package_findings(
    findings: list,  # ScoredFinding-like
    retriever: Optional[LexicalKnowledgeRetriever] = None,
) -> list[ClauseKnowledgeComparison]:
    """Build side-by-side comparisons for findings that map to a RAG intent."""
    if retriever is None:
        retriever, _, _ = build_default_retriever()
    out: list[ClauseKnowledgeComparison] = []
    seen_intents: set[str] = set()
    for f in findings:
        rule_id = f.grounded.finding.rule_id
        intent = RULE_TO_INTENT.get(rule_id)
        if not intent or intent in seen_intents:
            continue
        # Prefer MVP intents first but allow others
        quote = None
        page = None
        evs = f.grounded.finding.evidence
        if evs:
            quote = evs[0].quote
            page = evs[0].page
        out.append(
            compare_clause_to_knowledge(
                intent,
                contract_quote=quote,
                contract_page=page,
                rule_id=rule_id,
                retriever=retriever,
            )
        )
        seen_intents.add(intent)
    return out


# --------------------------------------------------------------------------- #
# Workflow grounding helpers (Phase 3)
# --------------------------------------------------------------------------- #
NEGOTIATION_QUERIES = [
    (
        "budget ceiling approval thresholds preferred term exception policy",
        ["procurement_policy", "negotiation_playbook"],
    ),
    (
        "minimum uptime 99.9% SLA service credits negotiation fallback",
        ["sla_policy", "negotiation_playbook", "approved_clauses"],
    ),
    (
        "liability cap indemnity legal review do not concede uncapped",
        ["approved_clauses", "negotiation_playbook"],
    ),
    (
        "data processing agreement DPA privacy mandatory",
        ["security_privacy_standard", "approved_clauses"],
    ),
]


def format_evidence_for_prompt(evidence: list[RetrievedEvidence], *, limit: int = 6) -> list[str]:
    """Compact cited passages for agent prompts (not free-form model knowledge)."""
    lines: list[str] = []
    for ev in evidence[:limit]:
        section = " > ".join(ev.heading_path[1:]) if len(ev.heading_path) > 1 else ""
        lines.append(
            f"[{ev.evidence_id} | {ev.title} | v{ev.version} | {ev.status}"
            f"{(' | ' + section) if section else ''}]\n{ev.excerpt}"
        )
    return lines


def retrieve_for_negotiation(
    *,
    retriever: Optional[LexicalKnowledgeRetriever] = None,
    top_k_per_query: int = 2,
) -> tuple[list[RetrievedEvidence], list[str], RetrievalAudit]:
    """Retrieve approved negotiation / policy evidence for the strategist."""
    if retriever is None:
        retriever, _, _ = build_default_retriever()
    merged: list[RetrievedEvidence] = []
    seen: set[str] = set()
    all_scores: dict[str, float] = {}
    queries: list[str] = []
    for query, doc_types in NEGOTIATION_QUERIES:
        queries.append(query)
        ev, _ = retriever.retrieve(
            query,
            top_k=top_k_per_query,
            filters=RetrievalFilter(
                statuses=["approved"],
                document_types=doc_types,
                exclude_precedent_from_normative=True,
            ),
        )
        for e in ev:
            if e.chunk_id in seen:
                continue
            seen.add(e.chunk_id)
            merged.append(e)
            if e.relevance_score is not None:
                all_scores[e.chunk_id] = e.relevance_score
    audit = RetrievalAudit(
        query=" | ".join(queries),
        filters={"statuses": ["approved"], "exclude_precedent": True},
        returned_chunk_ids=[e.chunk_id for e in merged],
        scores=all_scores,
        selected_evidence_ids=[e.evidence_id for e in merged],
        corpus_id=retriever.corpus_id,
        mode="lexical_negotiation",
    )
    return merged, format_evidence_for_prompt(merged), audit


def retrieve_for_approval_reason(
    reason_code: str,
    title: str,
    *,
    retriever: Optional[LexicalKnowledgeRetriever] = None,
    top_k: int = 3,
) -> list[RetrievedEvidence]:
    """Retrieve approval-matrix / policy passages that explain why a task exists."""
    if retriever is None:
        retriever, _, _ = build_default_retriever()
    query = f"{reason_code} {title} approval exception routing policy"
    # Map reason codes to doc types
    type_map = {
        "liability_exception": ["approved_clauses", "negotiation_playbook"],
        "governing_law": ["approved_clauses"],
        "term_renewal_assignment": ["renewal_and_exit_policy", "approved_clauses"],
        "privacy_dpa": ["security_privacy_standard", "approved_clauses"],
        "commercial_variance": ["procurement_policy", "payment_terms_policy"],
        "operational_sla": ["sla_policy", "approved_clauses"],
        "security_flag": ["security_privacy_standard"],
        "needs_information": ["procurement_policy"],
    }
    doc_types = None
    for key, types in type_map.items():
        if key in reason_code:
            doc_types = types
            break
    filters = RetrievalFilter(
        statuses=["approved"],
        document_types=doc_types,
        exclude_precedent_from_normative=True,
    )
    ev, _ = retriever.retrieve(query, top_k=top_k, filters=filters)
    return ev


def ground_review_package(
    pkg,  # ReviewPackage
    *,
    retriever: Optional[LexicalKnowledgeRetriever] = None,
) -> list[ClauseKnowledgeComparison]:
    """Attach contract-vs-knowledge comparisons for a review package."""
    return comparisons_for_package_findings(pkg.findings, retriever=retriever)
