"""Visual system for the Streamlit demo UI.

Colours, type and radius live in ``.streamlit/config.toml``; this module adds
the few pieces Streamlit has no widget for (status chips, banners, a status
stepper, key-term grids) plus small formatting helpers. Everything here is
presentation only — no business logic.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from html import escape
from typing import Iterable, Optional

import streamlit as st

IST = timezone(timedelta(hours=5, minutes=30), "IST")

# Tone -> (background, text). Tones double as outcome colours.
TONES = {
    "green": ("#E6F4EA", "#0D652D"),
    "amber": ("#FEF3E0", "#8A4B00"),
    "red": ("#FCE8E6", "#A50E0E"),
    "purple": ("#F1E6FC", "#5E2A9C"),
    "blue": ("#D7E3FB", "#0A2D6E"),
    "gray": ("#EEF0F4", "#41454C"),
}

# Finding outcome -> (label, tone). Colour follows the outcome, not severity:
# a compliant high-severity clause is green, not red.
OUTCOMES = {
    "preferred": ("Preferred", "green"),
    "fallback": ("Fallback", "amber"),
    "walk_away": ("Walk-away", "red"),
    "not_found": ("Missing", "purple"),
    "unclear": ("Needs review", "gray"),
}

ROUTES = {
    "manual_review": ("Manual review", "Fail-closed: a person must review everything"),
    "legal": ("Legal", "Walk-away on a high-severity or Legal-review clause"),
    "procurement_owner": ("Procurement owner", "Commercial walk-away or security flag"),
    "low_risk_queue": ("Low-risk queue", "Only preferred or fallback terms"),
}

_CSS = """
<style>
.block-container{padding-top:5rem;padding-bottom:4rem;max-width:1200px}
/* Streamlit Cloud shows a header bar (Share, GitHub) over the page top: keep titles below it */
[data-testid="stMainBlockContainer"]{padding-top:5rem}
h1{letter-spacing:-0.015em}
[data-testid="stMetric"]{background:#FFFFFF}
[data-testid="stMetricLabel"] p{color:#5B5F66;font-weight:500}
.cr-chip{display:inline-flex;align-items:center;gap:6px;height:26px;padding:0 10px;
  border-radius:13px;font-size:12.5px;font-weight:600;white-space:nowrap;line-height:1}
.cr-sub{color:#5B5F66;font-size:15px;margin:2px 0 0}
.cr-head{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.cr-head h1{margin:0;padding:0;font-size:2rem;font-weight:600;line-height:1.2}
.cr-banner{display:flex;gap:14px;align-items:flex-start;padding:14px 18px;border-radius:14px;
  margin:4px 0}
.cr-banner b{display:block;font-size:15px;font-weight:600;margin-bottom:2px}
.cr-banner span{font-size:14px;line-height:1.5}
.cr-banner .ic{font-size:20px;line-height:1.2}
.cr-steps{list-style:none;margin:0;padding:16px 20px !important;border:1px solid #E3E5EA;border-radius:14px;
  display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;background:#FFFFFF}
.cr-steps li{display:flex;align-items:center;gap:10px}
.cr-dot{flex:0 0 auto;width:28px;height:28px;border-radius:14px;display:inline-flex;
  align-items:center;justify-content:center;font-size:13px;font-weight:700}
.cr-steps .t{display:flex;flex-direction:column;font-size:14px;font-weight:600;line-height:1.25}
.cr-steps .t small{font-size:12px;font-weight:400;color:#5B5F66}
.cr-terms{padding:0 !important;display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px 24px;margin:0}
.cr-terms div{margin:0;padding:0}
.cr-terms dt{margin:0;padding:0;font-size:13px;color:#5B5F66;margin-bottom:2px}
.cr-terms dd{margin:0;font-size:15px;font-weight:600}
.cr-cite{display:inline-flex;align-items:center;height:20px;padding:0 7px;margin-left:6px;
  border-radius:6px;background:#EEF0F4;color:#41454C;font:500 11.5px 'JetBrains Mono',monospace;
  vertical-align:middle}
.cr-quote{padding:16px 18px;border-radius:12px;font-size:15px;line-height:1.55;white-space:pre-line}
.cr-quote .lbl{display:block;font-size:11.5px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:8px}
.cr-pipe{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0}
.cr-pipe .cr-chip{height:30px;padding:0 12px;background:#FFFFFF;border:1px solid #E3E5EA}
.cr-list{list-style:none;margin:0;padding:0 !important;display:flex;flex-direction:column;gap:12px}
.cr-list li{display:flex;gap:12px;align-items:flex-start}
.cr-list .b{display:flex;flex-direction:column;gap:2px;font-size:14px}
.cr-list .b small{color:#5B5F66;font-size:13px}
.cr-av{flex:0 0 auto;width:36px;height:36px;border-radius:18px;display:inline-flex;align-items:center;
  justify-content:center;font-size:12px;font-weight:700}
.cr-mono{font-family:'JetBrains Mono',monospace;font-size:12px;color:#5B5F66}
.cr-date{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;justify-content:center;
  width:60px;height:60px;border-radius:12px;background:#FFFFFF;color:#A50E0E}
.cr-date small{font-size:11px;font-weight:700}
.cr-date b{font-size:22px;line-height:1;margin:0}
.cr-empty{padding:40px 24px;border:1px dashed #C4C7CE;border-radius:16px;text-align:center;color:#41454C}
.cr-empty b{display:block;font-size:17px;margin-bottom:6px;color:#1B1C1F}
.cr-section{font-size:12px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#5B5F66;
  margin:6px 0 2px}
</style>
"""


def inject() -> None:
    st.html(_CSS)


# --------------------------------------------------------------------------- #
# Small HTML pieces (all dynamic text is escaped)
# --------------------------------------------------------------------------- #
def chip(text: str, tone: str = "gray") -> str:
    bg, fg = TONES.get(tone, TONES["gray"])
    return f'<span class="cr-chip" style="background:{bg};color:{fg}">{escape(str(text))}</span>'


def outcome_chip(status: str) -> str:
    label, tone = OUTCOMES.get(status, (humanize(status), "gray"))
    return chip(label, tone)


def page_header(title: str, subtitle: str = "", chips: Iterable[str] = ()) -> None:
    chips_html = "".join(chips)
    sub = f'<p class="cr-sub">{escape(subtitle)}</p>' if subtitle else ""
    st.html(f'<div class="cr-head"><h1>{escape(title)}</h1>{chips_html}</div>{sub}')


def banner(title: str, body: str = "", tone: str = "red", icon: str = "!") -> None:
    bg, fg = TONES.get(tone, TONES["red"])
    st.html(
        f'<div class="cr-banner" role="{"alert" if tone == "red" else "status"}" '
        f'style="background:{bg};color:{fg}">'
        f'<span class="ic" aria-hidden="true">{escape(icon)}</span>'
        f'<div><b>{escape(title)}</b><span style="color:#1B1C1F">{escape(body)}</span></div></div>'
    )


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="cr-empty"><b>{escape(title)}</b>{escape(body)}</div>')


def section(text: str) -> None:
    st.html(f'<div class="cr-section">{escape(text)}</div>')


def stepper(steps: list[tuple[str, str, str]]) -> None:
    """steps: (label, caption, state) where state is done | current | warn | todo."""
    styles = {
        "done": ("#0B57D0", "#FFFFFF", "✓"),
        "final": ("#0D652D", "#FFFFFF", "✓"),
        "current": ("#D7E3FB", "#0A2D6E", "•"),
        "warn": ("#FEF3E0", "#8A4B00", "!"),
        "todo": ("#EEF0F4", "#5B5F66", ""),
    }
    items = []
    for i, (label, caption, state) in enumerate(steps, 1):
        bg, fg, glyph = styles.get(state, styles["todo"])
        items.append(
            f'<li><span class="cr-dot" style="background:{bg};color:{fg}">{glyph or i}</span>'
            f'<span class="t">{escape(label)}<small>{escape(caption)}</small></span></li>'
        )
    st.html(f'<ol class="cr-steps" aria-label="Status">{"".join(items)}</ol>')


def key_terms(rows: list[tuple[str, str, Optional[int]]]) -> None:
    cells = []
    for label, value, page in rows:
        cite = f'<span class="cr-cite">p{page}</span>' if page else ""
        cells.append(
            f"<div><dt>{escape(label)}</dt><dd>{escape(value or '—')}{cite}</dd></div>"
        )
    st.html(f'<dl class="cr-terms">{"".join(cells)}</dl>')


def quote_box(label: str, text: str, tone: str = "gray", strike: bool = False) -> None:
    bg, fg = {
        "gray": ("#F6F7FA", "#5B5F66"),
        "blue": ("#EEF3FD", "#0A2D6E"),
        "red": ("#FDF0EE", "#8C1D18"),
        "green": ("#E9F5EC", "#0D652D"),
    }[tone]
    deco = "text-decoration:line-through;text-decoration-color:#C5221F;" if strike else ""
    st.html(
        f'<div class="cr-quote" style="background:{bg}"><span class="lbl" style="color:{fg}">'
        f'{escape(label)}</span><span style="{deco}">{escape(text)}</span></div>'
    )


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def humanize(code: Optional[str]) -> str:
    if not code:
        return "—"
    return str(code).replace("_", " ").strip().capitalize()


def fmt_date(d: Optional[date | str]) -> str:
    if not d:
        return "—"
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d[:10])
        except ValueError:
            return d
    return f"{d.day} {d:%b %Y}"


def fmt_ts(ts: Optional[datetime]) -> str:
    if not ts:
        return "—"
    if ts.tzinfo is not None:
        ts = ts.astimezone(IST)
    return f"{ts.day} {ts:%b}, {ts:%H:%M}"


def fmt_inr(amount: Optional[float | int | str]) -> str:
    """Indian digit grouping: 4800000 -> ₹48,00,000."""
    if amount in (None, ""):
        return "—"
    try:
        n = int(round(float(amount)))
    except (TypeError, ValueError):
        return str(amount)
    sign, s = ("-" if n < 0 else ""), str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"{sign}₹{s}"
