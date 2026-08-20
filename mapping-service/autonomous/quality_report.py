"""Render the stratified-sample quality review as a self-contained folder.

The folder is the deliverable: an HTML report next to the exact images the
reviewer saw, so a reader can disagree with a verdict by looking at the same
evidence instead of taking the number on trust.
"""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .quality_gate import GateOutcome
from .sampling_plan import SamplingPlan


REPORT_DIRNAME = "quality-report"
MATERIALS_DIRNAME = "materials"


@dataclass(frozen=True)
class RoundRecord:
    """One review round: what was sampled, what the reviewer said, what it meant."""

    index: int
    label: str
    case_results: tuple[dict[str, Any], ...]
    outcome: GateOutcome
    verification_report: dict[str, Any] = field(default_factory=dict)
    adjustment: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "sampled_ids": [str(result.get("oachargeid") or "") for result in self.case_results],
            "outcome": self.outcome.describe(),
            "adjustment": self.adjustment,
        }


def _verdict(result: dict[str, Any]) -> str:
    value = result.get("verdict")
    return str(getattr(value, "value", value) or "")


def _case_id(result: dict[str, Any]) -> str:
    return str(result.get("oachargeid") or "").strip()


def _safe_token(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value).strip("-")
    return cleaned or "case"


def copy_materials(rounds: Sequence[RoundRecord], destination: Path) -> dict[str, list[str]]:
    """Copy the reviewed images and per-case JSON next to the report.

    Paths recorded during the run point into the immutable job artifact store,
    which is pruned on its own schedule; the report has to keep its own copy or
    it stops being reviewable.
    """
    materials_root = destination / MATERIALS_DIRNAME
    copied: dict[str, list[str]] = {}
    for record in rounds:
        for result in record.case_results:
            case = _case_id(result)
            if not case:
                continue
            case_dir = materials_root / _safe_token(case)
            case_dir.mkdir(parents=True, exist_ok=True)
            names: list[str] = []
            for index, image in enumerate(result.get("selected_images") or [], start=1):
                source = Path(str(image))
                if not source.is_file():
                    continue
                target = case_dir / f"page-{index:02d}{source.suffix.lower()}"
                shutil.copy2(source, target)
                names.append(f"{MATERIALS_DIRNAME}/{_safe_token(case)}/{target.name}")
            for key in ("expectation_path", "observation_path"):
                source_value = result.get(key)
                if not source_value:
                    continue
                source = Path(str(source_value))
                if source.is_file():
                    shutil.copy2(source, case_dir / f"{key.replace('_path', '')}.json")
            copied[case] = names
    return copied


def _identity_rows(result: dict[str, Any]) -> list[tuple[str, str]]:
    expectation = result.get("expectation") or {}
    rows: list[tuple[str, str]] = []
    for label, key in (
        ("Reference", "reference_values"),
        ("Address", "address_values"),
        ("Description", "description_values"),
    ):
        values = expectation.get(key) or ()
        if values:
            rows.append((label, "; ".join(str(value) for value in values)))
    return rows


def _signal_badges(result: dict[str, Any]) -> str:
    signals = result.get("signals") or {}
    if not isinstance(signals, dict) or not signals:
        return '<span class="muted">no signals recorded</span>'
    parts = []
    for name, matched in signals.items():
        state = "hit" if matched else "miss"
        parts.append(f'<span class="badge {state}">{html.escape(str(name))}</span>')
    return "".join(parts)


def _case_card(result: dict[str, Any], images: Sequence[str]) -> str:
    verdict = _verdict(result)
    case = _case_id(result)
    identity = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in _identity_rows(result)
    )
    gallery = "".join(
        f'<a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="reviewed page"></a>'
        for name in images
    ) or '<p class="muted">No page image was reviewed for this case.</p>'
    return f"""
      <article class="case {html.escape(verdict)}">
        <header>
          <h3>{html.escape(case)}</h3>
          <span class="verdict {html.escape(verdict)}">{html.escape(verdict or 'unknown')}</span>
        </header>
        <p class="reason">{html.escape(str(result.get('reason') or ''))}</p>
        <div class="signals">{_signal_badges(result)}</div>
        <table class="identity">
          <tr><th>Route</th><td>{html.escape(str(result.get('route') or ''))}</td></tr>
          <tr><th>Basis</th><td>{html.escape(str(result.get('match_basis') or ''))}</td></tr>
          <tr><th>Mapped to</th><td class="path">{html.escape(str(result.get('mapping_path') or ''))}</td></tr>
          {identity}
        </table>
        <div class="gallery">{gallery}</div>
      </article>
    """


STYLE = """
:root { color-scheme: light dark; --fg:#111; --muted:#666; --line:#ddd; --bg:#fff;
        --ok:#0a7a3d; --bad:#b3261e; --warn:#8a6100; --chip:#f2f2f2; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --muted:#9a9a9a; --line:#333; --bg:#151515;
          --ok:#5bd18a; --bad:#ff6b60; --warn:#e0b355; --chip:#242424; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:1.6rem; margin:0 0 .25rem; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem; border-bottom:1px solid var(--line); }
h3 { font-size:1rem; margin:0; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.muted { color:var(--muted); }
.sub { color:var(--muted); margin:0 0 1.5rem; }
.banner { padding:1rem 1.25rem; border-radius:8px; border:1px solid var(--line); margin:1.5rem 0; }
.banner.pass { border-color:var(--ok); }
.banner.fail { border-color:var(--bad); }
.banner h2 { margin:0 0 .35rem; border:0; padding:0; font-size:1.1rem; }
.banner.pass h2 { color:var(--ok); }
.banner.fail h2 { color:var(--bad); }
table { border-collapse:collapse; width:100%; margin:.5rem 0; }
th, td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-weight:600; white-space:nowrap; width:1%; }
.wide { overflow-x:auto; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin:1rem 0; }
.stat { background:var(--chip); border-radius:6px; padding:.75rem; }
.stat b { display:block; font-size:1.5rem; line-height:1.2; }
.stat span { color:var(--muted); font-size:.85rem; }
.case { border:1px solid var(--line); border-radius:8px; padding:1rem; margin:1rem 0; }
.case header { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:.5rem; }
.verdict { font-size:.8rem; padding:.15rem .5rem; border-radius:999px; background:var(--chip); white-space:nowrap; }
.verdict.verified_same { color:var(--ok); }
.verdict.verified_wrong { color:var(--bad); }
.verdict.missing_document, .verdict.unreadable { color:var(--warn); }
.reason { margin:.25rem 0 .75rem; color:var(--muted); }
.badge { display:inline-block; font-size:.75rem; padding:.1rem .45rem; border-radius:999px;
         background:var(--chip); margin-right:.3rem; }
.badge.hit { color:var(--ok); }
.badge.miss { color:var(--bad); text-decoration:line-through; }
.identity td.path { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85rem; word-break:break-all; }
.gallery { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.75rem; }
.gallery img { height:190px; border:1px solid var(--line); border-radius:4px; }
ul.reasons { margin:.5rem 0 0; padding-left:1.2rem; }
"""


def _stat(value: Any, label: str) -> str:
    return f'<div class="stat"><b>{html.escape(str(value))}</b><span>{html.escape(label)}</span></div>'


def render_quality_report(
    *,
    destination: Path,
    council: str,
    batch: str,
    run_id: str,
    plan: SamplingPlan,
    rounds: Sequence[RoundRecord],
    acceptance: RoundRecord | None,
    mapping_summary: dict[str, Any],
    generated_at: str,
) -> Path:
    """Write index.html plus materials/ and return the report directory."""
    destination.mkdir(parents=True, exist_ok=True)
    all_rounds = list(rounds) + ([acceptance] if acceptance else [])
    images_by_case = copy_materials(all_rounds, destination)

    decisive = acceptance or (rounds[-1] if rounds else None)
    # A pass requires the holdout. A working round that clears the gate only
    # shows the loop satisfied the reviews it was tuned against, and reporting
    # that as "passed" is exactly the claim the holdout exists to prevent.
    passed = bool(acceptance and acceptance.outcome.passed)
    if acceptance is None:
        headline = "No independent acceptance sample was reviewed"
        detail = (
            "Every round below drew from the working pool, which the mapping was adjusted against. "
            "These numbers describe the loop, not the mapping."
        )
    elif passed:
        headline = "Passed on the holdout sample"
        detail = (
            "The acceptance sample was reserved before any adjustment and never informed a change "
            "to the mapping spec, so this result describes the mapping."
        )
    else:
        headline = "Failed on the holdout sample"
        detail = "The mapping did not clear the gate on cases it was never tuned against."

    # A sample can only speak for cases the mapping accepted. Coverage says how
    # many that was, and is shown next to the verdict so a high pass rate over a
    # thin slice cannot read as a clean result.
    coverage = mapping_summary.get("coverage") or {}
    coverage_metrics = coverage.get("metrics") or {}
    if coverage_metrics and not coverage.get("passed", True):
        headline = f"{headline} — but coverage is short"
        detail = (
            f"{detail} Only {coverage_metrics.get('accepted', 0)} of "
            f"{coverage_metrics.get('population', 0)} cases were resolved at all."
        )
        passed = False

    plan_summary = plan.describe()
    rounds_html = "".join(
        f"""<tr>
              <td>{record.index}</td>
              <td>{html.escape(record.label)}</td>
              <td>{len(record.case_results)}</td>
              <td>{record.outcome.metrics.get('verified_rate', 0):.0%}</td>
              <td>{record.outcome.metrics.get('verified_wrong', 0)}</td>
              <td>{'pass' if record.outcome.passed else 'fail'}</td>
              <td>{html.escape(record.adjustment or '—')}</td>
            </tr>"""
        for record in all_rounds
    )

    signature_rows = "".join(
        f"""<tr>
              <td>{html.escape(signature.route)}</td>
              <td>{html.escape(signature.match_basis)}</td>
              <td>{html.escape(', '.join(signature.failing_signals) or '—')}</td>
              <td>{signature.count}</td>
              <td>{'yes' if signature.systematic else 'no'}</td>
              <td class="muted">{html.escape(', '.join(signature.example_ids))}</td>
            </tr>"""
        for record in all_rounds
        for signature in record.outcome.signatures
    ) or '<tr><td colspan="6" class="muted">No case was confirmed wrong.</td></tr>'

    reasons_html = ""
    if decisive and decisive.outcome.reasons:
        items = "".join(f"<li>{html.escape(reason)}</li>" for reason in decisive.outcome.reasons)
        reasons_html = f'<ul class="reasons">{items}</ul>'

    cases_html = "".join(
        _case_card(result, images_by_case.get(_case_id(result), ()))
        for record in all_rounds
        for result in record.case_results
    )

    document = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(council)} {html.escape(batch)} — mapping quality review</title>
<style>{STYLE}</style>
<body>
<h1>{html.escape(council)} · {html.escape(batch)} — mapping quality review</h1>
<p class="sub">Run {html.escape(run_id)} · generated {html.escape(generated_at)}</p>

<div class="banner {'pass' if passed else 'fail'}">
  <h2>{html.escape(headline)}</h2>
  <p class="muted">{html.escape(detail)}</p>
  {reasons_html}
</div>

<div class="grid">
  {_stat(mapping_summary.get('case_count', '—'), 'cases mapped')}
  {_stat(f"{coverage_metrics.get('accepted_rate', 0):.0%}" if coverage_metrics else '—', 'resolved (coverage)')}
  {_stat(plan_summary['working_cases'], 'working pool')}
  {_stat(plan_summary['holdout_cases'], 'holdout (reserved)')}
  {_stat(len(all_rounds), 'review rounds')}
  {_stat(sum(len(record.case_results) for record in all_rounds), 'cases reviewed')}
</div>

<h2>Coverage</h2>
<p class="muted">
  Precision and coverage are judged separately. The reviewed sample shows whether accepted mappings
  point at the right document; it cannot show how many cases were accepted, because a case the mapping
  rejected has no document to review. Cases the source itself reports as having no scan are excluded
  from the resolvable count, since no mapping spec can resolve them.
</p>
<table>
  <tr><th>Population</th><td>{coverage_metrics.get('population', '—')}</td></tr>
  <tr><th>Resolved</th><td>{coverage_metrics.get('accepted', '—')}
      ({coverage_metrics.get('accepted_rate', 0):.1%} of population,
       {coverage_metrics.get('accepted_rate_of_resolvable', 0):.1%} of resolvable)</td></tr>
  <tr><th>Unmatched by the spec</th><td>{coverage_metrics.get('unmatched', '—')}
      ({coverage_metrics.get('unmatched_rate', 0):.1%})</td></tr>
  <tr><th>Source reports no scan</th><td>{coverage_metrics.get('source_reports_no_scan', '—')}
      <span class="muted">— not a mapping defect</span></td></tr>
</table>

<h2>Sampling design</h2>
<p class="muted">
  The population was split once, stratum by stratum, before any review. Optimisation rounds drew
  only from the working pool and never reused a case, so a later round tests new ground instead of
  re-confirming an earlier fix. The holdout was never used to decide an adjustment.
</p>
<table>
  <tr><th>Population</th><td>{plan_summary['population']}</td></tr>
  <tr><th>Working pool</th><td>{plan_summary['working_cases']}</td></tr>
  <tr><th>Holdout</th><td>{plan_summary['holdout_cases']} ({plan_summary['holdout_fraction_actual']:.1%} of population)</td></tr>
  <tr><th>Strata with no holdout</th><td>{len(plan_summary['strata_without_holdout'])}
      <span class="muted">— too small to reserve a case without starving the working pool</span></td></tr>
</table>

<h2>Rounds</h2>
<div class="wide"><table>
  <tr><th>#</th><th>Sample</th><th>Cases</th><th>Verified</th><th>Wrong</th><th>Gate</th><th>Adjustment made after</th></tr>
  {rounds_html}
</table></div>

<h2>Failure signatures</h2>
<p class="muted">
  Confirmed-wrong cases grouped by route, matching basis, and which identity signals failed.
  A group of two or more is treated as a rule-level defect rather than a one-off bad case.
</p>
<div class="wide"><table>
  <tr><th>Route</th><th>Basis</th><th>Failing signals</th><th>Count</th><th>Systematic</th><th>Examples</th></tr>
  {signature_rows}
</table></div>

<h2>Reviewed cases</h2>
<p class="muted">
  Each card shows what the source table expected and the pages the reviewer actually saw. Images were
  presented without filenames or paths, so a verdict rests on page content rather than on the mapping
  it was meant to check.
</p>
{cases_html}
</body>
"""
    index_path = destination / "index.html"
    index_path.write_text(document, encoding="utf-8")

    (destination / "quality-report.json").write_text(
        json.dumps(
            {
                "council": council,
                "batch": batch,
                "run_id": run_id,
                "generated_at": generated_at,
                "passed": passed,
                "acceptance_reviewed": acceptance is not None,
                "coverage": coverage,
                "sampling_plan": plan_summary,
                "rounds": [record.describe() for record in all_rounds],
                "mapping_summary": mapping_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
