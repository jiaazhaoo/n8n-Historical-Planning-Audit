#!/usr/bin/env python3
"""Drive one council work package from evidence to published mapping, locally.

The same chain the n8n workflow walks, without n8n: map, review the working
pool, act on what the gate says, review the holdout, publish. n8n contributes
the form, the branch names and the execution history; every decision comes back
from the service, so the loop here is the same loop, written out.

It carries one step further than the workflow does. Publication decision marks a
run cleared and stops, so nothing has ever published: this calls /publish, into
the council's own table.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Matching the workflow's own bounds. Two recompiles because an unbounded loop
# searches until something passes, which is a different thing from getting it
# right; four sampling rounds because a picture that has not changed by then is
# a judgement about specific cases, not a sampling problem.
MAX_REWORKS = 2
MAX_SAMPLING_ROUNDS = 4


def call(endpoint: str, payload: dict, *, base: str, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{base}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read() or b"{}") or {"error": f"HTTP {exc.code}"}


def fail(result: dict, step: str) -> None:
    print(f"  {step} 失败: {result.get('error')} — {str(result.get('detail'))[:200]}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--council", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--s3-inventory-paths", default="")
    parser.add_argument("--portal-evidence-paths", default="")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--base", default="http://127.0.0.1:5680")
    parser.add_argument("--timeout", type=int, default=4200)
    parser.add_argument("--publish", action="store_true", help="Publish once the holdout clears it.")
    args = parser.parse_args()

    request = {
        "council": args.council,
        "batch": args.batch,
        "input_directory": args.input_directory,
        "s3_inventory_paths": args.s3_inventory_paths,
        "portal_evidence_paths": args.portal_evidence_paths,
        "prior_findings": [],
        "rework_round": 0,
    }

    for rework in range(MAX_REWORKS + 1):
        print(f"\n=== 映射 (返工轮次 {rework}) ===")
        mapping = call("/run", request, base=args.base, timeout=args.timeout)
        if "error" in mapping:
            fail(mapping, "映射")
        counts = mapping.get("match_status_counts", {})
        total = mapping.get("case_count") or 1
        print(f"  {counts}  覆盖率 {counts.get('found', 0)}/{total} = {counts.get('found', 0) / total:.1%}")
        for missing in mapping.get("missing_evidence", []):
            print(f"  缺失证据: {missing['detail'][:180]}")

        quality = {
            "council": args.council,
            "batch": args.batch,
            "run_directory": mapping["outputs"]["run_report"].rsplit("/", 1)[0],
            "stage": "working",
            "sample_size": args.sample_size,
            "budget_usd": args.budget_usd,
        }

        outcome = None
        for round_index in range(1, MAX_SAMPLING_ROUNDS + 1):
            print(f"--- 工作池质检 第 {round_index} 轮 ---")
            outcome = call("/quality", quality, base=args.base, timeout=args.timeout)
            if "error" in outcome:
                fail(outcome, "质检")
            metrics = outcome.get("metrics", {})
            budget = outcome.get("budget", {})
            action = (outcome.get("next") or {}).get("action")
            print(f"  {metrics.get('verdict_counts')}  花费 ${budget.get('spent_usd', 0):.4f}  -> {action}")
            if action != "investigate_cases":
                break
            # Scattered failures are resolved by evidence, not by rewriting the
            # rule: another round says whether they are noise or a share the
            # first sample was too small to separate.
        action = (outcome.get("next") or {}).get("action")

        if action == "accept":
            break
        if action == "adjust_spec" and rework < MAX_REWORKS:
            finding = [str((outcome.get("next") or {}).get("detail") or "")]
            finding += [str(item) for item in (outcome.get("next") or {}).get("focus") or []]
            finding += [str(item) for item in outcome.get("reasons") or []]
            request = {
                **request,
                "rework_round": rework + 1,
                "prior_findings": request["prior_findings"] + [[x for x in finding if x]],
            }
            print(f"  规则有缺陷,带着 {len(finding)} 条发现重新编译")
            continue
        print(f"\n停在 {action}: {str((outcome.get('next') or {}).get('detail'))[:200]}")
        return 2

    print("\n=== 验收 (holdout) ===")
    acceptance = call("/quality", {**quality, "stage": "acceptance"}, base=args.base, timeout=args.timeout)
    if "error" in acceptance:
        fail(acceptance, "验收")
    print(f"  {acceptance.get('metrics', {}).get('verdict_counts')}  passed={acceptance.get('passed')}")
    if not acceptance.get("passed"):
        print(f"  未通过: {acceptance.get('reasons')}")
        return 3

    print("\n=== 发布 ===")
    publish = call(
        "/publish",
        {
            "council": args.council,
            "batch": args.batch,
            "run_directory": quality["run_directory"],
            "dry_run": not args.publish,
        },
        base=args.base,
        timeout=args.timeout,
    )
    if "error" in publish:
        fail(publish, "发布")
    print(f"  目标 {publish['runtime_path']}")
    print(f"  更新 {publish['rows_updated']} | 不变 {publish['rows_unchanged']} | 新增 {publish['rows_added']}")
    print(f"  {'已写入' if publish.get('production_published') else '试运行,未写入'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
