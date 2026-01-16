import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def load_devset(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    meta = data.get("meta", {})
    return samples, meta


def init_router(repo_root: Path):
    sys.path.append(str(repo_root))
    from app.utils.config_loader import ConfigLoader
    from app.router.router import Router

    config_dir = repo_root / "config"
    loader = ConfigLoader(
        str(config_dir / "action_catalog.yaml"), str(config_dir / "keyword_rules.yaml")
    )
    loader.load()
    return Router(loader)


def evaluate(router, samples: List[Dict]):
    from app.core.models import UserRequest

    total = len(samples)
    top1_hits = 0
    top3_hits = 0
    mrr_sum = 0.0
    per_domain = {}

    details = []
    for sample in samples:
        expected = sample["expected_action"]
        req = UserRequest(text=sample["text"], user_id="dev_eval")
        output = router.route(req)
        ranked = [c.action_id for c in output.top_actions]

        rank = ranked.index(expected) + 1 if expected in ranked else None
        top1_hits += 1 if rank == 1 else 0
        top3_hits += 1 if rank is not None and rank <= 3 else 0
        mrr_sum += 1.0 / rank if rank else 0.0

        domain = sample.get("domain", "unknown")
        dom_stats = per_domain.setdefault(domain, {"n": 0, "top1": 0, "top3": 0, "mrr": 0.0})
        dom_stats["n"] += 1
        dom_stats["top1"] += 1 if rank == 1 else 0
        dom_stats["top3"] += 1 if rank is not None and rank <= 3 else 0
        dom_stats["mrr"] += 1.0 / rank if rank else 0.0

        details.append(
            {
                "text": sample["text"],
                "expected": expected,
                "ranked": ranked,
                "rank": rank,
                "ui_strategy": output.ui_strategy.value,
            }
        )

    summary = {
        "total": total,
        "top1_accuracy": round(top1_hits / total, 4) if total else 0.0,
        "top3_recall": round(top3_hits / total, 4) if total else 0.0,
        "mrr": round(mrr_sum / total, 4) if total else 0.0,
        "per_domain": {},
    }
    for dom, stats in per_domain.items():
        n = stats["n"]
        summary["per_domain"][dom] = {
            "n": n,
            "top1_accuracy": round(stats["top1"] / n, 4) if n else 0.0,
            "top3_recall": round(stats["top3"] / n, 4) if n else 0.0,
            "mrr": round(stats["mrr"] / n, 4) if n else 0.0,
        }

    return summary, details


def _configure_stdout():
    """Force stdout to UTF-8 when có dấu để tránh lỗi encode trên Windows."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Router on a devset.")
    parser.add_argument(
        "--devset",
        type=str,
        default=str(Path(__file__).parent / "fixtures" / "devset.yaml"),
        help="Path to devset YAML file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()

    _configure_stdout()
    repo_root = Path(__file__).resolve().parents[2]
    samples, meta = load_devset(Path(args.devset))

    try:
        router = init_router(repo_root)
    except Exception as exc:
        print(f"Failed to initialize router: {exc}")
        sys.exit(1)

    summary, details = evaluate(router, samples)

    print("== Summary ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n== Meta ==")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "meta": meta,
                    "details": details,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
