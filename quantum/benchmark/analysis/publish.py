"""
Publish aggregated benchmark sweeps to a Hugging Face dataset repo, for the
FastAPI Space's /v1/analysis/* endpoints to serve.

Explicit sweep dirs only — nothing is auto-discovered, so you upload exactly
what you name on the command line.

What gets uploaded per sweep, to sweeps/<sweep_id>/ in the dataset:
    index.json, manifest.json          the run plan + reproducibility record
    analysis/*.csv                     aggregate_sweep()'s tables — what the
                                       Space actually reads
    analysis/benchmark_table.tex       the paper table snippet (tiny, handy)

What is NOT uploaded: the raw per-combo benchmark_*.json (they carry decoded
paths and run to tens of MB; the CSVs are sufficient to serve) and
analysis/plots/ (screen/print exports — the Space renders Plotly from the
CSVs). A ~30 MB sweep dir ships as a few hundred KB.

Commit awareness: manifest.json records the commit each sweep ran against.
Publishing a sweep whose commit already has other published sweeps just
warns (two different sweep configs on one commit is legitimate) and adds it
alongside; pass --replace to delete those prior same-commit sweeps and stand
this one in their place instead. Re-publishing the same sweep_id always
overwrites. published.json at the dataset root is the ledger.

Auth: your cached `hf auth login` token, or --token / the HF_TOKEN env var.
The dataset repo must already exist.

The dataset card is not published from here — its canonical source is
DATASET_CARD.md in this directory; copy it to the dataset repo's README.md
by hand when it changes.

Usage:
    python -m quantum.benchmark.analysis.publish --repo user/name \\
        results/sweeps/CML/sweep_A results/sweeps/CML/sweep_B
    python -m quantum.benchmark.analysis.publish --repo user/name --dry-run <dir>
    python -m quantum.benchmark.analysis.publish --repo user/name --replace <dir>
    python -m quantum.benchmark.analysis.publish --repo user/name \\
        --drop-solver qaoa_sim <dir>
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from quantum.benchmark.analysis.aggregate import aggregate_sweep

LEDGER_NAME = "published.json"
_GRID_RE = re.compile(r"(\d+x\d+)")

# Uploaded verbatim from <sweep>/ and <sweep>/analysis/.
_ROOT_FILES = ("index.json", "manifest.json")
_ANALYSIS_GLOBS = ("*.csv", "*.tex")

# Columns a --drop-solver filter has to look at, per table.
_SOLVER_COLUMNS = ("solver_name", "solver_a", "solver_b")


def _load_hf():
    """Import huggingface_hub lazily so --dry-run works without the optional
    dependency installed."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return None
    return {
        "HfApi": HfApi,
        "hf_hub_download": hf_hub_download,
        "EntryNotFoundError": EntryNotFoundError,
        "RepositoryNotFoundError": RepositoryNotFoundError,
    }


def _sweep_meta(sweep_dir: Path) -> Dict:
    index = json.loads((sweep_dir / "index.json").read_text(encoding="utf-8"))
    manifest = json.loads((sweep_dir / "manifest.json").read_text(encoding="utf-8"))
    instances = sorted({e["instance"] for e in index})
    grid_sizes = sorted({m.group(1) for i in instances if (m := _GRID_RE.search(i))})
    completed = [e for e in index if not e.get("dry_run") and e.get("benchmark_json")]
    git = manifest.get("git") or {}
    return {
        "sweep_id": sweep_dir.name,
        "commit": git.get("commit"),
        "branch": git.get("branch"),
        "dirty": git.get("dirty"),
        "start_time": manifest.get("start_time"),
        "end_time": manifest.get("end_time"),
        "n_completed": len(completed),
        "solvers": sorted({e["solver"] for e in index}),
        "problems": sorted({e["problem"] for e in index}),
        "grid_sizes": grid_sizes,
    }


def _ensure_csvs(sweep_dir: Path, reaggregate: bool) -> None:
    runs_csv = sweep_dir / "analysis" / "runs_long.csv"
    if runs_csv.exists() and not reaggregate:
        return
    action = "re-aggregating" if runs_csv.exists() else "aggregating (no CSVs yet)"
    print(f"[publish] {sweep_dir.name}: {action}")
    aggregate_sweep(str(sweep_dir))  # writes <sweep_dir>/analysis/*.csv


def _filter_solver_rows(csv_path: Path, drop: Set[str]) -> None:
    df = pd.read_csv(csv_path)
    cols = [c for c in _SOLVER_COLUMNS if c in df.columns]
    if not cols:
        return
    keep = pd.Series(True, index=df.index)
    for col in cols:
        keep &= ~df[col].isin(drop)
    df[keep].to_csv(csv_path, index=False)


def _stage_sweep(sweep_dir: Path, stage: Path, drop_solvers: Set[str]) -> List[str]:
    """Copy the publishable subset of sweep_dir into stage/, optionally
    filtering solvers out of the CSVs. Returns the relative paths staged."""
    staged: List[str] = []
    for name in _ROOT_FILES:
        shutil.copy2(sweep_dir / name, stage / name)
        staged.append(name)

    src_analysis = sweep_dir / "analysis"
    dst_analysis = stage / "analysis"
    dst_analysis.mkdir(parents=True, exist_ok=True)
    for pattern in _ANALYSIS_GLOBS:
        for path in sorted(src_analysis.glob(pattern)):
            target = dst_analysis / path.name
            shutil.copy2(path, target)
            if drop_solvers and path.suffix == ".csv":
                _filter_solver_rows(target, drop_solvers)
            staged.append(f"analysis/{path.name}")
    return staged


def _validate(sweep_dir: Path) -> Optional[str]:
    if not sweep_dir.is_dir():
        return f"{sweep_dir} is not a directory"
    for name in _ROOT_FILES:
        if not (sweep_dir / name).exists():
            return f"{sweep_dir} has no {name}"
    return None


def _load_ledger(hf, api, repo: str, revision: str) -> Dict[str, Dict]:
    try:
        path = hf["hf_hub_download"](
            repo_id=repo, repo_type="dataset", filename=LEDGER_NAME, revision=revision
        )
    except hf["EntryNotFoundError"]:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dir_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB"):
        if total < 1024:
            return f"{total:.0f} {unit}"
        total /= 1024
    return f"{total:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish aggregated benchmark sweeps to a Hugging Face dataset."
    )
    parser.add_argument(
        "sweep_dirs", nargs="+", type=Path, help="Sweep dirs to publish."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Target dataset repo id, e.g. user/Spooky-benchmark.",
    )
    parser.add_argument(
        "--branch", default="main", help="Dataset repo branch (default: main)."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete any other published sweeps sharing this sweep's git commit "
        "and stand this one in their place (default: publish alongside, warn).",
    )
    parser.add_argument(
        "--drop-solver",
        action="append",
        default=[],
        metavar="NAME",
        help="Omit this solver from every CSV before upload (repeatable).",
    )
    parser.add_argument(
        "--reaggregate",
        action="store_true",
        help="Re-run aggregate_sweep() even if analysis/*.csv already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage everything and print what would upload, without touching the repo.",
    )
    parser.add_argument("--message", help="Commit message (default: auto).")
    parser.add_argument("--token", help="HF token (else HF_TOKEN env / cached login).")
    args = parser.parse_args()

    drop_solvers: Set[str] = set(args.drop_solver)

    metas: List[Dict] = []
    for sweep_dir in args.sweep_dirs:
        sweep_dir = sweep_dir.resolve()
        problem = _validate(sweep_dir)
        if problem:
            parser.error(problem)
        _ensure_csvs(sweep_dir, args.reaggregate)
        meta = _sweep_meta(sweep_dir)
        meta["_dir"] = sweep_dir
        if not meta["commit"]:
            print(
                f"[publish] warning: {meta['sweep_id']} has no git commit in its "
                "manifest — the one-per-commit check can't apply to it."
            )
        if meta["dirty"]:
            print(
                f"[publish] warning: {meta['sweep_id']} ran on a dirty tree "
                f"(commit {(meta['commit'] or '?')[:8]}); its code state isn't "
                "fully captured by the commit."
            )
        metas.append(meta)

    hf = _load_hf()
    if not args.dry_run and hf is None:
        parser.error(
            'huggingface_hub is not installed — `pip install -e ".[publish]"` '
            "or run with --dry-run."
        )

    api = None
    ledger: Dict[str, Dict] = {}
    if not args.dry_run:
        api = hf["HfApi"](token=args.token or None)
        try:
            api.repo_info(repo_id=args.repo, repo_type="dataset")
        except hf["RepositoryNotFoundError"]:
            parser.error(
                f"dataset repo '{args.repo}' not found (or the token can't see it). "
                "Create it first at https://huggingface.co/new-dataset."
            )
        ledger = _load_ledger(hf, api, args.repo, args.branch)

    published = []
    for meta in metas:
        sweep_id, commit = meta["sweep_id"], meta["commit"]

        # Other already-published sweeps on the same commit (a prior run in
        # this same invocation counts, via the ledger updates below).
        clashes = {
            sid: e
            for sid, e in ledger.items()
            if commit and e.get("commit") == commit and sid != sweep_id
        }
        if clashes and not args.replace:
            print(
                f"[publish] note: commit {commit[:8]} already has "
                f"{', '.join(clashes)} — publishing {sweep_id} alongside "
                "(--replace to swap instead)."
            )

        with tempfile.TemporaryDirectory(prefix=f"publish-{sweep_id}-") as tmp:
            stage = Path(tmp)
            staged = _stage_sweep(meta["_dir"], stage, drop_solvers)
            path_in_repo = f"sweeps/{sweep_id}"

            if args.dry_run:
                print(
                    f"[dry-run] {sweep_id} -> {args.repo}:{path_in_repo}/ ({_dir_size(stage)})"
                )
                for rel in staged:
                    print(f"          {rel}")
                if clashes and args.replace:
                    print(f"          would --replace (delete): {', '.join(clashes)}")
                continue

            if args.replace:
                for old_id in clashes:
                    print(f"[publish] --replace: deleting sweeps/{old_id}")
                    api.delete_folder(
                        path_in_repo=f"sweeps/{old_id}",
                        repo_id=args.repo,
                        repo_type="dataset",
                        revision=args.branch,
                        commit_message=f"Replace {old_id} (commit {commit[:8]})",
                    )
                    ledger.pop(old_id, None)

            message = args.message or (
                f"Publish {sweep_id} (spooky@{(commit or '?')[:8]}, "
                f"{meta['n_completed']} runs)"
            )
            api.upload_folder(
                folder_path=str(stage),
                path_in_repo=path_in_repo,
                repo_id=args.repo,
                repo_type="dataset",
                revision=args.branch,
                commit_message=message,
                delete_patterns=["**"],  # prune anything stale under this sweep's dir
            )
            ledger[sweep_id] = {
                "commit": commit,
                "branch": meta["branch"],
                "published_at": datetime.now(timezone.utc).isoformat(),
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "n_completed": meta["n_completed"],
                "solvers": meta["solvers"],
                "problems": meta["problems"],
                "grid_sizes": meta["grid_sizes"],
                "dropped_solvers": sorted(drop_solvers) or None,
            }
            published.append(sweep_id)
            print(f"[publish] uploaded {sweep_id} -> {args.repo}:{path_in_repo}/")

    if published:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
            ledger_path = f.name
        api.upload_file(
            path_or_fileobj=ledger_path,
            path_in_repo=LEDGER_NAME,
            repo_id=args.repo,
            repo_type="dataset",
            revision=args.branch,
            commit_message=f"Update {LEDGER_NAME} ({', '.join(published)})",
        )
        Path(ledger_path).unlink(missing_ok=True)
        print(f"[publish] ledger now lists {len(ledger)} sweep(s)")
    elif not args.dry_run:
        print("[publish] nothing uploaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
