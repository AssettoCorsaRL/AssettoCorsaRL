import wandb
import sys
import statistics
from collections import defaultdict


def fetch_run_history(run_path: str, num_snapshots: int = 50) -> str:
    api = wandb.Api()

    parts = run_path.strip().split("/")
    if len(parts) == 2:
        runs = api.runs(run_path, per_page=1, order="-created_at")
        run = next(iter(runs))
    elif len(parts) == 3:
        run = api.run(run_path)
    else:
        raise ValueError(
            f"Invalid run path: '{run_path}'. Expected 'entity/project' or 'entity/project/run_id'."
        )

    output = []
    output.append("=" * 70)
    output.append("WANDB RUN REPORT (LLM-OPTIMIZED)")
    output.append("=" * 70)
    output.append(f"Run ID     : {run.id}")
    output.append(f"Name       : {run.name}")
    output.append(f"Path       : {'/'.join(run.path)}")
    output.append(f"State      : {run.state}")
    output.append(f"Created At : {run.created_at}")
    output.append(f"URL        : {run.url}")
    output.append("")

    output.append("── HYPERPARAMETERS (config) ──")
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    if config:
        for k, v in sorted(config.items()):
            output.append(f"  {k}: {v}")
    else:
        output.append("  (none logged)")
    output.append("")

    output.append("── FINAL SUMMARY METRICS ──")
    summary = {k: v for k, v in run.summary_metrics.items() if not k.startswith("_")}
    if summary:
        for k, v in sorted(summary.items()):
            output.append(f"  {k}: {v}")
    else:
        output.append("  (none logged)")
    output.append("")

    print("Scanning history (single pass)...", file=sys.stderr)

    key_data = defaultdict(list)
    all_rows_raw = []  # (step, row) for every row — we subsample after

    for row in run.scan_history(page_size=10_000):
        step = row.get("_step", 0)
        all_rows_raw.append((step, row))
        for k, v in row.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float)) and v == v:  # skip NaN
                key_data[k].append((step, v))

    if not all_rows_raw:
        output.append("No history found.")
        return "\n".join(output)

    all_rows_raw.sort(key=lambda x: x[0])
    total_steps = all_rows_raw[-1][0] + 1

    seen_keys = sorted(key_data.keys())
    output.append(f"Available metric keys: {seen_keys}")
    output.append(f"Total logged steps: {total_steps}")
    output.append("")

    n = len(all_rows_raw)
    if n <= num_snapshots:
        snapshot_rows = all_rows_raw
    else:
        step_size = (n - 1) / (num_snapshots - 1)
        indices = set(round(i * step_size) for i in range(num_snapshots))
        snapshot_rows = [all_rows_raw[i] for i in sorted(indices)]

    # ── Per-metric statistics ─────────────────────────────────────────────────
    output.append("── PER-METRIC STATISTICS (all steps) ──")
    for key in seen_keys:
        values = [v for _, v in key_data[key]]
        steps = [s for s, _ in key_data[key]]
        mn, mx = min(values), max(values)
        mean = statistics.mean(values)
        first_val, last_val = values[0], values[-1]
        trend = (
            "improving"
            if last_val > first_val
            else "declining" if last_val < first_val else "flat"
        )
        output.append(
            f"  {key}:"
            f" min={mn:.4g}, max={mx:.4g}, mean={mean:.4g},"
            f" first={first_val:.4g} (step {steps[0]}),"
            f" last={last_val:.4g} (step {steps[-1]}),"
            f" trend={trend},"
            f" n_logged={len(values)}"
        )
    output.append("")

    output.append(
        f"── SNAPSHOTS ({len(snapshot_rows)} snapshots across {total_steps} total steps) ──"
    )
    output.append(f"  Format: [snap XX/{len(snapshot_rows)} | step=N] key=value ...")
    output.append("")

    for snap_num, (step, row) in enumerate(snapshot_rows, 1):
        metrics = {
            k: (round(v, 5) if isinstance(v, float) else v)
            for k, v in sorted(row.items())
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }
        metrics_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
        output.append(
            f"  [snap {snap_num:02d}/{len(snapshot_rows)} | step={step}]  {metrics_str}"
        )

    output.append("")
    output.append("=" * 70)
    output.append("END OF REPORT")
    output.append("=" * 70)

    return "\n".join(output)


if __name__ == "__main__":
    run_path = (
        sys.argv[1] if len(sys.argv) > 1 else "ved-patel226-/AssetoCorsaRL-AssettoCorsa"
    )
    num_snapshots = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    report = fetch_run_history(run_path, num_snapshots)

    output_file = "training_history.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {output_file}", file=sys.stderr)
    print(report)
