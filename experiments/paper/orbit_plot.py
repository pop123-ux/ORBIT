#!/usr/bin/env python3
"""Generate publication-oriented ORBIT figures from the frozen evidence hierarchy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from orbit_paper_artifacts import build as freeze_evidence
from orbit_paper_artifacts import grouped, load_jsonl


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Match the paper's LaTeX typography instead of Matplotlib's default sans face.
        "font.family": "cmr10",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    }
)


def save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def plot_orbit_overview(out: Path) -> None:
    """Compact vector schematic of the ORBIT update path."""
    fig, ax = plt.subplots(figsize=(10.8, 3.35))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    def box(x, y, w, h, title, subtitle=None, *, lw=1.2):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            linewidth=lw,
            facecolor="white",
            edgecolor="0.25",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            linespacing=1.0,
        )
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=7.6,
                color="0.30",
                linespacing=1.1,
            )
        return (x, y, w, h)

    def arrow(a, b, *, yfrac=0.5):
        x1=a[0]+a[2]; y1=a[1]+a[3]*yfrac
        x2=b[0]; y2=b[1]+b[3]*yfrac
        ax.annotate("", xy=(x2-0.08,y2), xytext=(x1+0.08,y1),
                    arrowprops=dict(arrowstyle="->", lw=1.25, color="0.35"))

    b1 = box(0.20, 1.95, 1.65, 1.25, "Muon candidate", "Q/K gradients\n+ momentum")
    b2 = box(2.20, 1.95, 1.70, 1.25, "Q/K statistics", "EMA $2\\times2$\ncovariances")
    b3 = box(4.25, 1.95, 1.80, 1.25, "RoPE transport", "relative offsets\n$\\Delta$")
    b4 = box(6.40, 1.95, 1.75, 1.25, "Local metric", "per-frequency\n$2\\times2$")
    b5 = box(8.50, 1.95, 1.75, 1.25, "Precondition", r"analytic $M^{-1/2}$")
    b6 = box(10.60, 1.95, 1.40, 1.25, "Restore", "joint Q+K\nnorm")
    for a,b in zip((b1,b2,b3,b4,b5),(b2,b3,b4,b5,b6)):
        arrow(a,b)

    ax.text(6.10,3.62,"ORBIT: RoPE-conditioned function-space update",
            ha="center",va="center",fontsize=11.1,fontweight="bold")
    ax.text(
        6.10,
        0.42,
        r"Other hidden matrices $\rightarrow$ standard Muon   $\bullet$   "
        r"embeddings / biases / norms $\rightarrow$ auxiliary AdamW"
        "\nInference graph unchanged",
        ha="center",
        va="center",
        fontsize=8.3,
        color="0.28",
        linespacing=1.2,
    )
    ax.annotate("", xy=(11.30, 1.68), xytext=(11.30, 1.14),
                arrowprops=dict(arrowstyle="->", lw=1.15, color="0.35"))
    ax.text(11.30, 0.90, "parameter update", ha="center", va="center", fontsize=8.4)
    save(fig, out, "orbit_overview")


def plot_matched_effect(results: dict, out: Path) -> None:
    effect = results["matched_confirmation"]["orbit_vs_muon"]
    per_seed = {int(k): float(v) for k, v in effect["per_seed"].items()}
    seeds = sorted(per_seed)
    vals = [per_seed[s] for s in seeds]
    mean = float(effect["mean_delta"])
    lo, hi = effect["ci95"]

    x = list(range(len(seeds)))
    mean_x = len(seeds) + 0.75
    fig, ax = plt.subplots(figsize=(7.0, 3.60))
    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45")
    ax.scatter(x, vals, zorder=3, s=34)
    ax.errorbar(
        [mean_x], [mean],
        yerr=[[mean - lo], [hi - mean]],
        fmt="D", capsize=5, markersize=6.5, linewidth=1.8,
        label=r"paired mean $\pm$ 95% CI",
    )
    ax.set_xticks(x + [mean_x], [str(s) for s in seeds] + ["Mean"])
    ax.set_xlim(-0.55, mean_x + 0.65)
    ax.set_xlabel("Held-out run")
    ax.set_ylabel(r"$\Delta$ validation loss (ORBIT $-$ Muon)")
    ax.set_title("Matched ORBIT-Muon effect across held-out runs")
    ax.legend(
        frameon=False,
        loc="upper left",
        borderaxespad=0.25,
        handletextpad=0.6,
    )
    save(fig, out, "matched_effect")


def plot_matched_pairs(rows: list[dict], out: Path) -> None:
    by = grouped(rows, "optimizer")
    mu = {int(x["seed"]): float(x["val_loss"]) for x in by["muon"]}
    orb = {int(x["seed"]): float(x["val_loss"]) for x in by["orbit"]}
    seeds = sorted(set(mu) & set(orb))

    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    for seed in seeds:
        ax.plot([0, 1], [mu[seed], orb[seed]], marker="o", alpha=0.42, linewidth=1.0)
    ax.scatter(
        [0, 1],
        [statistics.fmean(mu.values()), statistics.fmean(orb.values())],
        marker="D",
        s=58,
        zorder=4,
        label="mean",
    )
    ax.set_xticks([0, 1], ["Muon", "ORBIT"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Validation loss")
    ax.set_title("Matched confirmation: paired absolute losses")
    ax.legend(frameon=False)
    save(fig, out, "matched_seed_pairs")


def plot_xconfig(results: dict, out: Path) -> None:
    sm = results["cross_configuration_isolation"]["summary"]
    x = [0, 1]
    labels = ["Muon-selected config", "ORBIT-selected config"]
    series = {
        "Muon": ["muon_at_muon_config", "muon_at_orbit_config"],
        "ORBIT": ["orbit_at_muon_config", "orbit_at_orbit_config"],
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    for name, keys in series.items():
        means = [sm[k]["mean_val_loss"] for k in keys]
        sds = [sm[k]["sd_val_loss"] for k in keys]
        ax.errorbar(x, means, yerr=sds, marker="o", capsize=4, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation loss")
    ax.set_title(r"Optimizer $\times$ hyperparameter-recipe cross-over")
    ax.legend(frameon=False)
    save(fig, out, "xconfig_interaction")


def plot_ablation_effects(results: dict, out: Path) -> None:
    ab = results["mechanism_ablation"]
    comparisons = [
        ("Identity control", ab["orbit_vs_identity"]),
        ("No RoPE rotation", ab["orbit_vs_norope"]),
        ("Diagonal metric", ab["orbit_vs_diag"]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    y = list(range(len(comparisons)))
    for yi, (label, effect) in zip(y, comparisons):
        mean = float(effect["mean_delta"])
        lo, hi = effect["ci95"]
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            capsize=4,
        )
        ax.text(
            hi + 0.00025,
            yi,
            f"{effect['a_wins']}/{effect['n']}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y, [x[0] for x in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel(r"Validation-loss difference (full ORBIT $-$ control)")
    ax.set_title("Mechanism ablation: paired effects with 95% CIs")
    save(fig, out, "ablation_effects")



def plot_mechanism_summary(results: dict, out: Path) -> None:
    """Two-panel summary of recipe interaction and mechanism ablations."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.55), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    sm = results["cross_configuration_isolation"]["summary"]
    x = [0, 1]
    labels = ["Muon recipe", "ORBIT recipe"]
    series = {
        "Muon": ["muon_at_muon_config", "muon_at_orbit_config"],
        "ORBIT": ["orbit_at_muon_config", "orbit_at_orbit_config"],
    }
    for name, keys in series.items():
        means = [sm[k]["mean_val_loss"] for k in keys]
        sds = [sm[k]["sd_val_loss"] for k in keys]
        ax.errorbar(x, means, yerr=sds, marker="o", capsize=4, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation loss")
    ax.set_title(r"Optimizer $\times$ recipe")
    ax.legend(frameon=False)
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1]
    ab = results["mechanism_ablation"]
    comparisons = [
        ("Identity", ab["orbit_vs_identity"]),
        ("No-RoPE", ab["orbit_vs_norope"]),
        ("Diagonal", ab["orbit_vs_diag"]),
    ]
    ax.axvline(0.0, linewidth=1.0, linestyle="--", color="0.45")
    for yi, (label, effect) in enumerate(comparisons):
        mean = float(effect["mean_delta"])
        lo, hi = effect["ci95"]
        ax.errorbar(
            mean, yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o", capsize=4,
        )
    ax.set_yticks(range(len(comparisons)), [x[0] for x in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ validation loss (ORBIT $-$ control)")
    ax.set_title("Mechanism ablations")
    ax.text(-0.14, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)
    save(fig, out, "mechanism_summary")


def plot_transfer_summary(horizon_rows: list[dict], scale_rows: list[dict], out: Path) -> None:
    """Compact two-panel transfer figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.45), gridspec_kw={"wspace": 0.30})
    order = ["muon", "normuon", "astro_v2", "orbit"]
    display = {"muon": "Muon", "normuon": "NorMuon", "astro_v2": "ASTRO", "orbit": "ORBIT"}

    for ax, rows, title, panel in (
        (axes[0], horizon_rows, "124M / 2700 steps", "a"),
        (axes[1], scale_rows, "355M / 900 steps", "b"),
    ):
        by = grouped(rows, "optimizer")
        for xi, name in enumerate(order):
            vals = [float(x["val_loss"]) for x in sorted(by[name], key=lambda r: r["seed"])]
            offsets = [-0.055, 0.055] if len(vals) == 2 else [0.0] * len(vals)
            ax.scatter([xi + o for o in offsets], vals, s=32, zorder=3)
            mean = statistics.fmean(vals)
            ax.plot([xi - 0.14, xi + 0.14], [mean, mean], linewidth=2.0)
        ax.set_xticks(range(len(order)), [display[x] for x in order])
        ax.set_title(title)
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Validation loss")
    save(fig, out, "transfer_summary")



def plot_transfer(rows: list[dict], out: Path, *, stem: str, title: str) -> None:
    by = grouped(rows, "optimizer")
    order = ["muon", "normuon", "astro_v2", "orbit"]
    display = {"muon": "Muon", "normuon": "NorMuon", "astro_v2": "ASTRO", "orbit": "ORBIT"}

    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    for xi, name in enumerate(order):
        vals = [float(x["val_loss"]) for x in sorted(by[name], key=lambda r: r["seed"])]
        offsets = [-0.045, 0.045] if len(vals) == 2 else [0.0] * len(vals)
        ax.scatter([xi + o for o in offsets], vals, s=42, zorder=3)
        mean = statistics.fmean(vals)
        ax.plot([xi - 0.15, xi + 0.15], [mean, mean], linewidth=2.4)
    ax.set_xticks(range(len(order)), [display[x] for x in order])
    ax.set_ylabel("Validation loss")
    ax.set_title(title)
    save(fig, out, stem)


def plot_broad_confirmation(rows: list[dict], out: Path) -> None:
    by = grouped(rows, "optimizer")
    display = {
        "adamw": "AdamW",
        "adamuon_ref": "AdaMuon",
        "muon": "Muon",
        "normuon": "NorMuon",
        "astro_v2": "ASTRO",
        "orbit": "ORBIT",
    }
    stats = []
    for name, items in by.items():
        vals = [float(x["val_loss"]) for x in items]
        mean, sd = mean_sd(vals)
        stats.append((mean, sd, name))
    stats.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for yi, (mean, sd, name) in enumerate(stats):
        ax.errorbar(mean, yi, xerr=sd, fmt="o", capsize=4)
    ax.set_yticks(range(len(stats)), [display.get(x[2], x[2]) for x in stats])
    ax.set_xlabel(r"Validation loss (mean $\pm$ SD)")
    ax.set_title("Broad 124M optimizer context")
    save(fig, out, "broad_confirmation")


def plot_training_curves(rows: list[dict], out: Path) -> None:
    curves = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for point in row.get("curve", []):
            curves[row["optimizer"]][int(point["step"])].append(float(point["train_loss"]))

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for name in ("muon", "orbit"):
        steps = sorted(curves[name])
        if not steps:
            continue
        means = [statistics.fmean(curves[name][s]) for s in steps]
        sds = [
            statistics.stdev(curves[name][s]) if len(curves[name][s]) > 1 else 0.0
            for s in steps
        ]
        (line,) = ax.plot(steps, means, label=name.upper() if name == "orbit" else "Muon")
        color = line.get_color()
        lower = [m - s for m, s in zip(means, sds)]
        upper = [m + s for m, s in zip(means, sds)]
        ax.fill_between(steps, lower, upper, alpha=0.12, color=color)
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Training loss")
    ax.set_title("Matched-confirmation training trajectories")
    ax.legend(frameon=False)
    save(fig, out, "matched_training_curves")


def plot_efficiency(rows: list[dict], out: Path) -> None:
    by = grouped(rows, "optimizer")
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    for name in ("muon", "orbit"):
        xs = [float(x["seconds"]) / 60 for x in by[name]]
        ys = [float(x["val_loss"]) for x in by[name]]
        ax.scatter(xs, ys, label=name.upper() if name == "orbit" else "Muon", alpha=0.82)
    ax.set_xlabel("Wall-clock minutes")
    ax.set_ylabel("Validation loss")
    ax.set_title("Matched confirmation: quality vs wall-clock")
    ax.legend(frameon=False)
    save(fig, out, "matched_efficiency")


def plot_metric_diagnostics(rows: list[dict], out: Path) -> None:
    orbit = [
        x for x in rows
        if x["optimizer"] == "orbit" and x.get("optimizer_diagnostics")
    ]
    if not orbit:
        return
    specs = [
        ("metric_condition_q", "Q metric condition"),
        ("metric_condition_k", "K metric condition"),
        ("joint_restore", "Joint restore"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for xi, (key, label) in enumerate(specs):
        vals = [float(x["optimizer_diagnostics"][key]) for x in orbit]
        offsets = [
            (j - (len(vals) - 1) / 2) * 0.012
            for j in range(len(vals))
        ]
        ax.scatter([xi + o for o in offsets], vals, alpha=0.72, s=28)
        mean = statistics.fmean(vals)
        ax.plot([xi - 0.14, xi + 0.14], [mean, mean], linewidth=2.5)
    ax.set_xticks(range(len(specs)), [x[1] for x in specs])
    ax.set_ylabel("Recorded diagnostic value")
    ax.set_title("ORBIT metric diagnostics across matched held-out seeds")
    save(fig, out, "metric_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    results, manifest = freeze_evidence(
        args.work_dir, allow_incomplete=args.allow_incomplete
    )
    merged = args.work_dir / "merged"
    out = args.out_dir or (args.work_dir / "paper_artifacts" / "figures")
    out.mkdir(parents=True, exist_ok=True)

    plot_orbit_overview(out)

    if "matched_confirmation" in results:
        plot_matched_effect(results, out)

    if (
        "cross_configuration_isolation" in results
        and "mechanism_ablation" in results
    ):
        plot_mechanism_summary(results, out)

    if (
        "long_horizon_transfer" in results
        and "scale_transfer" in results
    ):
        plot_transfer_summary(
            load_jsonl(merged / "horizon_with_astro.jsonl"),
            load_jsonl(merged / "scale_with_astro.jsonl"),
            out,
        )

    if "broad_independently_tuned_context" in results:
        plot_broad_confirmation(load_jsonl(merged / "confirm.jsonl"), out)

    print(
        json.dumps(
            {
                "figures": str(out),
                "status": manifest["status"],
                "generated": sorted(p.name for p in out.glob("*.pdf")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
