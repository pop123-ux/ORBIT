#!/usr/bin/env python3
"""Generate ORBIT paper tables, macros, claim ledger, figures, and manuscript PDF."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from orbit_paper_artifacts import build as freeze_evidence
import orbit_plot

ROW_END = r" \\"


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def ci(effect: dict) -> str:
    lo, hi = effect["ci95"]
    return f"[{lo:.4f}, {hi:.4f}]"


def display_name(name: str) -> str:
    return {
        "adamw": "AdamW",
        "adamuon_ref": "AdaMuon",
        "muon": "Muon",
        "normuon": "NorMuon",
        "astro_v2": "ASTRO",
        "orbit": "ORBIT",
        "orbit_identity": "Identity",
        "orbit_norope": "No-RoPE",
        "orbit_diag": "Diagonal",
    }.get(name, name)


def write_macros(results: dict, generated: Path) -> None:
    macro_names = [
        "MatchedMuonLoss", "MatchedOrbitLoss", "MatchedDelta", "MatchedCILow",
        "MatchedCIHigh", "MatchedWins", "MatchedRuntimeOverhead",
        "MatchedMemoryOverhead", "XConfigMechanismMuon",
        "XConfigMechanismMuonWins", "XConfigMechanismOrbit",
        "XConfigMechanismOrbitWins", "XConfigRecipeMuon", "XConfigRecipeOrbit",
        "AblIdentityDelta", "AblIdentityCILow", "AblIdentityCIHigh",
        "AblIdentityWins", "AblNoRoPEDelta", "AblNoRoPECILow",
        "AblNoRoPECIHigh", "AblNoRoPEWins", "AblDiagDelta", "AblDiagCILow",
        "AblDiagCIHigh", "AblDiagWins", "HorizonVsAstro", "HorizonVsMuon",
        "HorizonVsNorMuon", "ScaleVsAstro", "ScaleVsMuon", "ScaleVsNorMuon",
        "BroadAstroLoss", "BroadOrbitLoss", "BroadMuonLoss", "BroadNorMuonLoss",
    ]
    lines = ["% AUTO-GENERATED. Do not edit."]
    lines.extend(f"\\newcommand{{\\{name}}}{{--}}" for name in macro_names)

    mc = results.get("matched_confirmation")
    if mc:
        sm = mc["summary"]
        eff = mc["orbit_vs_muon"]
        lo, hi = eff["ci95"]
        lines += [
            f"\\renewcommand{{\\MatchedMuonLoss}}{{{sm['muon']['mean_val_loss']:.4f}}}",
            f"\\renewcommand{{\\MatchedOrbitLoss}}{{{sm['orbit']['mean_val_loss']:.4f}}}",
            f"\\renewcommand{{\\MatchedDelta}}{{{eff['mean_delta']:.4f}}}",
            f"\\renewcommand{{\\MatchedCILow}}{{{lo:.4f}}}",
            f"\\renewcommand{{\\MatchedCIHigh}}{{{hi:.4f}}}",
            f"\\renewcommand{{\\MatchedWins}}{{{eff['a_wins']}/{eff['n']}}}",
            f"\\renewcommand{{\\MatchedRuntimeOverhead}}{{{100*mc['runtime_overhead_fraction']:.1f}\\%}}",
            f"\\renewcommand{{\\MatchedMemoryOverhead}}{{{100*mc['memory_overhead_fraction']:.1f}\\%}}",
        ]

    xc = results.get("cross_configuration_isolation")
    if xc:
        m_mu = xc["mechanism_at_muon_config"]
        m_or = xc["mechanism_at_orbit_config"]
        r_mu = xc["recipe_effect_on_muon"]
        r_or = xc["recipe_effect_on_orbit"]
        lines += [
            f"\\renewcommand{{\\XConfigMechanismMuon}}{{{m_mu['mean_delta']:.4f}}}",
            f"\\renewcommand{{\\XConfigMechanismMuonWins}}{{{m_mu['a_wins']}/{m_mu['n']}}}",
            f"\\renewcommand{{\\XConfigMechanismOrbit}}{{{m_or['mean_delta']:.4f}}}",
            f"\\renewcommand{{\\XConfigMechanismOrbitWins}}{{{m_or['a_wins']}/{m_or['n']}}}",
            f"\\renewcommand{{\\XConfigRecipeMuon}}{{{r_mu['mean_delta']:.4f}}}",
            f"\\renewcommand{{\\XConfigRecipeOrbit}}{{{r_or['mean_delta']:.4f}}}",
        ]

    ab = results.get("mechanism_ablation")
    if ab:
        for stem, key in (
            ("Identity", "orbit_vs_identity"),
            ("NoRoPE", "orbit_vs_norope"),
            ("Diag", "orbit_vs_diag"),
        ):
            eff = ab[key]
            lo, hi = eff["ci95"]
            lines += [
                f"\\renewcommand{{\\Abl{stem}Delta}}{{{eff['mean_delta']:.4f}}}",
                f"\\renewcommand{{\\Abl{stem}CILow}}{{{lo:.4f}}}",
                f"\\renewcommand{{\\Abl{stem}CIHigh}}{{{hi:.4f}}}",
                f"\\renewcommand{{\\Abl{stem}Wins}}{{{eff['a_wins']}/{eff['n']}}}",
            ]

    horizon = results.get("long_horizon_transfer")
    if horizon:
        for stem, key in (
            ("Astro", "orbit_vs_astro_v2"),
            ("Muon", "orbit_vs_muon"),
            ("NorMuon", "orbit_vs_normuon"),
        ):
            eff = horizon[key]
            lines.append(
                f"\\renewcommand{{\\HorizonVs{stem}}}{{{eff['mean_delta']:.4f}}}"
            )

    scale = results.get("scale_transfer")
    if scale:
        for stem, key in (
            ("Astro", "orbit_vs_astro_v2"),
            ("Muon", "orbit_vs_muon"),
            ("NorMuon", "orbit_vs_normuon"),
        ):
            eff = scale[key]
            lines.append(
                f"\\renewcommand{{\\ScaleVs{stem}}}{{{eff['mean_delta']:.4f}}}"
            )

    broad = results.get("broad_independently_tuned_context")
    if broad:
        sm = broad["summary"]
        lines += [
            f"\\renewcommand{{\\BroadAstroLoss}}{{{sm['astro_v2']['mean_val_loss']:.4f}}}",
            f"\\renewcommand{{\\BroadOrbitLoss}}{{{sm['orbit']['mean_val_loss']:.4f}}}",
            f"\\renewcommand{{\\BroadMuonLoss}}{{{sm['muon']['mean_val_loss']:.4f}}}",
            f"\\renewcommand{{\\BroadNorMuonLoss}}{{{sm['normuon']['mean_val_loss']:.4f}}}",
        ]

    generated.joinpath("macros.tex").write_text("\n".join(lines) + "\n")


def write_matched_table(results: dict, generated: Path) -> None:
    mc = results.get("matched_confirmation")
    if not mc:
        generated.joinpath("table_matched.tex").write_text("% unavailable\n")
        return
    sm = mc["summary"]
    eff = mc["orbit_vs_muon"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Primary matched-hyperparameter confirmation across 10 held-out paired runs. "
        "Both optimizers use the same selected configuration. Lower validation loss is better.}",
        "\\label{tab:matched}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Optimizer & $n$ & Mean loss & SD & Mean min. & Peak GB" + ROW_END,
        "\\midrule",
    ]
    for name in ("muon", "orbit"):
        row = sm[name]
        lines.append(
            f"{display_name(name)} & {row['n']} & {row['mean_val_loss']:.4f} & "
            f"{row['sd_val_loss']:.4f} & {row['mean_seconds']/60:.1f} & "
            f"{row['mean_peak_cuda_gb']:.3f}" + ROW_END
        )
    lines += [
        "\\midrule",
        f"ORBIT $-$ Muon & {eff['n']} & {eff['mean_delta']:.4f} & "
        f"{eff['sd_delta']:.4f} & \\multicolumn{{2}}{{c}}{{95\\% CI {ci(eff)}, "
        f"wins {eff['a_wins']}/{eff['n']}}}" + ROW_END,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    generated.joinpath("table_matched.tex").write_text("\n".join(lines) + "\n")


def write_xconfig_table(results: dict, generated: Path) -> None:
    x = results.get("cross_configuration_isolation")
    if not x:
        generated.joinpath("table_xconfig.tex").write_text("% unavailable\n")
        return
    sm = x["summary"]
    muon_recipe = x["mechanism_at_muon_config"]
    orbit_recipe = x["mechanism_at_orbit_config"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-configuration analysis separating the update rule from the "
        "frozen training recipe. Entries are mean validation loss $\\pm$ one standard "
        "deviation across five paired runs. The final row reports the paired "
        "ORBIT-minus-Muon effect for each recipe.}",
        "\\label{tab:xconfig}",
        "\\small",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Frozen training recipe}" + ROW_END,
        "\\cmidrule(lr){2-3}",
        "Update rule & Muon-selected & ORBIT-selected" + ROW_END,
        "\\midrule",
        f"Muon & {sm['muon_at_muon_config']['mean_val_loss']:.4f} $\\pm$ "
        f"{sm['muon_at_muon_config']['sd_val_loss']:.4f} & "
        f"{sm['muon_at_orbit_config']['mean_val_loss']:.4f} $\\pm$ "
        f"{sm['muon_at_orbit_config']['sd_val_loss']:.4f}" + ROW_END,
        f"ORBIT & {sm['orbit_at_muon_config']['mean_val_loss']:.4f} $\\pm$ "
        f"{sm['orbit_at_muon_config']['sd_val_loss']:.4f} & "
        f"{sm['orbit_at_orbit_config']['mean_val_loss']:.4f} $\\pm$ "
        f"{sm['orbit_at_orbit_config']['sd_val_loss']:.4f}" + ROW_END,
        "\\midrule",
        f"ORBIT $-$ Muon (95\\% CI) & {muon_recipe['mean_delta']:.4f} "
        f"{ci(muon_recipe)} & {orbit_recipe['mean_delta']:.4f} {ci(orbit_recipe)}" + ROW_END,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    generated.joinpath("table_xconfig.tex").write_text("\n".join(lines) + "\n")


def write_ablation_table(results: dict, generated: Path) -> None:
    ab = results.get("mechanism_ablation")
    if not ab:
        generated.joinpath("table_ablation.tex").write_text("% unavailable\n")
        return
    sm = ab["summary"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Expanded 10-seed mechanism ablation under the frozen matched ORBIT "
        "configuration. $\\Delta$ is full ORBIT minus the control.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Control & Control loss & Mean $\\Delta$ & 95\\% CI & Wins" + ROW_END,
        "\\midrule",
    ]
    specs = [
        ("orbit_identity", "Identity", ab["orbit_vs_identity"]),
        ("orbit_norope", "No-RoPE", ab["orbit_vs_norope"]),
        ("orbit_diag", "Diagonal", ab["orbit_vs_diag"]),
    ]
    for key, label, effect in specs:
        lines.append(
            f"{label} & {sm[key]['mean_val_loss']:.4f} & {effect['mean_delta']:.4f} & "
            f"{ci(effect)} & {effect['a_wins']}/{effect['n']}" + ROW_END
        )
    lines += [
        "\\midrule",
        f"Full ORBIT & {sm['orbit']['mean_val_loss']:.4f} & -- & -- & --" + ROW_END,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    generated.joinpath("table_ablation.tex").write_text("\n".join(lines) + "\n")


def write_transfer_table(results: dict, generated: Path) -> None:
    horizon = results.get("long_horizon_transfer")
    scale = results.get("scale_transfer")
    if not horizon or not scale:
        generated.joinpath("table_transfer.tex").write_text("% unavailable\n")
        return
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Long-horizon and 355M transfer. These cells contain two seeds per "
        "optimizer and are treated as secondary transfer evidence rather than high-powered "
        "significance tests.}",
        "\\label{tab:transfer}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Setting & Optimizer & Mean loss & SD" + ROW_END,
        "\\midrule",
    ]
    for setting, block in (("124M / 2700", horizon), ("355M / 900", scale)):
        for name in ("muon", "normuon", "astro_v2", "orbit"):
            row = block["summary"][name]
            lines.append(
                f"{setting} & {display_name(name)} & {row['mean_val_loss']:.4f} & "
                f"{row['sd_val_loss']:.4f}" + ROW_END
            )
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    generated.joinpath("table_transfer.tex").write_text("\n".join(lines) + "\n")


def write_broad_table(results: dict, generated: Path) -> None:
    block = results.get("broad_independently_tuned_context")
    if not block:
        generated.joinpath("table_broad.tex").write_text("% unavailable\n")
        return
    sm = block["summary"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Broad independently tuned 124M confirmation. This table provides "
        "optimizer context but is not used as the primary causal estimate of the ORBIT "
        "mechanism because each optimizer carries its independently selected recipe.}",
        "\\label{tab:broad}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Optimizer & $n$ & Mean loss & SD & Mean min." + ROW_END,
        "\\midrule",
    ]
    for name, row in sorted(sm.items(), key=lambda kv: kv[1]["mean_val_loss"]):
        lines.append(
            f"{display_name(name)} & {row['n']} & {row['mean_val_loss']:.4f} & "
            f"{row['sd_val_loss']:.4f} & {row['mean_seconds']/60:.1f}" + ROW_END
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    generated.joinpath("table_broad.tex").write_text("\n".join(lines) + "\n")


def write_claim_ledger(results: dict, manifest: dict, artifact_dir: Path) -> None:
    lines = [
        "# ORBIT paper claim ledger",
        "",
        f"Evidence freeze status: **{manifest['status']}**",
        f"Core experiment digest: `{manifest['core_digest']}`",
        "",
        "## Primary supported claim",
    ]
    mc = results.get("matched_confirmation")
    if mc:
        e = mc["orbit_vs_muon"]
        lines += [
            f"- Under the same frozen hyperparameter configuration, ORBIT beats Muon on "
            f"{e['a_wins']}/{e['n']} held-out seeds.",
            f"- Paired mean validation-loss difference: **{e['mean_delta']:.6f}**; "
            f"95% CI **[{e['ci95'][0]:.6f}, {e['ci95'][1]:.6f}]**.",
            f"- Mean wall-clock overhead: **{100*mc['runtime_overhead_fraction']:.1f}%**; "
            f"peak-memory difference: **{100*mc['memory_overhead_fraction']:.1f}%**.",
        ]

    x = results.get("cross_configuration_isolation")
    if x:
        lines += [
            "",
            "## Hyperparameter-confound finding",
            f"- ORBIT vs Muon at Muon config: {x['mechanism_at_muon_config']['mean_delta']:.6f}.",
            f"- ORBIT vs Muon at ORBIT config: {x['mechanism_at_orbit_config']['mean_delta']:.6f}.",
            "- The original independently tuned gap must not be presented as a pure "
            "algorithmic effect; optimizer and recipe interact.",
        ]

    ab = results.get("mechanism_ablation")
    if ab:
        lines += [
            "",
            "## Mechanism claims",
            f"- Full vs identity: Delta={ab['orbit_vs_identity']['mean_delta']:.6f}, "
            f"wins {ab['orbit_vs_identity']['a_wins']}/{ab['orbit_vs_identity']['n']}.",
            f"- Full vs no-RoPE: Delta={ab['orbit_vs_norope']['mean_delta']:.6f}, "
            f"wins {ab['orbit_vs_norope']['a_wins']}/{ab['orbit_vs_norope']['n']}.",
            f"- Full vs diagonal: Delta={ab['orbit_vs_diag']['mean_delta']:.6f}, "
            f"95% CI {ab['orbit_vs_diag']['ci95']}.",
            "- Functional Q/K conditioning is supported; RoPE-aware conditioning has "
            "additional support; necessity of full off-diagonal coupling is not established.",
        ]

    horizon = results.get("long_horizon_transfer")
    scale = results.get("scale_transfer")
    if horizon and scale:
        lines += [
            "",
            "## Secondary transfer evidence",
            f"- 124M/2700: ORBIT - ASTRO = "
            f"{horizon['orbit_vs_astro_v2']['mean_delta']:.6f} over n=2 paired seeds.",
            f"- 355M/900: ORBIT - ASTRO = "
            f"{scale['orbit_vs_astro_v2']['mean_delta']:.6f}; one seed favors each method.",
            "- Treat both as transfer evidence, not high-powered significance tests.",
        ]

    lines += [
        "",
        "## Explicit non-claims",
        "- Do not claim the original ~0.14 ORBIT-vs-Muon gap is entirely algorithmic.",
        "- Do not claim off-diagonal 2x2 phase coupling is necessary.",
        "- Do not claim ORBIT beats ASTRO at 355M.",
        "- Do not claim the advantage grows with model scale; the 355M batch/token regime differs.",
        "- Do not convert the n=2 horizon/scale intervals into strong inferential claims.",
        "",
        "## Manuscript evidence order",
    ]
    for idx, item in enumerate(results.get("evidence_order", []), start=1):
        lines.append(f"{idx}. {item.replace('_', ' ')}")
    artifact_dir.joinpath("claim_ledger.md").write_text("\n".join(lines) + "\n")


def write_generated(results: dict, paper_dir: Path, artifact_dir: Path, manifest: dict) -> None:
    generated = paper_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    write_macros(results, generated)
    write_matched_table(results, generated)
    write_xconfig_table(results, generated)
    write_ablation_table(results, generated)
    write_transfer_table(results, generated)
    write_broad_table(results, generated)
    write_claim_ledger(results, manifest, artifact_dir)

    compatibility = [
        "% AUTO-GENERATED compatibility shim.",
        "\\input{generated/macros.tex}",
        "\\input{generated/table_matched.tex}",
    ]
    (paper_dir / "generated_results.tex").write_text("\n".join(compatibility) + "\n")


def sync_figures(work_dir: Path, paper_dir: Path) -> None:
    source = work_dir / "paper_artifacts" / "figures"
    target = paper_dir / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.pdf"):
        old.unlink()
    if not source.exists():
        return
    for path in source.glob("*.pdf"):
        shutil.copy2(path, target / path.name)


def compile_tex(paper_dir: Path) -> None:
    if shutil.which("latexmk"):
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
        return
    if shutil.which("pdflatex"):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
        if shutil.which("bibtex"):
            subprocess.run(["bibtex", "main"], cwd=paper_dir, check=False)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=paper_dir,
            check=True,
        )
        return
    raise SystemExit(
        "No latexmk/pdflatex found. Generated TeX is ready; install TeX Live to compile."
    )


def generate_plots(work_dir: Path, allow_incomplete: bool) -> None:
    results, _ = freeze_evidence(work_dir, allow_incomplete=allow_incomplete)
    merged = work_dir / "merged"
    out = work_dir / "paper_artifacts" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # Method overview is data-independent and is always generated with the paper.
    orbit_plot.plot_orbit_overview(out)

    if "matched_confirmation" in results:
        orbit_plot.plot_matched_effect(results, out)

    if (
        "cross_configuration_isolation" in results
        and "mechanism_ablation" in results
    ):
        orbit_plot.plot_mechanism_summary(results, out)

    if (
        "long_horizon_transfer" in results
        and "scale_transfer" in results
    ):
        orbit_plot.plot_transfer_summary(
            orbit_plot.load_jsonl(merged / "horizon_with_astro.jsonl"),
            orbit_plot.load_jsonl(merged / "scale_with_astro.jsonl"),
            out,
        )

    if "broad_independently_tuned_context" in results:
        orbit_plot.plot_broad_confirmation(
            orbit_plot.load_jsonl(merged / "confirm.jsonl"), out
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, default=Path("docs/orbit/paper"))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    results, manifest = freeze_evidence(
        args.work_dir, allow_incomplete=args.allow_incomplete
    )
    artifact_dir = args.work_dir / "paper_artifacts"
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
        generate_plots(args.work_dir, args.allow_incomplete)

    write_generated(results, args.paper_dir, artifact_dir, manifest)
    sync_figures(args.work_dir, args.paper_dir)

    print(f"evidence manifest -> {artifact_dir / 'manifest.json'}")
    print(f"claim ledger -> {artifact_dir / 'claim_ledger.md'}")
    print(f"paper tables -> {args.paper_dir / 'generated'}")
    print(f"paper figures -> {args.paper_dir / 'figures'}")

    if not args.no_compile:
        compile_tex(args.paper_dir)
        print(f"paper -> {args.paper_dir / 'main.pdf'}")


if __name__ == "__main__":
    main()
