"""Run searches first, then independently compute the exhaustive scoring reference."""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from rdkit import Chem

from .molecule import CANDIDATES, GRID, GEOMETRY_SEED, PentaneEnergy
from .search import run_search

TITLE = "Sample-Efficient Molecular Conformation Search with Bayesian Optimization"
METHODS = {"random": "Random search", "bayesian": "Bayesian optimization"}
COLORS = {"random": "#d87928", "bayesian": "#167b91"}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output, reference, runs, summaries):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mpl-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    best = int(np.argmin(reference))
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    # Transpose: first torsion on horizontal axis, second on vertical axis.
    mesh = ax.pcolormesh(np.arange(-185, 176, 10), np.arange(-185, 176, 10),
                         reference.reshape(36, 36).T, cmap="viridis",
                         norm=SymLogNorm(linthresh=20, vmin=reference.min(),
                                         vmax=reference.max()), shading="flat")
    ax.scatter(*CANDIDATES[best], marker="*", s=190, c="white",
               edgecolors="black", label="Grid minimum", clip_on=False)
    ax.set(xlabel=r"$\theta_1$ (degrees)", ylabel=r"$\theta_2$ (degrees)",
           title="Fixed-geometry n-pentane · MMFF94 energy",
           xticks=np.arange(-180, 181, 60), yticks=np.arange(-180, 181, 60))
    ax.legend(loc="upper right")
    bar = fig.colorbar(mesh, ax=ax, label="Energy (kcal/mol)")
    bar.ax.set_title("Symmetric log scale", fontsize=9)
    fig.savefig(output / "energy_heatmap.png")
    plt.close(fig)

    e_star = float(reference.min())
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for method, label in METHODS.items():
        gaps = np.array([np.minimum.accumulate(result.energies) - e_star
                         for (_, m), result in runs.items() if m == method])
        x = np.arange(1, gaps.shape[1] + 1)
        for gap in gaps:
            ax.plot(x, gap, color=COLORS[method], alpha=0.22, linewidth=1)
        ax.plot(x, gaps.mean(axis=0), color=COLORS[method], linewidth=2.5,
                label=label + " (mean)")
    ax.axvline(10, color="gray", linestyle="--", linewidth=1,
               label="End of shared initialization")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(bottom=0)
    ax.set_xlim(1, gaps.shape[1])
    ax.set(xlabel="Energy evaluations", ylabel="Remaining energy gap (kcal/mol)",
           title="Mean and individual trajectories across five paired seeds")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(output / "energy_gap.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    seeds = sorted(set(row["seed"] for row in summaries))
    for ax, column, ylabel in zip(axes, ["final_gap_kcal_mol", "runtime_seconds"],
                                  ["Final energy gap (kcal/mol)", "Runtime (seconds)"]):
        for seed in seeds:
            pair = [next(row[column] for row in summaries
                         if row["seed"] == seed and row["method"] == method)
                    for method in METHODS]
            ax.plot([0, 1], pair, color="#aaaaaa", alpha=0.6, linewidth=1)
        for i, method in enumerate(METHODS):
            values = [row[column] for row in summaries if row["method"] == method]
            offsets = np.linspace(-0.07, 0.07, len(values))
            ax.scatter(i + offsets, values, c=COLORS[method], s=48, zorder=3)
            ax.scatter(i, np.mean(values), c="black", marker="_", s=400,
                       zorder=4, label="Mean" if i == 0 else None)
        ax.set(xticks=[0, 1], xticklabels=["Random", "Bayesian"], ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    axes[1].set_yscale("log")
    fig.suptitle("50 evaluations per method · lines connect paired seeds")
    fig.savefig(output / "final_comparison.png")
    plt.close(fig)


def write_report(output, metadata, summaries):
    means = {m: float(np.mean([r["final_gap_kcal_mol"] for r in summaries
                               if r["method"] == m])) for m in METHODS}
    pairs = {seed: {r["method"]: r["final_gap_kcal_mol"] for r in summaries
                   if r["seed"] == seed} for seed in metadata["seeds"]}
    wins = sum(p["bayesian"] < p["random"] - 1e-8 for p in pairs.values())
    ties = sum(abs(p["bayesian"] - p["random"]) <= 1e-8 for p in pairs.values())
    verdict = ("lower" if means["bayesian"] < means["random"] else "higher")
    lines = [f"# {TITLE}", "", "## Observed result", "",
             f"Bayesian optimization had a **{verdict} mean final energy gap** in this run. "
             f"It won {wins} of 5 paired seeds, with {ties} ties (tolerance 1e-8 kcal/mol). "
             "Five seeds on one small molecule provide a demonstration, not a general performance claim.", "",
             f"The independently evaluated grid minimum is **{metadata['reference_minimum_kcal_mol']:.6f} kcal/mol** "
             f"at angles **{metadata['reference_minimum_angles_deg']} degrees**.", "",
             "| Seed | Method | Best energy (kcal/mol) | Final gap (kcal/mol) | Runtime (s) |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['seed']} | {METHODS[row['method']]} | "
                     f"{row['best_energy_kcal_mol']:.6f} | {row['final_gap_kcal_mol']:.6f} | "
                     f"{row['runtime_seconds']:.4f} |")
    lines += ["", f"Mean final gaps: random **{means['random']:.6f}**, Bayesian "
              f"**{means['bayesian']:.6f} kcal/mol**.", "",
              "## Figures", "", "![Energy landscape](energy_heatmap.png)", "",
              "![Energy gap trajectories](energy_gap.png)", "",
              "![Final gaps and runtime](final_comparison.png)", "",
              "The heatmap uses a symmetric log color scale (linear below 20 kcal/mol) "
              "to show low-energy structure alongside steric clashes. "
              "The gap plot uses a symmetric log scale with a linear region below 0.01 kcal/mol, "
              "so zero remains visible. Thin lines are individual seeds; thick lines are means. "
              "The comparison connects paired seeds and marks means with black horizontal ticks.", "",
              "## Protocol and interpretation", "",
              "- Explicit-hydrogen n-pentane; one ETKDGv3 starting geometry with embedding seed 2026. "
              "No force-field relaxation, including before the searches. Other internal coordinates stay fixed.",
              "- MMFF94 with all nonbonded pairs included (100 Å threshold); energies are in kcal/mol.",
              "- Two absolute carbon torsions, on the 36 × 36 grid from −180° through 170° at 10° spacing.",
              "- Seeds 0–4; each method makes 50 actual oracle calls, including the same 10 initial points "
              "per seed. Subsequent queries never repeat within a run.",
              "- Random search samples unseen grid points uniformly. BO uses sine/cosine features, "
              "a normalized-output Gaussian process with a constant × Matérn-5/2 kernel, "
              "one shared length scale, and expected improvement for minimization (no exploration offset). "
              "Kernel parameters are fitted by marginal likelihood with two additional optimizer restarts "
              "at each query; bounds and numerical jitter are fixed before the experiment.",
              "- All searches complete before a separate oracle evaluates the 1,296-point reference. "
              "The reference table and minimum are used only for scoring and plotting.",
              "- Runtime includes initial queries, geometry construction, force-field energy calls, "
              "and search overhead. It excludes starting-geometry embedding, exhaustive scoring, plotting, "
              "and file output. Methods run sequentially with one BLAS thread; timings depend on hardware.",
              "- MMFF energies are approximate and the fixed embedded geometry affects this landscape. "
              "This is a molecular-modeling optimization benchmark, not a drug-discovery result.", "",
              f"GP optimizer convergence/bound warnings recorded: {metadata['gp_convergence_warning_count']}. "
              "Their per-run counts are saved in summary.csv; fitted parameters can reach the predefined bounds.", "",
              "## Explain it tomorrow", "",
              '“I compared uncertainty-guided Bayesian optimization with random search on a molecular '
              'energy landscape. Each method had just 50 energy evaluations. I held the geometry fixed '
              'except for two torsions, and used an exhaustive grid only to measure how far each search '
              'remained from the best possible grid energy.”', "",
              "Implementation references: [RDKit torsion transforms](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolTransforms.html), "
              "[RDKit force-field energy API](https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html).", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds, budget, n_initial = list(range(5)), 50, 10
    runs = {}
    for seed in seeds:
        initial = np.random.default_rng(seed).choice(len(CANDIDATES), n_initial, replace=False)
        for method in METHODS:
            oracle = PentaneEnergy()
            print(f"Seed {seed}: {METHODS[method]} ...", flush=True)
            result = run_search(oracle, CANDIDATES, initial, budget, method, seed)
            assert oracle.evaluations == budget
            assert len(set(result.indices)) == budget
            runs[seed, method] = result
        assert np.array_equal(runs[seed, "random"].indices[:n_initial],
                              runs[seed, "bayesian"].indices[:n_initial])
        np.testing.assert_allclose(runs[seed, "random"].energies[:n_initial],
                                   runs[seed, "bayesian"].energies[:n_initial], atol=1e-10)
    print("Searches complete. Evaluating independent scoring reference ...", flush=True)
    reference_oracle = PentaneEnergy()
    start = perf_counter()
    reference = np.array([reference_oracle(angles) for angles in CANDIDATES])
    reference_seconds = perf_counter() - start
    e_star = float(reference.min())
    histories, summaries = [], []
    for (seed, method), result in runs.items():
        np.testing.assert_allclose(result.energies, reference[result.indices], atol=1e-8, rtol=0)
        gaps = np.minimum.accumulate(result.energies) - e_star
        assert np.all(gaps >= -1e-8)
        for t, (idx, energy, gap, elapsed) in enumerate(zip(
                result.indices, result.energies, gaps, result.elapsed_seconds), start=1):
            histories.append(dict(seed=seed, method=method, evaluation=t,
                                  candidate_index=int(idx), theta1_deg=CANDIDATES[idx, 0],
                                  theta2_deg=CANDIDATES[idx, 1], energy_kcal_mol=energy,
                                  best_energy_kcal_mol=e_star+gap,
                                  gap_kcal_mol=gap, elapsed_seconds=elapsed))
        best_index = int(result.indices[np.argmin(result.energies)])
        summaries.append(dict(seed=seed, method=method, evaluations=budget,
                              best_energy_kcal_mol=float(result.energies.min()),
                              final_gap_kcal_mol=float(gaps[-1]),
                              runtime_seconds=float(result.elapsed_seconds[-1]),
                              best_theta1_deg=float(CANDIDATES[best_index, 0]),
                              best_theta2_deg=float(CANDIDATES[best_index, 1]),
                              gp_convergence_warning_count=result.fit_bound_warnings))
    best = int(np.argmin(reference))
    metadata = dict(title=TITLE, smiles="CCCCC", force_field="MMFF94",
                    geometry_seed=GEOMETRY_SEED, geometry_relaxed=False,
                    torsion_atom_indices=[[0, 1, 2, 3], [1, 2, 3, 4]],
                    grid_degrees=GRID.tolist(), candidate_count=len(CANDIDATES),
                    seeds=seeds, budget=budget, shared_initial_evaluations=n_initial,
                    search_energy_evaluations=2*len(seeds)*budget,
                    reference_energy_evaluations=reference_oracle.evaluations,
                    reference_minimum_kcal_mol=e_star,
                    reference_minimum_angles_deg=CANDIDATES[best].tolist(),
                    reference_runtime_seconds=reference_seconds,
                    gp_convergence_warning_count=sum(r.fit_bound_warnings for r in runs.values()),
                    python=platform.python_version(), platform=platform.platform(),
                    packages={name: importlib.metadata.version(name) for name in
                              ["numpy", "scipy", "scikit-learn", "matplotlib", "rdkit", "threadpoolctl"]})
    write_csv(output / "history.csv", histories)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "reference_grid.csv", [dict(theta1_deg=a, theta2_deg=b, energy_kcal_mol=e)
                                              for (a, b), e in zip(CANDIDATES, reference)])
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    Chem.MolToMolFile(reference_oracle.template, str(output / "initial_geometry.mol"))
    Chem.MolToMolFile(reference_oracle.geometry(CANDIDATES[best]), str(output / "grid_minimum.mol"))
    plot_results(output, reference, runs, summaries)
    write_report(output, metadata, summaries)
    print(f"Reference minimum: {e_star:.6f} kcal/mol at {CANDIDATES[best].tolist()}")
    print(f"Results: {output / 'report.md'}")


if __name__ == "__main__":
    main()
