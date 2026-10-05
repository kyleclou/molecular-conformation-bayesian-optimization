# Sample-Efficient Molecular Conformation Search with Bayesian Optimization

Find a low-energy shape of n-pentane using as few MMFF94 energy calculations as possible. This is a numerical optimization benchmark on a fixed molecular geometry.

## Run

Use Python 3.11 or later. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m conformation_search
```

The experiment runs five paired seeds (0–4), with 50 evaluations per method, then evaluates the independent scoring reference. Results are written to `results/`. To choose another destination:

```sh
.venv/bin/python -m conformation_search --output results_repeat
```

Read [the generated report](results/report.md) for results, plots, interpretation, and protocol. Existing files at the chosen output destination are replaced on rerun. The installed package versions are recorded in `results/metadata.json`; `requirements-lock.txt` captures the original environment.

## Experiment

- Molecule: `CCCCC`, with explicit hydrogens. RDKit ETKDGv3 embeds one starting geometry using seed 2026. No force-field relaxation occurs at any point.
- Variables: absolute torsions of carbon atoms `(0,1,2,3)` and `(1,2,3,4)`. Carbon indices 0–4 mean C1–C5. Bond lengths, bond angles, and all other torsions stay fixed.
- Grid: −180°, −170°, …, 170° for each torsion, giving 1,296 candidates.
- Objective: construct a fresh copy of the geometry, set the two torsions, and call RDKit MMFF94 `CalcEnergy()` (kcal/mol). Each search query makes an actual energy call.
- Initialization: 10 distinct uniform grid points per seed, shared by both methods and counted in each 50-query budget.
- Random search: 40 further uniform queries without replacement.
- Bayesian optimization: 40 further queries chosen by expected improvement, using a Gaussian process on `(sin θ1, cos θ1, sin θ2, cos θ2)` features. Constant × Matérn-5/2 covariance; normalized outputs; one shared length scale; hyperparameters fitted from observed energies only with two additional optimizer restarts; jitter `1e-8`; length-scale bounds `[0.05,5]`, constant bounds `[1e-3,1e3]`. No acquisition exploration offset.
- Scoring: after **all searches finish**, a separate oracle computes the exhaustive grid. Searches receive no reference energies or reference minimum. Remaining gap is `best_observed_energy − reference_minimum`.
- Total calls: 500 search evaluations plus 1,296 reference evaluations. Geometry embedding is setup, outside these counts.

The reference minimum is the best **grid** conformation of this fixed embedded geometry, not the continuous or fully relaxed molecular minimum. Pentane is a demonstration molecule, not a drug-discovery result. Five seeds are too few for broad conclusions.

## Outputs

| File | Contents |
| --- | --- |
| `results/report.md` | Results table, explanation, plots, and presentation wording |
| `results/energy_heatmap.png` | Two-dimensional MMFF94 energy landscape |
| `results/energy_gap.png` | Individual and mean gap trajectories |
| `results/final_comparison.png` | Paired final gap and runtime comparison |
| `results/history.csv` | All 500 queries, energies, best energies, gaps, and elapsed times |
| `results/summary.csv` | Final gap, best angles, runtime, and GP warning counts per run |
| `results/reference_grid.csv` | All 1,296 reference energies, used only after searches |
| `results/metadata.json` | Protocol, reference minimum, runtime, package versions, platform |
| `results/initial_geometry.mol` | Explicit-hydrogen starting geometry |
| `results/grid_minimum.mol` | Best fixed geometry on the exhaustive grid |

Search runtime includes initial calls, geometry creation, energy evaluation, GP fitting, and candidate selection. It excludes embedding, reference scoring, plotting, and output writing. A single BLAS thread is used; wall-clock times depend on hardware and include normal timing noise.

## Verify

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests verify bond and angle preservation, other torsion preservation, target angles, periodic energy and features, order-independent energies, reproducibility, expected improvement, evaluation budgets, and absence of repeat queries. The full experiment also checks each queried energy against the independently evaluated reference and confirms shared initialization.

Code: `molecule.py` defines the chemistry oracle; `search.py` defines both optimizers; `experiment.py` performs scoring, exports, plotting, and reporting.

API references: [RDKit torsion transforms](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolTransforms.html), [RDKit force-field energy](https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html).
