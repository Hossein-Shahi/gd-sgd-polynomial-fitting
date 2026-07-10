# Comparative Analysis of Gradient Descent and Stochastic Gradient Descent for Polynomial Curve Fitting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21294019.svg)](https://doi.org/10.5281/zenodo.21294019)

Code, data, and complete experimental records for the paper:

> Hossein Shahi, Juefei Yuan, Zahra Zolnourian.
> *Comparative Analysis of Gradient Descent and Stochastic Gradient Descent for
> Polynomial Curve Fitting: A Transparent From-Scratch Implementation Study.* 2026.

Both optimizers are implemented in **pure Python** (no NumPy, no machine learning
libraries), so every gradient computation and weight update is visible. The
experiments cover the full 3 × 3 grid of polynomial degrees (2, 3, 4) and learning
rates (0.001, 0.01, 0.1), each configuration repeated over **ten random seeds
(0–9)** with GD and SGD sharing the same initialization within each seed. Every
number and figure in the paper is reproduced by the scripts below.

## Contents

| File | Purpose |
| --- | --- |
| `gradient_descent.py` | Pure-Python full-batch Gradient Descent |
| `stochastic_gradient_descent.py` | Pure-Python per-sample SGD with per-epoch shuffling |
| `main.py` | Original single-run driver (one unseeded run per configuration) |
| `Part1_x_y_Values.txt` | Dataset: 100 (x, y) points |
| `run_experiments.py` | 10-seed grid (180 runs) → `results.json`, `results_summary.csv`, four core figures |
| `make_figures.py` | Remaining main-text figures and the Appendix A gallery; asserts that every plotted seed-0 run matches `results.json` |
| `verify_closed_form.py` | Exact least-squares minima for degrees 2–4 (the only script requiring NumPy) |
| `results.json`, `results_summary.csv` | Final training losses: all seeds, plus mean ± SD per configuration (Table 1) |
| `figures/` | All 15 figures exactly as they appear in the paper |

## Reproducing the paper

```bash
pip install matplotlib pillow numpy
python run_experiments.py     # 180 training runs, a few minutes (pure Python)
python make_figures.py        # remaining figures + Appendix A gallery
python verify_closed_form.py  # closed-form minima quoted in Sections 4.1 and 5.4
```

Run all commands from the repository root. `run_experiments.py` regenerates
`results.json` / `results_summary.csv` (the data behind Table 1) and
`make_figures.py` refuses to draw anything unless its seed-0 reruns match those
results exactly.

Figure file → paper mapping: `fig1_fit_deg3` → Fig. 1, `fig4_degrees` → Fig. 2,
`fig3_lr_compare` → Fig. 3, `fig_instability` → Fig. 4, `fig2_loss_deg3` → Fig. 5,
`fig_weights_deg3` → Fig. 6, `figA_d{degree}_lr{rate}` → Figs. A1–A9.

## Data provenance

The dataset was provided as part of a graduate course project at Southeast
Missouri State University. Its exact generating process is not documented, so the
paper characterizes it empirically: 100 points, x evenly spaced on [0, 10],
y bounded in [0, 20] with saturation at both bounds (five points exactly at 0,
seven exactly at 20).

## License

Code released under the MIT License (see `LICENSE`). The dataset is included for
reproducibility of the accompanying paper.

## Citation

If you use this work, please cite the paper:

```bibtex
@article{shahi2026gdsgd,
  title  = {Comparative Analysis of Gradient Descent and Stochastic Gradient
            Descent for Polynomial Curve Fitting: A Transparent From-Scratch
            Implementation Study},
  author = {Shahi, Hossein and Yuan, Juefei and Zolnourian, Zahra},
  year   = {2026},
  note   = {arXiv:XXXX.XXXXX}
}
```

To cite the code and data archive itself:

```bibtex
@software{shahi2026gdsgd_code,
  author    = {Shahi, Hossein and Yuan, Juefei and Zolnourian, Zahra},
  title     = {gd-sgd-polynomial-fitting: Code and data for "Comparative
               Analysis of Gradient Descent and Stochastic Gradient Descent
               for Polynomial Curve Fitting"},
  year      = {2026},
  version   = {v1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21294019},
  url       = {https://doi.org/10.5281/zenodo.21294019}
}
```
