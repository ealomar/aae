# Supplementary material: UBJBAS-D-26-00004 (Revision 2)

*Integrating AI, Analytics, and Automation in Accounting Education: A Pedagogical Framework for the Digital Era*

## What these files are

The 150 records analysed in the manuscript are **simulated**. No questionnaire was administered to students, and no institution-linked student records were analysed. The public dataset of Hasan et al. (2021) supplies only two marginal distributions: the shape of coursework marks, used for accounting performance (PERF), and the rank distribution of Moodle activity, used for the LMS Activity Score.

Hasan, R., Palaniappan, S., Mahmood, S., Abbas, A., and Sarker, K.U. (2021). Dataset of students' performance using student information system, Moodle and the mobile application "eDify". *Data*, 6(11), 110. https://doi.org/10.3390/data6110110 (files on Zenodo).

The structural path values in `simulation_parameters.json` are effect sizes specified by the authors. They reproduce the coefficients reported in the first version of the manuscript, which analysed an earlier simulated dataset whose data file and generator settings were lost (manuscript Section 3.2; Table 17).

## Files

| File | Content | Manuscript |
|---|---|---|
| `simulate_dataset.py` | Generator (seed 2026). Writes `simulated_dataset.csv` and `simulation_parameters.json`. | Section 3.2 |
| `simulation_parameters.json` | Every generator value: seed, N, exposure correlation, path values, item loadings and calibration, targets, composition, source anchors. | Section 3.2, Table 17 |
| `simulated_dataset.csv` | The 150 released records (44 columns, no missing values). | Tables 4, 9 |
| `reanalysis.py` | Descriptives, EFA/CFA, reliability, HTMT, hierarchical regression, diagnostics, Wald tests, bootstrap mediation, Figures 4–7. Writes `reanalysis_results.xlsx` (released as `analysis_output.xlsx`). | Tables 9–14, Appendix B |
| `power_analysis.py` | Monte Carlo power, 500 replications at N = 100–400. Writes `power_analysis.csv`. | Section 7.7 |
| `sensitivity_analysis.py` | Parameter recovery and eight sensitivity scenarios, 500 replications at N = 150 and 200. Writes `sensitivity_results.csv` and `sensitivity_summary.csv`. | Section 7.5, Tables 15–16 |
| `sem_robustness.py` | Full structural equation model with latent constructs and controls; 1,000 bootstrap resamples for indirect effects. Writes `sem_paths.csv`, `sem_fit.csv` and `sem_indirect.csv`. | Section 7.6, Table 17 |
| `make_figures.py` | Figures 1–7 at 600 dpi. | Figures 1–7 |

## How to reproduce

1. Install Python 3.11 or later and the packages listed below.
2. Download the Hasan et al. files from Zenodo and unzip them into a folder, e.g. `hasan/`. The generator reads `Result*.csv` and `Activity*.csv`.
3. Run, in this order:

```
python simulate_dataset.py hasan/          # reproduces simulated_dataset.csv byte for byte
python reanalysis.py simulated_dataset.csv
python power_analysis.py
python sensitivity_analysis.py             # 8 scenarios x 500 replications; takes several minutes
python sem_robustness.py simulated_dataset.csv
python make_figures.py
```

Every script uses a fixed seed. We verified the package in a clean environment:
- the generator output is byte-identical to the released `simulated_dataset.csv`;
- every sheet of the analysis output matches `analysis_output.xlsx`;
- `power_analysis.csv` is reproduced exactly.

## Environment used

Python 3.13; numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, statsmodels 0.15.0, factor_analyzer 0.5.1, semopy 2.3.11, matplotlib 3.11.2, openpyxl 3.1.5.
