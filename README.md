# Medical Equipment Suppliers — EDA Notebook

Summary
- Lightweight exploratory data analysis of the Medical Equipment Suppliers dataset.
- The main notebook is [medical_equipment.ipynb](medical_equipment.ipynb). It loads the CSV dataset into a DataFrame [`df`](medical_equipment.ipynb) and computes a missing-value summary stored in [`missing_df`](medical_equipment.ipynb).

Files
- [Medical-Equipment-Suppliers.csv](Medical-Equipment-Suppliers.csv) — input dataset.
- [medical_equipment.ipynb](medical_equipment.ipynb) — Jupyter notebook performing EDA and saving the missing-value report.
- [eda_missing_report.csv](eda_missing_report.csv) — generated missing-value report (saved by the notebook).

Quick start (local)
1. Install minimal dependencies:
```bash
pip install pandas numpy matplotlib seaborn jupyterlab
