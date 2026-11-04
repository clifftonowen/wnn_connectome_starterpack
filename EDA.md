# Initial data exploration

Open [`notebooks/01_connectome_eda.ipynb`](notebooks/01_connectome_eda.ipynb)
with the **Python (Alfa Connectome)** kernel. The notebook reads one complete
sequence first, then samples whole Parquet row groups across the training and
validation files. It keeps only summary statistics in memory and writes
`reports/eda.md`, `reports/eda.json`, and `reports/feature_stats.csv` when run
through to the final cell.

## Local notebook environment

The `.venv` in this checkout uses the available Python 3.14 interpreter and
inherits the machine's existing Jupyter, NumPy, pandas, and Matplotlib packages.
Its kernel is registered under `.venv/share/jupyter/kernels/alfa-connectome`.

PyArrow is required to read the dataset. Install the notebook dependencies
from a terminal with package access:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\01_connectome_eda.ipynb
```

The competition's `requirements.txt` targets its Python 3.11 runtime and pins
PyArrow 19.0.1. The notebook environment uses a Python 3.14 compatible
PyArrow release; it is for EDA, while the final submission should be tested
against the competition runtime.

The same bounded-memory analysis is available from the command line with
`.\.venv\Scripts\python.exe eda.py`. Both the notebook and script default to
64 training sequences and 16 validation sequences, spaced across their files.
Numeric distributions and correlations are sampled; full-file sizes, row
counts, and row-group counts come from Parquet metadata.
