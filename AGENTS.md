# TMS_formfinding

Form-finding of tensile membrane structures (TMS) using the Updated Weight Method (UWM).
This is a small, dependency-light scientific Python project (no build step, no server).

## Runnable scripts

All scripts are self-contained and write an interactive Plotly `*.html` visualization to the
current directory, then exit:

- `uwm_hypar_formfinding.py` — square hypar form-finding (writes `hypar_fenicsx_uwm.html`).
- `example1_conic` — conic exercise 1 (writes `conic_ex1_formfound.html`). Runs as `python3 example1_conic` (no `.py` extension).
- `example2_conic` — conic exercise 2 (writes `conic_ex2_formfound.html`). Runs as `python3 example2_conic`.
- `twin hypar_from finding_UWM.ipynb` — notebook mirroring the hypar script (requires Jupyter, not installed by default).

Run any script directly, e.g. `python3 uwm_hypar_formfinding.py`.

## Cursor Cloud specific instructions

- Dependencies are `numpy`, `scipy`, `plotly` (installed by the update script). `scipy` and `plotly`
  are imported behind `try/except`; scripts fall back to slower/no-plot paths if missing, so a
  successful run alone does not prove they are installed — check import success if in doubt.
- `example1_conic` and `example2_conic` have no file extension; invoke them as
  `python3 example1_conic`, not by double-clicking or module import.
- `uwm_hypar_formfinding.py` optionally uses `dolfinx`/`mpi4py` (FEniCSx) for meshing but transparently
  falls back to a NumPy structured mesh when they are absent. FEniCSx is NOT installed and is not
  needed; do not attempt to install it just to run the scripts.
- There is no lint config, test suite, or build step in this repo. "Running the app" means executing
  one of the scripts above and opening the generated `.html` in a browser.
- Generated `*.html` outputs are gitignored; they are regenerated on each run.
