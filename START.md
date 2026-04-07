# How to Start the Project

## Option 1: 3D Cube Input (recommended)

Runs the 3D interface: click stickers, set colors, get a solution, then step through moves with the cube updating on screen.

```bash
cd /path/to/Rubik-s-Cube-Solver

# Install dependencies (one-time)
pip install numpy matplotlib opencv-python

# Use the local Kociemba solver (no need to install it)
PYTHONPATH=./kociemba python cube3d.py
```

Or with `python3`:

```bash
PYTHONPATH=./kociemba python3 cube3d.py
```

**Controls:** Enter cube state (click stickers + palette or use keys 1–6), press **Enter** to solve, then **N** to apply each move and see the cube update. Hold your physical cube with **BLUE on top**, **RED in front**, **YELLOW on right**.

---

## Option 2: Camera + manual mode (main.py)

Uses the webcam to scan faces (or manual input). Needs a camera and may require camera permissions.

```bash
cd /path/to/Rubik-s-Cube-Solver
pip install numpy opencv-python
PYTHONPATH=./kociemba python main.py
```

Press **M** to switch between manual and camera mode. Press **Enter** to solve.

---

## Option 3: Install Kociemba and run without PYTHONPATH

If you can install the local package (e.g. no permission issues):

```bash
pip install -e ./kociemba
python cube3d.py
# or
python main.py
```

---

## Reproducing paper results

All tables and figures in `report.tex` can be regenerated from this repository using the commands below (run from the project root). The files in `results/*.json` are the source for the report tables.

**1. Main benchmark table (Table 2: baseline vs multi-solution Phase 1)**  
Compares baseline ($K=1$) with multi-solution settings; outputs mean/median moves, % ≤20 moves, mean/P95/P99 solve time.

```bash
PYTHONPATH=./kociemba python3 benchmark.py --scrambles 200 --compare-baseline --max-phase1 5 --output-json results/benchmark.json --seed 42 -v
```

**2. Anytime sweep (Table: solution quality vs time budget)**  
Runs the same scrambles under different time budgets (0.1 s, 0.3 s, 0.5 s, 1.0 s). Omit `--compare-baseline`.

```bash
PYTHONPATH=./kociemba python3 benchmark.py --scrambles 200 --max-phase1 5 --time-budgets 0.1 0.3 0.5 1.0 --output-json results/anytime_sweep.json --seed 42 -v
```

Results are written to `results/benchmark.json` and `results/anytime_sweep.json`. See `report.tex` for methodology and table references.

---

## Troubleshooting

- **"No module named 'kociemba'"** → Use `PYTHONPATH=./kociemba` as in Option 1.
- **Camera not working in main.py** → Use manual mode (default) or run `cube3d.py` instead.
- **Matplotlib/font warnings** → Safe to ignore on first run; the app still runs.
