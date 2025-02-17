# Branch-and-Bound with Basic Parallelization & ML

This repository contains an implementation of the Branch-and-Bound algorithm with basic parallelization and machine learning enhancements.

## Project Structure

- **`branch-and-bound.ipynb`** – The main notebook that initiates the experiments.
- **Modules:**
  - `baseline_bb.py` – Standard Branch-and-Bound implementation.
  - `parallel_bb.py` – Parallelized version of Branch-and-Bound.
  - `ml_bb.py` – Machine Learning-enhanced Branch-and-Bound.
  - `brute_force.py` – Brute-force approach for comparison.
- **Data Directory (`data/`)** – Contains test data files:
  - Test data is a text file containing five instances of a 10-city Traveling Salesman Problem (TSP) cost matrix.
- **Utils Directory (`utils/`)** – Contains helper functions:
  - `load_data.py` – Function to read test data.
