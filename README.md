# Zooming in on weather and market access: how much does map resolution matter?

Code and data for a study of how much spatial resolution matters when market
access is conditioned on drought and wet-spell conditions (SPEI-3), using 564
half-degree cells in CA, FL, TX and IL and 24,205 census tracts.

Nicholas J. Sokol, Department of Geography & Sustainability, University of
Tennessee, Knoxville.

## Contents

- `code/run_all.py`: runs experiments E0 to E4 and makes the figures. Reads only `data/`.
- `code/fetch_data.py`: rebuilds `data/` from the public inputs listed below.
- `code/verify_synthetic.py`: checks on synthetic data only; not used for the results.
- `data/`: input files (see below).
- `results/`: output tables and figures written by `run_all.py`.
- `figures/`: copies of the five figures in PNG and PDF.
- `run_manifest.json`: run settings and output list from the recorded run.

## Install and run

```
pip install -r requirements.txt
cd code
python3 run_all.py
```

The recorded run took about 550 seconds (`wall_clock_seconds` in `run_manifest.json`).
Settings: alpha = beta = 0.5, d0 = 100 km.

## Data

| File | Contents |
| --- | --- |
| `data/cells.csv` | 564 occupied 0.5-degree cells: id, lat, lon, state, population |
| `data/tracts.csv` | 24,205 tracts: geoid, state, population, centroid lat/lon, cell id |
| `data/spei_cells.npz` | SPEI-3 for 528 months (1980-01 to 2023-12) by 564 cells |
| `data/state_boundaries.geojson` | simplified state boundaries used in Figure 5 |
| `data/source_manifest.json` | SHA-256 hashes of the source files and built outputs |

Sources:

- SPEIbase v2.10, SPEI-3 NetCDF, DOI 10.20350/digitalCSIC/16497.
- 2023 American Community Survey tract populations for CA, FL, TX and IL.
- U.S. Census Bureau cartographic boundary file `cb_2023_us_tract_500k`.

`fetch_data.py` needs these files locally; set the paths at the top of the
script before running it. It is not needed to run `run_all.py`.

## Output in `results/`

- `e0_gate_replication.csv`: consistency checks on the access measure.
- `e1_ladder.csv`, `e1_coherence_bins.csv`, `e1_coherence_leafmonths.csv.gz`: coarsening to 1, 2 and 4 degree blocks.
- `e2_equal_budget.csv`: partition rules at 64, 128 and 256 leaves.
- `e4_barnes_hut.csv`: Barnes-Hut approximation, 24,205 tract targets and 564 sources.
- `fig1_ladder`, `fig2_equal_budget`, `fig3_representation`, `fig4_barnes_hut`, `fig5_case_anomaly_maps`, each as PNG and PDF.

## Headline numbers in the recorded run

- E0: zero-anomaly loss is 0 (max 2.7e-15). Random-weather mean loss is -26.50%
  and the all-months mean loss relative to static access is -26.64%. The
  climatology-relative aggregate is -4.50% in the driest decile of months and
  -7.53% in the wettest. Mean per-cell absolute deviation is 14.09%. The
  correlation of aggregate deviation with mean absolute SPEI is -0.987.
- E1, field averaging at 1 / 2 / 4 degrees: mean-loss bias +0.29 / +0.58 / +1.17
  percentage points and anomaly RMSE 2.71 / 3.62 / 7.55 points. With source
  aggregation the anomaly RMSE is 0.32 / 0.89 / 2.98 points. Mean source bias
  of the leaf aggregate is +0.090 where coherence is below 0.25 and -0.003
  where it is above 0.9.
- E2, anomaly RMSE of field averaging at 64 / 128 / 256 leaves: 2.73 / 1.09 /
  0.27 points with the response-error partition (`l1g_static`) and
  4.06 / 2.72 / 2.26 points with the uniform partition. Source aggregation
  error is 0.13 to 0.46 times field averaging at equal leaves.
- E4: with theta = 0.3 the mean relative error is 0.75 to 0.80% at a cost
  ratio of 0.235. With theta = 0.5 it is 4.2 to 4.3% at 0.155. No theta gives
  error below 1% at a cost ratio below 0.2.
