#!/usr/bin/env python3
"""Builds data/ from the public inputs.

Inputs: SPEIbase v2.10 SPEI-3 NetCDF (DOI 10.20350/digitalCSIC/16497), 2023 ACS tract population
CSVs for CA, FL, TX and IL, and the cb_2023_us_tract_500k Census shapefile (paths set below).
Outputs: cells.csv, spei_cells.npz, tracts.csv, state_boundaries.geojson, source_manifest.json.
"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPEI_NC = Path.home() / "workspace/papers-finish/cdpe_sensitivity/data/spei03.nc"
TRACT_TMPL = str(Path.home() / "workspace/gceia-papers/paper2-amr/pipeline/outputs/tract_scores_{state}.csv")
TRACT_SHP = Path.home() / "workspace/gceia-papers/paper2-amr/pipeline/data/cb_2023_us_tract_500k.shp"
STATES = ["CA", "FL", "TX", "IL"]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    import geopandas as gpd, xarray as xr
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {}}
    for name, p in [("spei03_nc", SPEI_NC), ("tract_shp", TRACT_SHP)] + \
                   [(f"tract_scores_{s}", Path(TRACT_TMPL.format(state=s))) for s in STATES]:
        manifest["sources"][name] = {"path": str(p), "sha256": sha256(p)}
    # tract table
    frames = []
    for s in STATES:
        d = pd.read_csv(TRACT_TMPL.format(state=s), dtype={"geoid": str})
        frames.append(d[["geoid", "state", "population"]])
    tracts = pd.concat(frames, ignore_index=True)
    tracts = tracts[np.isfinite(tracts["population"]) & (tracts["population"] > 0)].copy()
    shp = gpd.read_file(TRACT_SHP)
    cents = shp.geometry.centroid
    geo = pd.DataFrame({"GEOID": shp["GEOID"], "clat": cents.y.values, "clon": cents.x.values}).set_index("GEOID")
    tracts = tracts.join(geo, on="geoid", how="inner").reset_index(drop=True)
    assert len(tracts) == 24205, len(tracts)
    # native grid and SPEI subset, clipped to the domain and 1980-2023
    ds = xr.open_dataset(SPEI_NC)
    lat_all, lon_all = ds["lat"].values, ds["lon"].values
    time_all = pd.to_datetime(ds["time"].values)
    tmask = (time_all >= "1980-01-01") & (time_all <= "2023-12-31")
    latmask = (lat_all >= 25.25) & (lat_all <= 49.75)
    lonmask = (lon_all >= -125.25) & (lon_all <= -66.75)
    sub = ds["spei"].values[np.ix_(tmask, latmask, lonmask)].astype(np.float64)  # (T,50,118)
    lat, lon = lat_all[latmask], lon_all[lonmask]
    months = [f"{t.year:04d}-{t.month:02d}" for t in time_all[tmask]]
    assert sub.shape == (528, 50, 118), sub.shape
    lat_edges = np.concatenate([[lat[0] - 0.25], lat + 0.25])
    lon_edges = np.concatenate([[lon[0] - 0.25], lon + 0.25])
    rows = np.clip(((tracts["clat"].values - lat_edges[0]) / 0.5).astype(int), 0, 49)
    cols = np.clip(((tracts["clon"].values - lon_edges[0]) / 0.5).astype(int), 0, 117)
    flat = rows * 118 + cols
    pop_grid = np.zeros(50 * 118); np.add.at(pop_grid, flat, tracts["population"].values)
    occ = np.where(pop_grid > 0)[0]
    assert len(occ) == 564 and abs(pop_grid.sum() - 103424435) < 1
    cell_state = []
    for f in occ:
        m = flat == f
        cell_state.append(tracts.loc[m].groupby("state")["population"].sum().idxmax())
    orr, occ_c = np.divmod(occ, 118)
    cells = pd.DataFrame({"cell_id": np.arange(len(occ)), "lat": lat[orr], "lon": lon[occ_c],
                          "state": cell_state, "population": pop_grid[occ]})
    spei_cells = sub[:, orr, occ_c]  # (528, 564)
    tracts["cell_id"] = pd.Series(flat).map({int(f): i for i, f in enumerate(occ)}).values
    tracts_out = tracts.rename(columns={"clat": "lat", "clon": "lon"})[["geoid", "state", "population", "lat", "lon", "cell_id"]]
    # state boundaries for the maps (dissolved tract boundaries, simplified)
    shp2 = shp[shp["STUSPS"].isin(STATES)][["STUSPS", "geometry"]].copy()
    shp2["geometry"] = shp2.geometry.buffer(0)  # repair before dissolve
    bnd = shp2.dissolve(by="STUSPS").reset_index()
    bnd["geometry"] = bnd.geometry.simplify(0.03)
    bnd.to_file(DATA / "state_boundaries.geojson", driver="GeoJSON")
    cells.to_csv(DATA / "cells.csv", index=False)
    tracts_out.to_csv(DATA / "tracts.csv", index=False)
    np.savez_compressed(DATA / "spei_cells.npz", months=np.array(months), spei=spei_cells)
    manifest["outputs"] = {"cells": len(cells), "tracts": len(tracts_out), "months": len(months),
                           "total_population": float(cells.population.sum()),
                           "nan_cell_months": int((~np.isfinite(spei_cells)).sum())}
    for f in ["cells.csv", "tracts.csv", "spei_cells.npz"]:
        manifest["outputs"][f + "_sha256"] = sha256(DATA / f)
    (DATA / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["outputs"], indent=2))

if __name__ == "__main__":
    main()
