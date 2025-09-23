# backend/ml_utils.py
import pandas as pd
import joblib
from pathlib import Path
from functools import lru_cache
import geopandas as gpd
from shapely.geometry import Point

DATA_PATH = Path("../data/processed/occurrences_clean.csv")
MODELS_DIR = Path("../models")

@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["eventDate"], low_memory=False)
    # create geometry for geospatial joins
    if "decimalLatitude" in df.columns and "decimalLongitude" in df.columns:
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.decimalLongitude, df.decimalLatitude)], crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(df)
    return gdf

def list_species():
    df = load_data()
    return sorted(df["scientificName"].dropna().unique().tolist())

def get_species_records(scientificName, limit=500, start_date=None, end_date=None):
    df = load_data()
    q = df[df["scientificName"] == scientificName]
    if start_date is not None:
        q = q[q["eventDate"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        q = q[q["eventDate"] <= pd.to_datetime(end_date)]
    return q.sort_values("eventDate", ascending=False).head(limit).to_dict(orient="records")

def load_model_for_species(species_name):
    p = MODELS_DIR / f"{species_name.replace(' ','_')}_rf.pkl"
    if p.exists():
        return joblib.load(p)
    # fallback: try a generic model
    generic = MODELS_DIR / "generic_rf.pkl"
    if generic.exists():
        return joblib.load(generic)
    return None
