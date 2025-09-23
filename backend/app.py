# backend/app.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import ml_utils as utils
import pandas as pd

app = FastAPI(title="Fish Biodiversity Backend")

# Allow your frontend (change origin as needed)
origins = [
    "http://localhost:3000",
    "https://your-teams-app-domain",
    "http://localhost:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/species")
def species_list():
    try:
        return {"species": utils.list_species()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/species/{name}")
def species_records(name: str, limit: int = 500, start_date: Optional[str] = None, end_date: Optional[str] = None):
    recs = utils.get_species_records(name, limit=limit, start_date=start_date, end_date=end_date)
    return {"records": recs, "count": len(recs)}

class PredictRequest(BaseModel):
    species: str
    sst: Optional[float] = None
    sss: Optional[float] = None
    depth: Optional[float] = None
    decimalLatitude: Optional[float] = None
    decimalLongitude: Optional[float] = None

@app.post("/predict")
def predict(req: PredictRequest):
    # load model
    model = utils.load_model_for_species(req.species)
    if model is None:
        raise HTTPException(status_code=404, detail="Model for species not found")
    # prepare features in same order as training
    X = pd.DataFrame([{
        "sst": req.sst if req.sst is not None else -999,
        "sss": req.sss if req.sss is not None else -999,
        "depth": req.depth if req.depth is not None else -999
    }])
    prob = float(model.predict_proba(X)[:,1][0])
    return {"species": req.species, "probability": prob}

@app.get("/nearest_occurrence")
def nearest_occ(lat: float = Query(...), lon: float = Query(...)):
    gdf = utils.load_data()
    if "geometry" not in gdf.columns:
        raise HTTPException(status_code=500, detail="Occurrence data lacks geometry")
    pt = gdf.geometry.unary_union  # not used directly; we'll compute distance
    # compute distances (project to metric CRS for accurate meters)
    try:
        # fast brute force: compute Euclidean in degrees -> approximate; for exact use pyproj to project to local CRS
        gdf["dist_deg"] = ((gdf.decimalLatitude - lat)**2 + (gdf.decimalLongitude - lon)**2)**0.5
        nearest = gdf.loc[gdf["dist_deg"].idxmin()]
        return {
            "nearest_species": nearest.get("scientificName"),
            "distance_degrees": float(nearest["dist_deg"]),
            "nearest_lat": float(nearest["decimalLatitude"]),
            "nearest_lon": float(nearest["decimalLongitude"]),
            "eventDate": str(nearest.get("eventDate"))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
