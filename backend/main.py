import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Fish Occurrence API")

# Load CSV
DATA_PATH = r"D:\Dev\SIH\Database\fish-platform\data\processed\occurrences_clean.csv"
df = pd.read_csv(DATA_PATH)

# Ensure scientificName is string
df["scientificName"] = df["scientificName"].astype(str)

# Function to safely convert DataFrame to JSON
def df_to_safe_json(df_subset, fields):
    # Keep only relevant columns
    df_subset = df_subset[fields]
    
    # Replace NaN / inf with None
    df_subset = df_subset.replace([float("nan"), float("inf"), -float("inf")], None)
    
    # Convert to dict
    return df_subset.to_dict(orient="records")

# -----------------------------
# Endpoint: species list
# -----------------------------
@app.get("/species")
def list_species():
    species_list = df["scientificName"].dropna().unique().tolist()
    return {"num_species": len(species_list), "species": species_list}

# -----------------------------
# Endpoint: species occurrences
# -----------------------------
@app.get("/species/{species_name}")
def species_occurrences(species_name: str):
    species_data = df[df["scientificName"].str.lower() == species_name.lower()]
    if species_data.empty:
        raise HTTPException(status_code=404, detail="Species not found")
    
    fields = ["decimalLatitude","decimalLongitude","eventDate","sst","sss","depth"]
    result = df_to_safe_json(species_data, fields)
    return {"species": species_name, "num_records": len(result), "records": result}

# -----------------------------
# Endpoint: environment stats
# -----------------------------
@app.get("/species/{species_name}/environment")
def species_environment(species_name: str):
    species_data = df[df["scientificName"].str.lower() == species_name.lower()]
    if species_data.empty:
        raise HTTPException(status_code=404, detail="Species not found")
    
    env_fields = ["sst","sss","depth"]
    env_stats = species_data[env_fields].agg(["min","max","mean","median"]).replace([float("nan"), float("inf"), -float("inf")], None).to_dict()
    return {"species": species_name, "environment": env_stats}
