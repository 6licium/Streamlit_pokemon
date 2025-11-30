import requests
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://pokeapi.co/api/v2"

def get_json(url):
    """Safe API call."""
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def get_generation_pokemon(generation_id):
    """Récupère tous les Pokémon d'une génération."""
    url = f"{BASE_URL}/generation/{generation_id}"
    data = get_json(url)
    return [p["name"] for p in data["pokemon_species"]]

def get_pokemon_id(name):
    """Déduit l'ID Pokémon à partir de son endpoint."""
    data = get_json(f"{BASE_URL}/pokemon/{name}")
    return data["id"]

def get_location_encounters(pokemon_id):
    """Récupère toutes les location area d'un Pokémon."""
    url = f"{BASE_URL}/pokemon/{pokemon_id}/encounters"
    try:
        locations = get_json(url)
    except:
        return []

    results = []
    for loc in locations:
        loc_area = loc["location_area"]["name"]

        # On récupère la REGION de la location-area
        loc_area_url = loc["location_area"]["url"]
        loc_area_data = get_json(loc_area_url)

        # La location-area -> location (ex: "kanto-route-1")
        if loc_area_data.get("location"):
            location_data = get_json(loc_area_data["location"]["url"])
            region = location_data["region"]["name"]
            location_name = location_data["name"]
        else:
            region = None
            location_name = None

        results.append({
            "pokemon_id": pokemon_id,
            "pokemon_name": get_json(f"{BASE_URL}/pokemon/{pokemon_id}")["name"],
            "location_area": loc_area,
            "location": location_name,
            "region": region,
        })

    return results

all_rows = []

# Générations 1 à 9 (PokéAPI s’arrête à G9)
for gen in range(1, 10):
    print(f"=== Génération {gen} ===")
    pokemon_list = get_generation_pokemon(gen)

    for name in tqdm(pokemon_list):
        try:
            pid = get_pokemon_id(name)
            encounters = get_location_encounters(pid)
            for e in encounters:
                e["generation"] = gen
                all_rows.append(e)
        except Exception as e:
            print(f"Erreur pour {name}: {e}")

# DataFrame final
df = pd.DataFrame(all_rows)

# Nettoyage et suppression doublons
df = df.drop_duplicates()

# Export CSV
df.to_csv("pokemon_location_encounters_full.csv", index=False)

print("Fichier généré : pokemon_location_encounters_full.csv")
