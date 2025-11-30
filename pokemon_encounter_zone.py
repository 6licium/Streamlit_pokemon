import requests
import json
from collections import defaultdict

BASE = "https://pokeapi.co/api/v2"

def get(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def get_generation_regions():
    """Retourne pour chaque génération la liste des régions concernées."""
    mapping = {}
    for gen in range(1, 10):
        data = get(f"{BASE}/generation/{gen}")
        regions = list({v["region"]["name"] for v in data["version_groups"]})
        mapping[gen] = regions
    return mapping

def get_all_locations_by_region():
    """Retourne toutes les locations (et leurs location_areas) par région."""
    regions = get(f"{BASE}/region/").get("results")

    region_locations = defaultdict(list)
    for r in regions:
        reg_name = r["name"]
        reg_data = get(r["url"])

        for loc in reg_data["locations"]:
            loc_name = loc["name"]
            loc_data = get(loc["url"])

            # Ajouter chaque location-area
            for area in loc_data["areas"]:
                region_locations[reg_name].append({
                    "location": loc_name,
                    "location_area": area["name"]
                })

    return region_locations

# Récupération
region_locations = get_all_locations_by_region()

# Génération du fichier python contenant les dicts coords = (None, None)
output = ""

for region, entries in region_locations.items():
    output += f"{region}_coords = {{\n"
    for ent in entries:
        loc_name = ent["location"]
        output += f"    '{loc_name}': (None, None),\n"
    output += "}\n\n"

with open("regions_coords_template.py", "w") as f:
    f.write(output)

print("Fichier généré : regions_coords_template.py")
