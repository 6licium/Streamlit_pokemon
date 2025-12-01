import requests
import pandas as pd
from tqdm import tqdm
import time

BASE = "https://pokeapi.co/api/v2/"

# ============================================================
# UTILITAIRES API (sécurisés contre les erreurs PokéAPI)
# ============================================================

def safe_get(url, retries=5, delay=1):
    """Effectue un GET fiable (PokéAPI rate souvent)."""
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        time.sleep(delay)
    raise Exception(f"❌ API failure: {url}")

# ============================================================
# BASE POKÉMON GEN 1 À 5
# ============================================================

def get_species_list():
    data = safe_get(BASE + "pokemon-species?limit=10000")
    return data["results"]

def get_species_data(url):
    return safe_get(url)

def get_evolution_chain(url):
    chain = safe_get(url)["chain"]
    evo = []
    node = chain
    while node:
        evo.append(node["species"]["name"])
        node = node["evolves_to"][0] if node["evolves_to"] else None
    return evo

def get_full_pokemon(pokemon_url, evo_url, generation):
    p = safe_get(pokemon_url)

    return {
        "id": p["id"],
        "name": p["name"],
        "generation": generation,
        "types": [t["type"]["name"] for t in p["types"]],
        "abilities": [a["ability"]["name"] for a in p["abilities"]],
        "moves": [m["move"]["name"] for m in p["moves"]],
        "sprite": p["sprites"]["front_default"],
        "evolution_chain": get_evolution_chain(evo_url)
    }

# ============================================================
# BASE ITEMS
# ============================================================

def get_items_list():
    data = safe_get(BASE + "item?limit=2000")
    return data["results"]

def get_item_data(url):
    d = safe_get(url)

    # Effet anglais
    effect = None
    for e in d.get("effect_entries", []):
        if e["language"]["name"] == "en":
            effect = e["effect"]

    return {
        "id": d["id"],
        "name": d["name"],
        "category": d["category"]["name"],
        "effect": effect,
        "sprite": d["sprites"]["default"],
    }

# ============================================================
# MAIN : MET À JOUR LES CSV
# ============================================================

def update_pokemon_database(
        pokemon_csv="pokemon_full_db.csv",
        items_csv="all_items.csv"
    ):
    print("🔄 Mise à jour de la base Pokémon…")

    # ----- 1) FILTRER GEN 1–5 -----
    species = get_species_list()

    gen15_species = []
    for sp in tqdm(species, desc="Filtrage Gen1–5"):
        d = get_species_data(sp["url"])
        gen = int(d["generation"]["url"].split("/")[-2])
        if gen <= 5:
            gen15_species.append({
                "name": sp["name"],
                "generation": gen,
                "pokemon_url": d["varieties"][0]["pokemon"]["url"],
                "evolution_chain_url": d["evolution_chain"]["url"]
            })

    # ----- 2) CONSTRUIRE BASE POKÉMON -----
    pokemon_rows = []
    for sp in tqdm(gen15_species, desc="Téléchargement Pokémon"):
        try:
            pokemon_rows.append(
                get_full_pokemon(
                    sp["pokemon_url"],
                    sp["evolution_chain_url"],
                    sp["generation"]
                )
            )
        except Exception as e:
            print(f"⚠️ Erreur Pokémon {sp['name']}: {e}")

    df_pokemon = pd.DataFrame(pokemon_rows)
    df_pokemon.to_csv(pokemon_csv, index=False)
    print(f"✅ Base mise à jour : {pokemon_csv}")

    # ----- 3) ITEMS -----
    print("🔄 Mise à jour des objets…")

    items = get_items_list()
    item_rows = []

    for it in tqdm(items, desc="Téléchargement Items"):
        try:
            item_rows.append(get_item_data(it["url"]))
        except Exception as e:
            print(f"⚠️ Erreur item {it['name']}: {e}")

    df_items = pd.DataFrame(item_rows)
    df_items.to_csv(items_csv, index=False)
    print(f"🎒 Base Items mise à jour : {items_csv}")

    print("✨ Mise à jour terminée !")
