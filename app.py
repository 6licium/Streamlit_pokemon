# app.py
import os
import time
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from collections import defaultdict
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
from coords import regions_coords  # ton fichier existant

# ---------------------------
# Config & styles
# ---------------------------
st.set_page_config(page_title="Pokédex Avancé", layout="wide")
st.markdown(
    """
<style>
    .block-container { padding-top: 0rem !important; }
    h1 { text-align: center !important; margin-top: 0px !important; }
    .map-container { display: flex; justify-content: center; }
    .stPlotlyChart { max-width: 900px; width: 900px; margin: 0 auto; }
    .pokemon-list { margin-top: 20px; }
    .pokemon-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
    .pokemon-item:hover { background-color: #272727; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Constants & filenames
# ---------------------------
POKEMON_CSV = "pokemon_full_db_gen1_5.csv"
ITEMS_CSV = "all_items.csv"
BASE_API = "https://pokeapi.co/api/v2/"
POKEMON_LIMIT = 800  # couvert dans tes précédents essais (Gen1-5 well below 800)

def eval_list(s):
    """
    If string looks like a Python list saved in CSV, convert to list safely.
    If it's already a list, return it.
    """
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    # try a simple parse: items separated by commas or '|' for evolution chain
    if isinstance(s, str):
        if s.startswith("[") or s.startswith("'") or ("," in s and ("[" in s or "]" in s) is False):
            # try to split by comma
            parts = [p.strip().strip("'\"") for p in s.split(",")]
            return [p for p in parts if p]
        if "|" in s:
            return [x.strip() for x in s.split("|")]
    return [s]

# ---------------------------
# Helpers: safe API get + small utilities
# ---------------------------
def safe_get(url, retries=4, backoff=1.0, timeout=10):
    """Robust GET with retries. Returns JSON or raises."""
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(backoff * (i + 1))
    raise RuntimeError(f"API unreachable or failed for URL: {url}")


def load_image_as_base64(path):
    img = Image.open(path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue(), base64.b64encode(buffer.getvalue()).decode()


# ---------------------------
# Core: build/update DB (Gen1→Gen5 + items)
# ---------------------------
@st.cache_data
def fetch_species_list():
    """Return all species results from API (name + url)."""
    url = BASE_API + "pokemon-species?limit=10000"
    data = safe_get(url)
    return data["results"]


def _get_evolution_chain_names(chain_obj):
    """internal: traverse an evolution chain object recursively and return list of species names (order)."""
    out = []
    node = chain_obj
    while node:
        out.append(node["species"]["name"])
        node = node["evolves_to"][0] if node["evolves_to"] else None
    return out


def build_db_gen1_5(progress_callback=None):
    """
    Build two CSVs:
    - POKEMON_CSV: records for Pokemon introduced in Gen 1..5
    - ITEMS_CSV: items list
    This function is intentionally not cached (called on-demand), but uses safe_get to fetch.
    """
    # 1) Get species list and keep only generation <= 5
    species_results = fetch_species_list()
    gen15 = []
    for sp in species_results:
        try:
            s = safe_get(sp["url"])
            gen_id = int(s["generation"]["url"].split("/")[-2])
            if gen_id <= 5:
                # keep the first variety pokemon URL (usually default)
                pokemon_url = s["varieties"][0]["pokemon"]["url"]
                evo_url = s["evolution_chain"]["url"]
                gen15.append(
                    {
                        "species_name": s["name"],
                        "species_id": int(s["id"]),
                        "generation": gen_id,
                        "pokemon_url": pokemon_url,
                        "evolution_chain_url": evo_url,
                    }
                )
        except Exception as e:
            # skip if error for this species
            print(f"Warning species {sp['name']} failed: {e}")

    rows = []
    total = len(gen15)
    for idx, sp in enumerate(gen15):
        # progress update callback for UI
        if progress_callback:
            progress_callback(idx / total)
        try:
            p = safe_get(sp["pokemon_url"])
            types = [t["type"]["name"] for t in p["types"]]
            abilities = [a["ability"]["name"] for a in p["abilities"]]
            moves = [m["move"]["name"] for m in p["moves"]]
            sprite = p["sprites"]["front_default"]

            # evolution chain (names)
            evo_data = safe_get(sp["evolution_chain_url"])
            evo_chain = _get_evolution_chain_names(evo_data["chain"])

            total_stats = sum([s["base_stat"] for s in p["stats"]])

            rows.append(
                {
                    "pokemon_id": p["id"],
                    "name": p["name"],
                    "generation": sp["generation"],
                    "type_1": types[0] if len(types) > 0 else None,
                    "type_2": types[1] if len(types) > 1 else None,
                    "abilities": abilities,
                    "moves": moves,
                    "sprite": sprite,
                    "evolution_chain": "|".join(evo_chain),
                    "total_stats": int(total_stats),
                }
            )
        except Exception as e:
            print(f"Warning: failed to fetch pokemon {sp['species_name']}: {e}")

    df_pokemon = pd.DataFrame(rows)
    df_pokemon.to_csv(POKEMON_CSV, index=False)

    # Items
    items_list = safe_get(BASE_API + "item?limit=2000")["results"]
    item_rows = []
    for it in items_list:
        try:
            d = safe_get(it["url"])
            effect = None
            for entry in d.get("effect_entries", []):
                if entry["language"]["name"] == "en":
                    effect = entry.get("effect")
                    break
            item_rows.append(
                {
                    "item_id": d["id"],
                    "name": d["name"],
                    "category": d["category"]["name"],
                    "effect": effect,
                    "sprite": d.get("sprites", {}).get("default"),
                }
            )
        except Exception as e:
            print(f"Warning item {it['name']} failed: {e}")

    df_items = pd.DataFrame(item_rows)
    df_items.to_csv(ITEMS_CSV, index=False)

    if progress_callback:
        progress_callback(1.0)
    return True


# ---------------------------
# Cached loaders (fast)
# ---------------------------
@st.cache_data
def load_local_pokemon_df():
    if os.path.exists(POKEMON_CSV):
        return pd.read_csv(POKEMON_CSV)
    return None


@st.cache_data
def load_local_items_df():
    if os.path.exists(ITEMS_CSV):
        return pd.read_csv(ITEMS_CSV)
    return None


# ---------------------------
# Reuse your existing get_pokemon_data and get_ability_data
# (cached to avoid repeated API calls)
# ---------------------------
@st.cache_data
def get_pokemon_data(pokemon_id):
    try:
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}")
        pokemon_data = response.json()

        abilities = [a["ability"]["name"].replace("-", " ").capitalize() for a in pokemon_data["abilities"]]

        species_response = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}")
        species_data = species_response.json()
        evolution_chain_url = species_data["evolution_chain"]["url"]

        evolution_response = requests.get(evolution_chain_url)
        evolution_data = evolution_response.json()

        types = [t["type"]["name"] for t in pokemon_data["types"]]
        stats = {s["stat"]["name"]: s["base_stat"] for s in pokemon_data["stats"]}

        def get_evolution_chain(chain):
            evolutions = []
            current = chain
            while current:
                species = current["species"]
                evolutions.append({"name": species["name"], "id": int(species["url"].split("/")[-2])})
                if current["evolves_to"]:
                    current = current["evolves_to"][0]
                else:
                    current = None
            return evolutions

        evolution_chain = get_evolution_chain(evolution_data["chain"])

        return {
            "name": pokemon_data["name"],
            "id": pokemon_data["id"],
            "types": types,
            "stats": stats,
            "sprite": pokemon_data["sprites"]["front_default"],
            "evolution_chain": evolution_chain,
            "abilities": abilities,
        }
    except Exception as e:
        # Avoid st.error in cached function — raise to caller to handle
        raise RuntimeError(f"Erreur get_pokemon_data: {e}")


@st.cache_data
def get_ability_data(ability_name):
    try:
        api_name = (
            ability_name.lower()
            .replace("é", "e")
            .replace(" ", "-")
            .replace("_", "-")
            .replace("–", "-")
            .strip()
        )
        if "(" in api_name:
            api_name = api_name.split("(")[0].strip()

        url = f"https://pokeapi.co/api/v2/ability/{api_name}"
        response = requests.get(url)
        if response.status_code != 200:
            return f"Impossible de récupérer le talent : {ability_name} (URL : {url})"
        data = response.json()

        for e in data.get("effect_entries", []):
            if e["language"]["name"] == "fr":
                return e["effect"]
        for e in data.get("effect_entries", []):
            if e["language"]["name"] == "en":
                return e["effect"]
        return "Aucune description disponible."
    except Exception as e:
        return f"Erreur lors de la récupération du talent : {e}"


# ---------------------------
# UI: top controls (update DB)
# ---------------------------
st.title("📘 Pokédex Multifeatures — Optimized")

col_update, col_info = st.columns([1, 3])
with col_update:
    if st.button("🔄 Actualiser la base (Gen1→Gen5 + items)"):
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress_cb(p):
            progress_bar.progress(min(max(p, 0.0), 1.0))
            status.text(f"Progress: {int(p*100)}%")

        # Build DB (blocking call)
        try:
            build_db_gen1_5(progress_callback=progress_cb)
            # clear cached loaders so streamlit will re-read CSVs
            load_local_pokemon_df.clear()
            load_local_items_df.clear()
            st.success("Bases mises à jour (CSV écrits). Recharge l'app si nécessaire.")
        except Exception as e:
            st.error(f"Erreur lors de la mise à jour: {e}")
        progress_bar.empty()
        status.empty()

with col_info:
    st.markdown(
        """
        **Mode d'emploi rapide**  
        - Clique sur *Actualiser la base* pour créer/mettre à jour `pokemon_full_db_gen1_5.csv` & `all_items.csv`.  
        - L'app utilise ensuite ces CSVs (chargement local rapide).  
        """
    )


# ---------------------------
# Load local CSVs (fast) or warn user
# ---------------------------
df_pokemon = load_local_pokemon_df()
df_items = load_local_items_df()

if df_pokemon is None or df_items is None:
    st.warning(
        f"Fichiers CSV non trouvés ({POKEMON_CSV} ou {ITEMS_CSV}). Clique sur 'Actualiser la base' pour les générer depuis PokéAPI."
    )

# ---------------------------
# Session state init
# ---------------------------
if "clicked_location" not in st.session_state:
    st.session_state.clicked_location = None
if "selected_pokemon" not in st.session_state:
    st.session_state.selected_pokemon = None
if "selected_ability" not in st.session_state:
    st.session_state.selected_ability = None

# ---------------------------
# Build layout tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Localisation par Génération", "📊 Statistiques Globales", "🧩 Team Builder", "⚔️ Comparateur de Pokémon"]
)

# ---------------------------
# TAB 1 : Location map (uses user's CSV pokemon_location_encounters_full.csv)
# ---------------------------
with tab1:
    st.header("📍 Localisation des Pokémon par Génération / Cartes")

    # region selection and map code re-used from your original app, but optimized
    regions = list(regions_coords.keys())
    region = st.selectbox("Choisir une région", regions)

    search_term_global = st.text_input("Filtrer les lieux par Pokémon (nom)", value="", key="search_global_tab1")

    # load encounters CSV (unchanged)
    @st.cache_data
    def load_encounters():
        return pd.read_csv("pokemon_location_encounters_full.csv")
    df_enc = load_encounters()
    df_region = df_enc[df_enc["region"] == region]

    col1, col2 = st.columns([1.5, 2])
    with col1:
        try:
            img, encoded_img = load_image_as_base64(f"cartes/carte_{region.lower()}.png")
            img_w, img_h = Image.open(BytesIO(img)).size
        except FileNotFoundError:
            st.error(f"Carte introuvable pour {region}")
            st.stop()

        # group pokemon by location and only keep unique species per location
        pokemon_by_location = defaultdict(list)
        for _, row in df_region.iterrows():
            pokemon_by_location[row["location"]].append(row.to_dict())

        x_points, y_points, locations, hover_texts = [], [], [], []
        for lieu, pokes in pokemon_by_location.items():
            if lieu in regions_coords[region]:
                unique_pokes = {p["name"]: p for p in pokes}.values()
                if search_term_global and not any(search_term_global.lower() in p["name"].lower() for p in unique_pokes):
                    continue
                x, y = regions_coords[region][lieu]
                x_points.append(x)
                y_points.append(y)
                locations.append(lieu)
                hover_texts.append(f"<b>{lieu}</b><br>{len(list(unique_pokes))} Pokémon<br>Cliquez pour voir la liste")

        FIXED_WIDTH = 900
        FIXED_HEIGHT = int(FIXED_WIDTH * img_h / img_w)

        fig = go.Figure()
        fig.add_layout_image(dict(source=f"data:image/png;base64,{encoded_img}", xref="x", yref="y", x=0, y=FIXED_HEIGHT,
                                  sizex=FIXED_WIDTH, sizey=FIXED_HEIGHT, sizing="stretch", layer="below"))
        fig.add_trace(go.Scatter(
            x=x_points, y=[FIXED_HEIGHT - y for y in y_points],
            mode="markers",
            marker=dict(size=14, color="rgba(255,215,0,0.8)", line=dict(width=1, color="rgba(200,180,0,0.7)")),
            customdata=locations, hoverinfo="text", hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>"
        ))
        fig.update_xaxes(visible=False, range=[0, FIXED_WIDTH], fixedrange=True)
        fig.update_yaxes(visible=False, range=[0, FIXED_HEIGHT], fixedrange=True)
        fig.update_layout(width=FIXED_WIDTH, height=FIXED_HEIGHT, margin=dict(l=0, r=0, t=0, b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest")

        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        selected_point = plotly_events(fig, click_event=True, override_width=FIXED_WIDTH, override_height=FIXED_HEIGHT)
        st.markdown('</div>', unsafe_allow_html=True)

    # right column: list and details (optimized)
    with col2:
        if selected_point:
            pt = selected_point[0]
            idx = pt["pointIndex"]
            if st.session_state.clicked_location != locations[idx]:
                st.session_state.selected_pokemon = None
            st.session_state.clicked_location = locations[idx]

        if st.session_state.clicked_location:
            lieu = st.session_state.clicked_location
            pokemons = pokemon_by_location[lieu]
            unique_pokemons = list({p["name"]: p for p in pokemons}.values())
            filtered = [p for p in unique_pokemons if (not search_term_global) or (search_term_global.lower() in p["name"].lower())]

            st.markdown(f"<div class='pokemon-list'><h3>Pokémon à {lieu} ({len(unique_pokemons)} au total, {len(filtered)} filtrés)</h3></div>", unsafe_allow_html=True)

            if filtered:
                for poke in filtered:
                    colA, colB = st.columns([4, 1])
                    with colA:
                        st.markdown(f"<div class='pokemon-item'><span><b>{poke['name']}</b> (ID: {poke['pokemon_id']})</span></div>", unsafe_allow_html=True)
                    with colB:
                        # unique key prevents duplicates even if same pokemon at different places
                        key = f"info_{poke['name']}_{lieu}"
                        if st.button("Plus d'infos", key=key):
                            st.session_state.selected_pokemon = poke
                            st.session_state.selected_ability = None
            else:
                st.warning("Aucun Pokémon ne correspond à votre recherche.")

    # details
    if st.session_state.selected_pokemon:
        poke = st.session_state.selected_pokemon
        try:
            pokemon_data = get_pokemon_data(poke["pokemon_id"])
        except Exception as e:
            st.error(f"Impossible de récupérer les données détaillées: {e}")
            pokemon_data = None

        if pokemon_data:
            st.markdown(f"### {pokemon_data['name'].capitalize()} (ID: {pokemon_data['id']})")
            c1, c2, c3 = st.columns([1, 1.5, 1])
            with c1:
                st.image(pokemon_data["sprite"], width=220)
                st.markdown("#### Évolution")
                if pokemon_data["evolution_chain"]:
                    evo_html = '<div style="display:flex; gap:15px; flex-wrap:wrap; justify-content:center">'
                    for i, evo in enumerate(pokemon_data["evolution_chain"]):
                        if i > 0:
                            evo_html += '<div style="font-size:18px;color:#666;">→</div>'
                        evo_html += f'<div style="text-align:center;"><img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{evo["id"]}.png" style="width:60px;height:60px;"><div style="font-size:12px">{evo["name"].capitalize()}</div></div>'
                    evo_html += "</div>"
                    st.markdown(evo_html, unsafe_allow_html=True)
                else:
                    st.markdown("Pas d'évolution connue")
            with c2:
                st.markdown("#### Types")
                type_colors = {
                    "normal": "#A8A77A","fire": "#EE8130","water": "#6390F0","electric": "#F7D02C",
                    "grass": "#7AC74C","ice": "#96D9D6","fighting": "#C22E28","poison": "#A33EA1",
                    "ground": "#E2BF65","flying": "#A98FF3","psychic": "#F95587","bug": "#A6B91A",
                    "rock": "#B6A136","ghost": "#735797","dragon": "#6F35FC","dark": "#705746",
                    "steel": "#B7B7CE","fairy": "#D685AD"
                }
                badges = "".join([f'<span style="background-color:{type_colors.get(t,"#68A090")}; padding:5px 10px; border-radius:12px; color:white; margin-right:8px;">{t.capitalize()}</span>' for t in pokemon_data["types"]])
                st.markdown(f"<div style='margin-bottom:15px'>{badges}</div>", unsafe_allow_html=True)

                st.markdown("#### Statistiques")
                stat_names = {'hp':'PV','attack':'Attaque','defense':'Défense','special-attack':'Attaque Spé.','special-defense':'Défense Spé.','speed':'Vitesse'}
                stat_colors = {'hp':'#FF5959','attack':'#F5AC78','defense':'#FAE078','special-attack':'#9DB7F5','special-defense':'#A7DB8D','speed':'#FA92B2'}

                for stat_name, stat_value in pokemon_data["stats"].items():
                    display_name = stat_names.get(stat_name, stat_name.capitalize())
                    color = stat_colors.get(stat_name, "#4CAF50")
                    capped_value = max(0, min(stat_value, 200))
                    percent = (capped_value / 200) * 100
                    st.markdown(f"""
                    <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-weight:bold; margin-bottom:4px;">
                        <span>{display_name}</span><span>{stat_value}</span>
                    </div>
                    <div style="height:8px; background:#e0e0e0; border-radius:4px; overflow:hidden;">
                        <div style="height:8px; background:{color}; width:{percent}%"></div>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

            with c3:
                st.markdown("#### Talents")
                abilities = pokemon_data["abilities"]
                if abilities:
                    cols = st.columns(len(abilities))
                    for i, ability in enumerate(abilities):
                        if cols[i].button(ability, key=f"ability_{ability}_{poke['pokemon_id']}"):
                            st.session_state.selected_ability = ability
                    if st.session_state.selected_ability:
                        desc = get_ability_data(st.session_state.selected_ability)
                        st.markdown(f"**{st.session_state.selected_ability} :** {desc}")
                else:
                    st.markdown("Aucun talent listé.")

# ---------------------------
# TAB 2 : Stats Globales (optimized local filtering)
# ---------------------------
with tab2:
    st.header("📊 Statistiques Globales sur les Pokémon (Local CSV)")

    if df_pokemon is None:
        st.info("Génère d'abord le CSV via le bouton 'Actualiser la base' pour obtenir les statistiques complètes.")
    else:
        # Mapping génération -> région
        gen_to_region = {
            "1": "Kanto",
            "2": "Johto",
            "3": "Hoenn",
            "4": "Sinnoh",
            "5": "Unys",
        }

        # GRAPH 1: Nombre de Pokémon par génération (non filtré)
        st.subheader("📌 Nombre de Pokémon par génération")
        df_g1 = df_pokemon.copy()
        counts_gen = df_g1["generation"].value_counts().sort_index()
        labels = [f"Gen {int(i)}" for i in counts_gen.index]
        fig1 = go.Figure()
        fig1.add_bar(x=labels, y=counts_gen.values)
        fig1.update_layout(
            title="Nombre de Pokémon par génération",
            xaxis_title="Génération",
            yaxis_title="Nombre"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Séparateur visuel
        st.divider()

        # Sélecteurs pour les graphiques 2 et 3
        st.subheader("🔧 Filtres pour les graphiques suivants")
        col1, col2 = st.columns(2)
        with col1:
            region_choice = st.selectbox(
                "Filtrer par région :",
                ["Toutes"] + list(gen_to_region.values()),
                key="region_select_tab2"
            )
        with col2:
            all_types = sorted({t for t in pd.concat([df_pokemon["type_1"], df_pokemon["type_2"]]).dropna().unique()})
            type_choice = st.multiselect(
                "Filtrer par type :",
                all_types,
                key="type_select_tab2"
            )

        # Helper filter for graphs 2 and 3 (utilise uniquement le CSV)
        @st.cache_data
        def filter_local(df_local, types=None, region=None):
            df_loc = df_local.copy()
            if region and region != "Toutes":
                gen = [k for k, v in gen_to_region.items() if v == region][0]
                df_loc = df_loc[df_loc["generation"].astype(str) == gen]
            if types:
                df_loc = df_loc[(df_loc["type_1"].isin(types)) | (df_loc["type_2"].isin(types))]
            return df_loc

        # GRAPH 2: Camembert du nombre de Pokémon par type (filtré)
        st.subheader("📌 Répartition des Pokémon par type")
        df_g2 = filter_local(df_pokemon, type_choice if type_choice else None, region_choice)
        type_counts = pd.Series(dtype=float)
        for col in ["type_1", "type_2"]:
            type_counts = type_counts.add(df_g2[col].value_counts(), fill_value=0)
        type_counts = type_counts.sort_values(ascending=False)

        fig2 = go.Figure()
        fig2.add_pie(
            labels=[t.capitalize() for t in type_counts.index],
            values=type_counts.values,
            hole=0.3,  # Ajoute un trou pour un style "donut"
            marker_colors=px.colors.qualitative.Plotly  # Palette de couleurs attrayante
        )
        fig2.update_layout(
            title=f"Répartition des Pokémon par type ({'Toutes régions' if region_choice=='Toutes' else region_choice})",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Séparateur visuel
        st.divider()

        # GRAPH 3: Distribution total_stats (filtré)
        st.subheader("📌 Distribution du total des statistiques de base")
        df_g3 = filter_local(df_pokemon, type_choice if type_choice else None, region_choice)
        df_g3 = df_g3[df_g3["total_stats"].notna()]

        fig3 = go.Figure()
        fig3.add_histogram(
            x=df_g3["total_stats"],
            nbinsx=30,
            marker_color='#636EFA',  # Couleur bleue par défaut
            opacity=0.7
        )
        fig3.update_layout(
            title=f"Distribution du total des stats (base) ({'Toutes régions' if region_choice=='Toutes' else region_choice})",
            xaxis_title="Total stats",
            yaxis_title="Nombre",
            bargap=0.1
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Ajout d'un résumé statistique
        st.divider()
        st.subheader("📊 Résumé statistique")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Nombre total de Pokémon", len(df_pokemon))
        with col_b:
            st.metric("Moyenne des stats totales", f"{df_pokemon['total_stats'].mean():.1f}")
        with col_c:
            st.metric("Écart-type des stats", f"{df_pokemon['total_stats'].std():.1f}")

# ---------------------------
# TAB 3 : Team Builder (uses local df_pokemon & df_items)
# ---------------------------
with tab3:
    st.header("🧩 Générateur de Team Pokémon (6 slots)")

    if df_pokemon is None or df_items is None:
        st.info("Génère d'abord les CSV via 'Actualiser la base' pour utiliser le Team Builder.")
    else:
        # create mapping id->name for selectboxes
        choices = ["--- Choisir ---"] + sorted(df_pokemon["name"].str.capitalize().tolist())
        cols_team = st.columns(3)
        team = [None] * 6
        for i in range(6):
            with cols_team[i % 3]:
                pick = st.selectbox(f"Slot {i+1}", choices, key=f"slot_{i}")
                if pick != "--- Choisir ---":
                    team[i] = pick
                    st.write(f"**{pick}**")
                    # show quick info: types, total stats, abilities
                    rec = df_pokemon[df_pokemon["name"].str.lower() == pick.lower()].iloc[0]
                    st.write(f"Types: {rec['type_1']}{' / ' + rec['type_2'] if pd.notna(rec['type_2']) else ''}")
                    st.write(f"Total stats: {rec['total_stats']}")
                    st.write(f"Talents: {', '.join(eval_list(rec['abilities']) if 'abilities' in rec else rec.get('abilities', [])) if 'abilities' in rec else ''}")
        st.info("Tu peux améliorer ce builder (objets, movesets, compatibilité, export/import).")

# ---------------------------
# TAB 4 : Comparateur 2x2
# ---------------------------
with tab4:
    st.header("⚔️ Comparateur de Pokémon (2 vs 2)")

    if df_pokemon is None:
        st.info("Génère d'abord le CSV via 'Actualiser la base' pour comparer.")
    else:
        names = sorted(df_pokemon["name"].str.capitalize().tolist())
        colA, colB = st.columns(2)
        with colA:
            a = st.selectbox("Pokémon A", ["---"] + names, key="comp_a")
        with colB:
            b = st.selectbox("Pokémon B", ["---"] + names, key="comp_b")

        if a and b and a != "---" and b != "---":
            ra = df_pokemon[df_pokemon["name"].str.lower() == a.lower()].iloc[0]
            rb = df_pokemon[df_pokemon["name"].str.lower() == b.lower()].iloc[0]
            # show a compact comparison table
            comp_df = pd.DataFrame({
                "Stat": ["Generation", "Type 1", "Type 2", "Total stats"],
                a: [ra["generation"], ra["type_1"], ra["type_2"] if pd.notna(ra["type_2"]) else "-", ra["total_stats"]],
                b: [rb["generation"], rb["type_1"], rb["type_2"] if pd.notna(rb["type_2"]) else "-", rb["total_stats"]],
            })
            st.table(comp_df)
