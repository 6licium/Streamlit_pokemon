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
from coords import regions_coords

# ---------------------------
# Config & styles
# ---------------------------
st.set_page_config(page_title="Pokédex Avancé", layout="wide")
st.markdown(
    """
<style>
    /* Correction pour le titre */
    .stApp { padding-top: 0 !important; }
    .st-emotion-cache-1v0mbdj { padding-top: 0 !important; }
    h1 { margin-bottom: 1.5rem !important; margin-top: 1rem !important; }
    .stTabs { margin-top: 1rem !important; }
    /* Correction pour la carte */
    .stPlotlyChart { margin-top: 1rem !important; }
    /* Correction pour les colonnes */
    .stColumn { gap: 1rem !important; }
</style>
""",
    unsafe_allow_html=True,
)



# ---------------------------
# Constants & filenames
# ---------------------------
POKEMON_CSV = "pokemon_full_db_gen1_5.csv"
ITEMS_CSV = "all_items.csv"
ENCOUNTERS_CSV = "pokemon_location_encounters_full.csv"
BASE_API = "https://pokeapi.co/api/v2/"

# ---------------------------
# Utilities
# ---------------------------
def eval_list(s):
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        if s.startswith("[") or s.startswith("'") or ("," in s and ("[" in s or "]" in s) is False):
            parts = [p.strip().strip("'\"") for p in s.split(",")]
            return [p for p in parts if p]
        if "|" in s:
            return [x.strip() for x in s.split("|")]
    return [s]

def safe_get(url, retries=3, backoff=1.0, timeout=10):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(backoff)
    raise RuntimeError(f"API unreachable or failed for URL: {url}")

def load_image_as_base64(path):
    img = Image.open(path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue(), base64.b64encode(buffer.getvalue()).decode()

# ---------------------------
# Cached loaders (fast)
# ---------------------------
@st.cache_data
def load_local_pokemon_df():
    return pd.read_csv(POKEMON_CSV) if os.path.exists(POKEMON_CSV) else None

@st.cache_data
def load_local_items_df():
    return pd.read_csv(ITEMS_CSV) if os.path.exists(ITEMS_CSV) else None

@st.cache_data
def load_encounters():
    return pd.read_csv(ENCOUNTERS_CSV) if os.path.exists(ENCOUNTERS_CSV) else None

# ---------------------------
# API fetchers (only when needed)
# ---------------------------
@st.cache_data
def get_pokemon_data(pokemon_id):
    try:
        response = requests.get(f"{BASE_API}pokemon/{pokemon_id}")
        response.raise_for_status()
        pokemon_data = response.json()

        abilities = [a["ability"]["name"].replace("-", " ").capitalize() for a in pokemon_data["abilities"]]
        types = [t["type"]["name"] for t in pokemon_data["types"]]
        stats = {s["stat"]["name"]: s["base_stat"] for s in pokemon_data["stats"]}

        # Get evolution chain
        species_response = requests.get(f"{BASE_API}pokemon-species/{pokemon_id}")
        species_response.raise_for_status()
        species_data = species_response.json()

        evolution_response = requests.get(species_data["evolution_chain"]["url"])
        evolution_response.raise_for_status()
        evolution_data = evolution_response.json()

        def get_evolution_chain(chain):
            evolutions = []
            current = chain
            while current:
                species = current["species"]
                evolutions.append({"name": species["name"], "id": int(species["url"].split("/")[-2])})
                current = current["evolves_to"][0] if current["evolves_to"] else None
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

        url = f"{BASE_API}ability/{api_name}"
        response = requests.get(url)
        response.raise_for_status()
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
# UI: top controls
# ---------------------------
st.title("📘 Pokédex Multifeatures")

col_update, col_info = st.columns([1, 3])
with col_update:
    if st.button("🔄 Actualiser la base (Gen1→Gen5 + items)"):
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress_cb(p):
            progress_bar.progress(min(max(p, 0.0), 1.0))
            status.text(f"Progression: {int(p*100)}%")

        try:
            # Ici tu devrais avoir ta fonction build_db_gen1_5
            # build_db_gen1_5(progress_callback=progress_cb)
            st.success("Bases mises à jour. Recharge l'app si nécessaire.")
        except Exception as e:
            st.error(f"Erreur lors de la mise à jour: {e}")
        finally:
            progress_bar.empty()
            status.empty()

with col_info:
    st.markdown(
        """
        **Mode d'emploi rapide**
        - Clique sur *Actualiser la base* pour la créer/mettre à jour.
        - L'app utilise ensuite ces fichiers.
        """
    )

# ---------------------------
# Load local CSVs
# ---------------------------
df_pokemon = load_local_pokemon_df()
df_items = load_local_items_df()
df_enc = load_encounters()

if df_pokemon is None or df_items is None:
    st.warning("Fichiers CSV manquants. Utilise le bouton 'Actualiser la base' pour les générer.")

# ---------------------------
# Session state init
# ---------------------------
for key in ["clicked_location", "selected_pokemon", "selected_ability"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------
# Mapping génération -> région
# ---------------------------
gen_to_region = {
    "1": "Kanto",
    "2": "Johto",
    "3": "Hoenn",
    "4": "Sinnoh",
    "5": "Unys",
}

# ---------------------------
# Build layout tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Localisation", "📊 Statistiques", "🧩 Team Builder", "⚔️ Comparateur"]
)

# ---------------------------
# TAB 1 : Location map (uses user's CSV pokemon_location_encounters_full.csv)
# ---------------------------
with tab1:
    st.header("📍 Localisation des Pokémon par Génération / Cartes")

    # Reset des sélections quand on change de région
    if "current_region" not in st.session_state:
        st.session_state.current_region = None

    # region selection and map code re-used from your original app, but optimized
    regions = list(regions_coords.keys())
    region = st.selectbox("Choisir une région", regions)

    if st.session_state.current_region != region:
        st.session_state.clicked_location = None
        st.session_state.selected_pokemon = None
        st.session_state.selected_ability = None
        st.session_state.current_region = region

    search_term_global = st.text_input("Filtrer les lieux par Pokémon (nom)", value="", key="search_global_tab1")

    # load encounters CSV (unchanged)

    df_enc = load_encounters()
    df_region = df_enc[df_enc["region"] == region]

    col1, col2 = st.columns([1.2, 2])
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
            marker=dict(size=15, color="rgba(150,50,100,0.8)", line=dict(width=1, color="rgba(200,180,0,0.7)")),
            customdata=locations, hoverinfo="text", hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>"
        ))
        fig.update_xaxes(visible=False, range=[0, FIXED_WIDTH], fixedrange=True)
        fig.update_yaxes(visible=False, range=[0, FIXED_HEIGHT], fixedrange=True)
        fig.update_layout(width=FIXED_WIDTH, height=FIXED_HEIGHT, margin=dict(l=0, r=0, t=0, b=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest")

        event = st.plotly_chart(fig, on_select="rerun", selection_mode=["points"])
        selected_point = event["selection"].get("points", [])

    # right column: list and details (optimized)
    with col2:
        if selected_point:
            pt = selected_point[0]
            idx = pt["point_index"]
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
# TAB 2 : Stats Globales (100% local)
# ---------------------------
with tab2:
    st.header("📊 Statistiques Globales")

    if df_pokemon is None:
        st.info("Génère d'abord les CSV pour voir les statistiques.")
    else:
        # Filtres
        st.subheader("🔧 Filtres")
        col1, col2 = st.columns(2)
        with col1:
            region_choice = st.selectbox(
                "Région:",
                ["Toutes"] + list(gen_to_region.values()),
                key="region_select"
            )

            st.divider()
        with col2:
            all_types = sorted({t for t in pd.concat([df_pokemon["type_1"], df_pokemon["type_2"]]).dropna().unique()})
            type_choice = st.multiselect(
                "Types:",
                all_types,
                key="type_select"
            )

            st.divider()

        # Fonction de filtrage
        @st.cache_data
        def filter_data(types=None, region=None):
            df = df_pokemon.copy()
            if region and region != "Toutes":
                gen = [k for k, v in gen_to_region.items() if v == region][0]
                df = df[df["generation"].astype(str) == gen]
            if types:
                df = df[(df["type_1"].isin(types)) | (df["type_2"].isin(types))]
            return df
        
        st.divider()

        with col1:
            # Graphique 1: Nombre par génération
            st.subheader("📊 Nombre de Pokémon par génération")

            # Filtrer uniquement par type (sans filtre de région)
            filtered_df_gen = filter_data(types=type_choice)

            # Calcul des comptes par génération sur les données filtrées
            gen_counts = filtered_df_gen["generation"].value_counts().sort_index()

            # Création du graphique
            fig1 = go.Figure()
            fig1.add_bar(
                x=[f"Gen {g}" for g in gen_counts.index],
                y=gen_counts.values,
                marker_color=px.colors.qualitative.Plotly
            )

            # Mise à jour du titre pour refléter les filtres de type
            title = "Pokémon par génération"
            if type_choice:
                title += f" (Types: {', '.join(type_choice)})"

            fig1.update_layout(
                title=title,
                xaxis_title="Génération",
                yaxis_title="Nombre"
            )
            st.plotly_chart(fig1, use_container_width=True)


            # st.divider()

        with col2:
        # Graphique 2: Répartition par type
            st.subheader("🎨 Répartition par type")
            filtered_df = filter_data(type_choice, region_choice)
            type_counts = pd.Series(dtype=float)
            for col in ["type_1", "type_2"]:
                type_counts = type_counts.add(filtered_df[col].value_counts(), fill_value=0)
            type_counts = type_counts.sort_values(ascending=False)

            fig2 = go.Figure()
            fig2.add_pie(
                labels=[t.capitalize() for t in type_counts.index],
                values=type_counts.values,
                hole=0.3,
                marker_colors=px.colors.qualitative.Plotly
            )
            fig2.update_layout(
                title=f"Répartition par type ({region_choice})",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # st.divider()

        with col1:
        # Graphique 3: Distribution des stats
            st.subheader("📈 Distribution des statistiques totales")
            stats_df = filter_data(type_choice, region_choice)
            if "total_stats" in stats_df.columns:
                stats_df = stats_df[stats_df["total_stats"].notna()]

                fig3 = go.Figure()
                fig3.add_histogram(
                    x=stats_df["total_stats"],
                    nbinsx=30,
                    marker_color="#636EFA",
                    opacity=0.7
                )
                fig3.update_layout(
                    title=f"Distribution des stats totales ({region_choice})",
                    xaxis_title="Stats totales",
                    yaxis_title="Nombre"
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Colonne 'total_stats' manquante dans les données.")
            # Résumé statistique
            # st.divider()
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total", len(stats_df))
            with col_b:
                st.metric("Moyenne", f"{stats_df['total_stats'].mean():.1f}")
            with col_c:
                st.metric("Écart-type", f"{stats_df['total_stats'].std():.1f}")
        

with tab3:
    st.header("🧩 Générateur de Team Pokémon (6 slots)")

    # --- Initialisation de l'équipe dans session_state ---
    if "team" not in st.session_state:
        st.session_state.team = [None] * 6

    if df_pokemon is None or df_items is None:
        st.warning("Génère d'abord les CSV pour utiliser le Team Builder.")
    else:
        # --- Filtres pour la sélection ---
        st.subheader("🔍 Filtres de sélection")
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            type_filter = st.multiselect(
                "Filtrer par type:",
                options=sorted({t for t in pd.concat([df_pokemon["type_1"], df_pokemon["type_2"]]).dropna().unique()}),
                key="type_filter_team"
            )
        with col_filter2:
            search_name = st.text_input(
                "Rechercher par nom:",
                placeholder="Ex: pikachu",
                key="search_name_team"
            )

        # --- Fonction pour nettoyer les noms d'attaques ---
        def clean_move_name(move_name):
            return move_name.replace("'", "").replace("[", "").replace("]", "").replace("-", " ").title()

        # --- Préparation des données filtrées ---
        @st.cache_data
        def get_filtered_pokemon(type_filter=None, search_name=None):
            df = df_pokemon.copy()
            if type_filter:
                df = df[(df["type_1"].isin(type_filter)) | (df["type_2"].isin(type_filter))]
            if search_name:
                df = df[df["name"].str.contains(search_name.lower(), case=False)]
            return df.sort_values("name")

        filtered_df = get_filtered_pokemon(type_filter, search_name)
        pokemon_choices = ["--- Choisir ---"] + sorted(filtered_df["name"].str.capitalize().tolist())

        # --- Couleurs pour les types et stats ---
        type_colors = {
            "normal": "#A8A77A", "fire": "#EE8130", "water": "#6390F0", "electric": "#F7D02C",
            "grass": "#7AC74C", "ice": "#96D9D6", "fighting": "#C22E28", "poison": "#A33EA1",
            "ground": "#E2BF65", "flying": "#A98FF3", "psychic": "#F95587", "bug": "#A6B91A",
            "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC", "dark": "#705746",
            "steel": "#B7B7CE", "fairy": "#D685AD"
        }
        stat_colors = {
            'hp': '#FF5959', 'attack': '#F5AC78', 'defense': '#FAE078',
            'special-attack': '#9DB7F5', 'special-defense': '#A7DB8D', 'speed': '#FA92B2'
        }
        stat_names = {
            'hp': 'PV', 'attack': 'Attaque', 'defense': 'Défense',
            'special-attack': 'Atq. Spé.', 'special-defense': 'Déf. Spé.', 'speed': 'Vitesse'
        }

        # --- Sélection des 6 Pokémon ---
        st.divider()
        st.subheader("🎮 Composition de l'équipe")
        team_slots = st.columns(3)

        for i in range(6):
            with team_slots[i % 3]:
                st.subheader(f"Slot {i+1}")

                # --- Sélecteur de Pokémon ---
                selected_pokemon = st.selectbox(
                    "Pokémon:",
                    options=pokemon_choices,
                    key=f"team_pokemon_{i}"
                )

                # --- Si un Pokémon est déjà dans l'équipe, on le réaffiche ---
                current_slot = st.session_state.team[i]
                if current_slot and selected_pokemon == "--- Choisir ---":
                    selected_pokemon = current_slot["pokemon_api"]["name"].capitalize()

                if selected_pokemon != "--- Choisir ---":
                    try:
                        # --- Récupération des données depuis le CSV ---
                        pokemon_csv = filtered_df[filtered_df["name"].str.lower() == selected_pokemon.lower()]

                        # Vérification que le Pokémon existe dans le DataFrame filtré
                        if pokemon_csv.empty:
                            # Si non trouvé dans le filtre actuel, on prend depuis le CSV complet
                            pokemon_csv = df_pokemon[df_pokemon["name"].str.lower() == selected_pokemon.lower()]

                        if not pokemon_csv.empty:
                            pokemon_csv = pokemon_csv.iloc[0]
                            pokemon_api = get_pokemon_data(pokemon_csv["pokemon_id"])

                            # --- Affichage du sprite ---
                            st.image(pokemon_api["sprite"], width=120)

                            # --- Infos de base ---
                            st.caption(f"ID: {pokemon_api['id']}")

                            # --- Types (badges colorés) ---
                            types_container = st.container()
                            with types_container:
                                cols = st.columns(len(pokemon_api["types"]))
                                for idx, type_name in enumerate(pokemon_api["types"]):
                                    cols[idx].markdown(
                                        f'<span style="background-color:{type_colors.get(type_name, "#68A090")}; '
                                        f'padding:5px 10px; border-radius:10px; color:white;">'
                                        f'{type_name.capitalize()}</span>',
                                        unsafe_allow_html=True
                                    )

                            # --- Statistiques (barres de progression) ---
                            st.write("**📊 Statistiques:**")
                            for stat_name, stat_value in pokemon_api["stats"].items():
                                display_name = stat_names.get(stat_name, stat_name.capitalize())
                                color = stat_colors.get(stat_name, "#4CAF50")
                                normalized_value = min(stat_value, 255)
                                progress_percent = (normalized_value / 255) * 100

                                st.progress(
                                    value=int(progress_percent),
                                    text=f"{display_name}: {stat_value}"
                                )

                            # --- Talents (sélectionnable) ---
                            st.write("**⚔️ Talents:**")
                            abilities = pokemon_api["abilities"]
                            if current_slot and "ability" in current_slot and current_slot["ability"] in abilities:
                                default_index = abilities.index(current_slot["ability"])
                            else:
                                default_index = 0

                            selected_ability = st.radio(
                                "Choisir un talent:",
                                options=abilities,
                                key=f"ability_select_{i}",
                                index=default_index
                            )

                            if selected_ability:
                                with st.expander(f"📜 Description de {selected_ability}"):
                                    try:
                                        desc = get_ability_data(selected_ability)
                                        st.write(desc)
                                    except Exception as e:
                                        st.error(f"Erreur: {e}")

                            # --- Attaques (depuis le CSV) ---
                            st.write("**⚔️ Attaques (Top 4):**")
                            moves = []
                            if "moves" in pokemon_csv and pokemon_csv["moves"]:
                                moves = [clean_move_name(move) for move in eval_list(pokemon_csv["moves"])]
                            moves = sorted(list(set(moves)))  # Suppression des doublons

                            selected_moves = []
                            if current_slot and "moves" in current_slot:
                                selected_moves = current_slot["moves"].copy()

                            # On complète avec des attaques vides si nécessaire
                            while len(selected_moves) < 4:
                                selected_moves.append("--- Aucune ---")

                            for j in range(4):
                                move_key = f"move_{i}_{j}"
                                if j < len(selected_moves) and selected_moves[j] != "--- Aucune ---":
                                    default_index = moves.index(selected_moves[j]) + 1 if selected_moves[j] in moves else 0
                                else:
                                    default_index = 0

                                move = st.selectbox(
                                    f"Attaque {j+1}:",
                                    options=["--- Aucune ---"] + moves,
                                    key=move_key,
                                    index=default_index
                                )
                                if move != "--- Aucune ---":
                                    if j < len(selected_moves):
                                        selected_moves[j] = move
                                    else:
                                        selected_moves.append(move)

                            # Nettoyage des attaques vides en trop
                            selected_moves = [m for m in selected_moves if m != "--- Aucune ---"]

                            if selected_moves:
                                st.write(f"**Attaques choisies:** {', '.join(selected_moves)}")

                            # --- Sélecteur d'item ---
                            item_choices = ["--- Aucun ---"] + sorted(df_items["name"].str.capitalize().tolist())
                            if current_slot and current_slot["item"]:
                                default_item = current_slot["item"]["name"].capitalize()
                                default_index = item_choices.index(default_item) if default_item in item_choices else 0
                            else:
                                default_index = 0

                            selected_item = st.selectbox(
                                "Item:",
                                options=item_choices,
                                key=f"team_item_{i}",
                                index=default_index
                            )

                            item_data = None
                            if selected_item != "--- Aucun ---":
                                try:
                                    item_data = df_items[df_items["name"].str.lower() == selected_item.lower()].iloc[0]
                                    st.write(f"**📦 Item:** {item_data['name'].capitalize()}")
                                    if pd.notna(item_data["effect"]):
                                        with st.expander("Voir l'effet"):
                                            st.write(item_data["effect"])
                                except IndexError:
                                    st.warning("Item non trouvé dans la base de données.")

                            # --- Mise à jour de l'équipe dans session_state ---
                            st.session_state.team[i] = {
                                "pokemon": pokemon_csv.to_dict(),
                                "pokemon_api": pokemon_api,
                                "item": item_data.to_dict() if item_data else None,
                                "types": pokemon_api["types"],
                                "moves": selected_moves,
                                "ability": selected_ability
                            }

                    except IndexError:
                        st.error(f"Pokémon '{selected_pokemon}' non trouvé dans la base de données.")
                    except Exception as e:
                        st.error(f"Erreur: {e}")

        # --- Résumé de l'équipe ---
        if any(st.session_state.team):
            st.divider()
            st.subheader("📋 Résumé de l'équipe")

            summary_cols = st.columns(3)
            for i, slot in enumerate(st.session_state.team):
                if slot:
                    with summary_cols[i % 3]:
                        pokemon = slot["pokemon_api"]
                        st.image(pokemon["sprite"], width=80)
                        st.write(f"**{pokemon['name'].capitalize()}**")
                        st.write(f"ID: {pokemon['id']}")

                        # Types
                        cols = st.columns(len(pokemon["types"]))
                        for idx, type_name in enumerate(pokemon["types"]):
                            cols[idx].markdown(
                                f'<span style="background-color:{type_colors.get(type_name, "#68A090")}; '
                                f'padding:3px 8px; border-radius:8px; color:white; font-size:12px;">'
                                f'{type_name.capitalize()}</span>',
                                unsafe_allow_html=True
                            )

                        st.write(f"Stats: {sum(pokemon['stats'].values())}")
                        st.write(f"Talents: {slot.get('ability', 'Aucun')}")
                        st.write(f"Attaques: {len(slot.get('moves', []))}/4")
                        st.write(f"Item: {slot['item']['name'].capitalize() if slot['item'] else 'Aucun'}")

                        if st.button(f"Supprimer", key=f"delete_{i}"):
                            st.session_state.team[i] = None
                            st.rerun()

            # --- Export de l'équipe ---
            st.divider()
            if st.button("📤 Exporter l'équipe"):
                export_data = {
                    "team": [
                        {
                            "name": slot["pokemon_api"]["name"],
                            "id": slot["pokemon_api"]["id"],
                            "types": slot["pokemon_api"]["types"],
                            "stats": slot["pokemon_api"]["stats"],
                            "ability": slot.get("ability"),
                            "moves": slot.get("moves", []),
                            "item": slot["item"]["name"] if slot["item"] else None,
                            "sprite": slot["pokemon_api"]["sprite"]
                        }
                        for slot in st.session_state.team if slot
                    ]
                }
                st.download_button(
                    label="Télécharger le JSON",
                    data=str(export_data),
                    file_name="pokemon_team.json",
                    mime="application/json"
                )

with tab4:
    st.header("⚔️ Comparateur de Pokémon (2 vs 2)")

    if df_pokemon is None:
        st.info("Génère d'abord le CSV pour comparer.")
    else:
        # --- Définition des relations de type (corrigée) ---
        type_chart = {
            "normal": {"strong_against": [], "weak_against": ["fighting"], "resistant_to": [], "vulnerable_to": ["ghost"]},
            "fire": {"strong_against": ["grass", "ice", "bug", "steel"], "weak_against": ["water", "ground", "rock"], "resistant_to": ["fire", "grass", "ice", "bug", "steel", "fairy"], "vulnerable_to": []},
            "water": {"strong_against": ["fire", "ground", "rock"], "weak_against": ["electric", "grass"], "resistant_to": ["fire", "water", "ice", "steel"], "vulnerable_to": []},
            "electric": {"strong_against": ["water", "flying"], "weak_against": ["ground"], "resistant_to": ["electric", "flying", "steel"], "vulnerable_to": []},
            "grass": {"strong_against": ["water", "ground", "rock"], "weak_against": ["fire", "ice", "poison", "flying", "bug"], "resistant_to": ["water", "electric", "grass", "ground"], "vulnerable_to": []},
            "ice": {"strong_against": ["grass", "ground", "flying", "dragon"], "weak_against": ["fire", "fighting", "rock", "steel"], "resistant_to": ["ice"], "vulnerable_to": []},
            "fighting": {"strong_against": ["normal", "ice", "rock", "dark", "steel"], "weak_against": ["flying", "psychic", "fairy"], "resistant_to": ["bug", "rock", "dark"], "vulnerable_to": []},
            "poison": {"strong_against": ["grass", "fairy"], "weak_against": ["ground", "psychic"], "resistant_to": ["grass", "fighting", "poison", "bug", "fairy"], "vulnerable_to": []},
            "ground": {"strong_against": ["fire", "electric", "poison", "rock", "steel"], "weak_against": ["water", "grass", "ice"], "resistant_to": ["poison", "rock"], "vulnerable_to": ["electric"]},
            "flying": {"strong_against": ["grass", "fighting", "bug"], "weak_against": ["electric", "ice", "rock"], "resistant_to": ["grass", "fighting", "bug"], "vulnerable_to": ["ground"]},
            "psychic": {"strong_against": ["fighting", "poison"], "weak_against": ["bug", "ghost", "dark"], "resistant_to": ["fighting", "psychic"], "vulnerable_to": []},
            "bug": {"strong_against": ["grass", "psychic", "dark"], "weak_against": ["fire", "flying", "rock"], "resistant_to": ["grass", "fighting", "ground"], "vulnerable_to": []},
            "rock": {"strong_against": ["fire", "ice", "flying", "bug"], "weak_against": ["water", "grass", "fighting", "ground", "steel"], "resistant_to": ["normal", "fire", "poison", "flying"], "vulnerable_to": []},
            "ghost": {"strong_against": ["psychic", "ghost"], "weak_against": ["ghost", "dark"], "resistant_to": ["poison", "bug"], "vulnerable_to": ["normal", "fighting"]},
            "dragon": {"strong_against": ["dragon"], "weak_against": ["ice", "dragon", "fairy"], "resistant_to": ["fire", "water", "electric", "grass"], "vulnerable_to": []},
            "dark": {"strong_against": ["psychic", "ghost"], "weak_against": ["fighting", "bug", "fairy"], "resistant_to": ["ghost", "dark"], "vulnerable_to": ["psychic"]},
            "steel": {"strong_against": ["ice", "rock", "fairy"], "weak_against": ["fire", "fighting", "ground"], "resistant_to": ["normal", "grass", "ice", "flying", "psychic", "bug", "rock", "dragon", "steel", "fairy"], "vulnerable_to": ["poison"]},
            "fairy": {"strong_against": ["fighting", "dark", "dragon"], "weak_against": ["poison", "steel"], "resistant_to": ["fighting", "bug", "dragon", "dark"], "vulnerable_to": []}
        }

        # --- Fonction corrigée pour calculer les avantages de type ---
        def calculate_type_matchup(attacker_types, defender_types):
            effectiveness = 1.0  # Neutre par défaut

            for at in attacker_types:
                for dt in defender_types:
                    if at in type_chart and dt in type_chart[at]["strong_against"]:
                        effectiveness *= 2.0  # Super efficace
                    elif at in type_chart and dt in type_chart[at]["weak_against"]:
                        effectiveness *= 0.5  # Peu efficace
                    elif at in type_chart and dt in type_chart[at]["vulnerable_to"]:
                        effectiveness *= 0.0  # Inefficace

            return effectiveness

        # --- Sélection des Pokémon ---
        names = sorted(df_pokemon["name"].str.capitalize().tolist())
        colA, colB = st.columns(2)

        with colA:
            a = st.selectbox("Pokémon A", ["---"] + names, key="comp_a")
            if a != "---":
                try:
                    ra = df_pokemon[df_pokemon["name"].str.lower() == a.lower()].iloc[0]
                    pokemon_a = get_pokemon_data(ra["pokemon_id"])
                    st.image(pokemon_a["sprite"], width=150)
                    st.write(f"**Types:** {', '.join(pokemon_a['types'])}")
                except IndexError:
                    st.error("Pokémon non trouvé")

        with colB:
            b = st.selectbox("Pokémon B", ["---"] + names, key="comp_b")
            if b != "---":
                try:
                    rb = df_pokemon[df_pokemon["name"].str.lower() == b.lower()].iloc[0]
                    pokemon_b = get_pokemon_data(rb["pokemon_id"])
                    st.image(pokemon_b["sprite"], width=150)
                    st.write(f"**Types:** {', '.join(pokemon_b['types'])}")
                except IndexError:
                    st.error("Pokémon non trouvé")

        if a != "---" and b != "---":
            # --- Récupération des données complètes ---
            pokemon_a = get_pokemon_data(df_pokemon[df_pokemon["name"].str.lower() == a.lower()].iloc[0]["pokemon_id"])
            pokemon_b = get_pokemon_data(df_pokemon[df_pokemon["name"].str.lower() == b.lower()].iloc[0]["pokemon_id"])

            # --- Comparaison des stats ---
            st.divider()
            st.subheader("📊 Comparaison des statistiques")

            # Calcul des stats totales
            stats_a = sum(pokemon_a["stats"].values())
            stats_b = sum(pokemon_b["stats"].values())

            # Affichage des stats détaillées
            stat_names = {
                'hp': 'PV', 'attack': 'Attaque', 'defense': 'Défense',
                'special-attack': 'Atq. Spé.', 'special-defense': 'Déf. Spé.', 'speed': 'Vitesse'
            }

            stat_colors = {
                'hp': '#FF5959', 'attack': '#F5AC78', 'defense': '#FAE078',
                'special-attack': '#9DB7F5', 'special-defense': '#A7DB8D', 'speed': '#FA92B2'
            }

            stat_cols = st.columns(2)
            with stat_cols[0]:
                st.write(f"**{a}** (Total: {stats_a})")
                for stat_name, stat_value in pokemon_a["stats"].items():
                    progress_value = min(100, (stat_value / 255) * 100)
                    st.progress(
                        value=int(progress_value),
                        text=f"{stat_names[stat_name]}: {stat_value}"
                    )

            with stat_cols[1]:
                st.write(f"**{b}** (Total: {stats_b})")
                for stat_name, stat_value in pokemon_b["stats"].items():
                    progress_value = min(100, (stat_value / 255) * 100)
                    st.progress(
                        value=int(progress_value),
                        text=f"{stat_names[stat_name]}: {stat_value}"
                    )

            # --- Analyse des types (NOUVELLE VERSION) ---
            st.divider()
            st.subheader("🔍 Analyse des types")

            # Récupération des types
            types_a = pokemon_a["types"]
            types_b = pokemon_b["types"]

            # Calcul des efficacités
            effectiveness_a_vs_b = calculate_type_matchup(types_a, types_b)
            effectiveness_b_vs_a = calculate_type_matchup(types_b, types_a)

            # Affichage des relations de type
            st.write(f"**Efficacité des types:**")
            st.write(f"- {a} contre {b}: {'Super efficace' if effectiveness_a_vs_b > 1 else 'Peu efficace' if effectiveness_a_vs_b < 1 else 'Normale'} (x{effectiveness_a_vs_b:.1f})")
            st.write(f"- {b} contre {a}: {'Super efficace' if effectiveness_b_vs_a > 1 else 'Peu efficace' if effectiveness_b_vs_a < 1 else 'Normale'} (x{effectiveness_b_vs_a:.1f})")

            # --- Calcul du gagnant probable ---
            st.divider()
            st.subheader("🏆 Résultat final")

            # Calcul du score basé sur les stats (60%)
            stats_score_a = (stats_a / (stats_a + stats_b)) * 60 if (stats_a + stats_b) > 0 else 30
            stats_score_b = (stats_b / (stats_a + stats_b)) * 60 if (stats_a + stats_b) > 0 else 30

            # Calcul du score basé sur les types (40%)
            # Plus l'efficacité est élevée, plus le score est bon
            type_score_a = 50 + (20 * (effectiveness_a_vs_b - 1))  # 50% de base + bonus/malus
            type_score_b = 50 + (20 * (effectiveness_b_vs_a - 1))  # 50% de base + bonus/malus

            # Normalisation pour que la somme fasse 100%
            type_score_a = max(0, min(100, type_score_a))
            type_score_b = max(0, min(100, type_score_b))

            # Score final (60% stats, 40% types)
            final_score_a = stats_score_a + (type_score_a * 0.4)
            final_score_b = stats_score_b + (type_score_b * 0.4)

            # Normalisation finale
            total_score = final_score_a + final_score_b
            if total_score > 0:
                final_score_a = (final_score_a / total_score) * 100
                final_score_b = (final_score_b / total_score) * 100

            # Affichage du résultat
            if abs(final_score_a - final_score_b) < 1:
                st.write(f"⚖️ **Match équilibré!** {a}: {final_score_a:.1f}% vs {b}: {final_score_b:.1f}%")
            elif final_score_a > final_score_b:
                st.write(f"🏆 **{a} a l'avantage!** ({final_score_a:.1f}% vs {final_score_b:.1f}%)")
            else:
                st.write(f"🏆 **{b} a l'avantage!** ({final_score_b:.1f}% vs {final_score_a:.1f}%)")

            # Détails des calculs
            with st.expander("Voir les détails des calculs"):
                st.write(f"**Score basé sur les stats (60%):**")
                st.write(f"- {a}: {stats_score_a:.1f}%")
                st.write(f"- {b}: {stats_score_b:.1f}%")

                st.write(f"**Score basé sur les types (40%):**")
                st.write(f"- {a} contre {b}: {type_score_a:.1f}% (x{effectiveness_a_vs_b:.1f})")
                st.write(f"- {b} contre {a}: {type_score_b:.1f}% (x{effectiveness_b_vs_a:.1f})")

                st.write(f"**Score final (100%):**")
                st.write(f"- {a}: {final_score_a:.1f}%")
                st.write(f"- {b}: {final_score_b:.1f}%")
