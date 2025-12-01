import streamlit as st
import pandas as pd
import requests
from PIL import Image
from collections import defaultdict
from io import BytesIO
import base64
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from coords import regions_coords

# Style CSS simplifié pour lisibilité
st.markdown("""
<style>
    .block-container { padding-top: 0rem !important; }
    h1 { text-align: center !important; margin-top: 0px !important; }
    .map-container { display: flex; justify-content: center; }
    .stPlotlyChart { max-width: 900px; width: 900px; margin: 0 auto; }
    .pokemon-list { margin-top: 20px; }
    .pokemon-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
    .pokemon-item:hover { background-color: #272727; }
</style>
""", unsafe_allow_html=True)

# Fonction pour récupérer les données depuis PokéAPI
@st.cache_data
def get_pokemon_data(pokemon_id):
    try:
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}")
        pokemon_data = response.json()

        # Talents
        abilities = [a['ability']['name'].replace('-', ' ').capitalize() for a in pokemon_data['abilities']]

        species_response = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}")
        species_data = species_response.json()
        evolution_chain_url = species_data['evolution_chain']['url']

        evolution_response = requests.get(evolution_chain_url)
        evolution_data = evolution_response.json()

        types = [t['type']['name'] for t in pokemon_data['types']]
        stats = {s['stat']['name']: s['base_stat'] for s in pokemon_data['stats']}

        def get_evolution_chain(chain):
            evolutions = []
            current = chain
            while current:
                species = current['species']
                evolutions.append({
                    'name': species['name'],
                    'id': int(species['url'].split('/')[-2])
                })
                if current['evolves_to']:
                    current = current['evolves_to'][0]
                else:
                    current = None
            return evolutions

        evolution_chain = get_evolution_chain(evolution_data['chain'])

        return {
            'name': pokemon_data['name'],
            'id': pokemon_data['id'],
            'types': types,
            'stats': stats,
            'sprite': pokemon_data['sprites']['front_default'],
            'evolution_chain': evolution_chain,
            'abilities': abilities
        }
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return None

@st.cache_data
def get_ability_data(ability_name):
    try:
        # Nettoyage très robuste
        api_name = (
            ability_name.lower()
                        .replace("é", "e")     # gestion accents éventuels
                        .replace(" ", "-")     # espaces → tirets
                        .replace("_", "-")     # underscores → tirets
                        .replace("–", "-")     # tirets spéciaux
                        .strip()
        )

        # Cas des talents avec parenthèses (ex : "Sand Veil (Hidden)")
        if "(" in api_name:
            api_name = api_name.split("(")[0].strip()

        url = f"https://pokeapi.co/api/v2/ability/{api_name}"
        response = requests.get(url)

        if response.status_code != 200:
            return f"Impossible de récupérer le talent : {ability_name} (URL : {url})"

        data = response.json()

        # Chercher d'abord en français
        for e in data["effect_entries"]:
            if e["language"]["name"] == "fr":
                return e["effect"]

        # Sinon anglais
        for e in data["effect_entries"]:
            if e["language"]["name"] == "en":
                return e["effect"]

        return "Aucune description disponible."

    except Exception as e:
        return f"Erreur lors de la récupération du talent : {e}"

# Chargement des images et données
@st.cache_data
def load_image(path):
    img = Image.open(path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return img, encoded

@st.cache_data
def load_data():
    return pd.read_csv("pokemon_location_encounters_full.csv")

st.set_page_config(page_title="Pokédex Avancé", layout="wide")

st.title("📘 Pokédex Multifeatures")

# --- Création des onglets ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Localisation par Génération",
    "📊 Statistiques Globales",
    "🧩 Team Builder",
    "⚔️ Comparateur de Pokémon"
])

with tab1:
    st.header("📍 Localisation des Pokémon par Génération")
    # Sélection de la région
    regions = list(regions_coords.keys())
    region = st.selectbox("Choisir une région", regions)

    # Barre de recherche globale
    search_term_global = st.text_input("Filtrer les lieux par Pokémon", value="", key="search_global")

    # Carte et données
    col1, col2 = st.columns([1.5, 2])

    with col1:
        try:
            img, encoded_img = load_image(f"cartes/carte_{region.lower()}.png")
            img_w, img_h = img.size
        except FileNotFoundError:
            st.error(f"Carte introuvable pour {region}")
            st.stop()

        df = load_data()
        df_region = df[df["region"] == region]

        # Pokémon par location
        pokemon_by_location = defaultdict(list)
        for _, row in df_region.iterrows():
            pokemon_by_location[row["location"]].append(row.to_dict())

        # Coordonnées des points (filtrés si recherche globale)
        x_points, y_points, locations, hover_texts = [], [], [], []
        for lieu, pokes in pokemon_by_location.items():
            if lieu in regions_coords[region]:
                unique_pokes = {p["name"]: p for p in pokes}.values()
                if search_term_global:
                    if not any(search_term_global.lower() in p["name"].lower() for p in unique_pokes):
                        continue
                x, y = regions_coords[region][lieu]
                x_points.append(x)
                y_points.append(y)
                locations.append(lieu)
                hover_texts.append(f"<b>{lieu}</b><br>{len(unique_pokes)} Pokémon<br>Cliquez pour voir la liste")

        FIXED_WIDTH = 900
        FIXED_HEIGHT = int(FIXED_WIDTH * img_h / img_w)

        if "clicked_location" not in st.session_state:
            st.session_state.clicked_location = None
        if "selected_pokemon" not in st.session_state:
            st.session_state.selected_pokemon = None
        if "selected_ability" not in st.session_state:
            st.session_state.selected_ability = None

        # Carte Plotly
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_img}",
                xref="x", yref="y",
                x=0, y=FIXED_HEIGHT,
                sizex=FIXED_WIDTH,
                sizey=FIXED_HEIGHT,
                sizing="stretch",
                layer="below"
            )
        )

        fig.add_trace(go.Scatter(
            x=x_points,
            y=[FIXED_HEIGHT - y for y in y_points],
            mode="markers",
            marker=dict(size=14, color="rgba(255, 215, 0, 0.8)", line=dict(width=1, color="rgba(200, 180, 0, 0.7)")),
            customdata=locations,
            hoverinfo="text",
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>"
        ))

        fig.update_xaxes(visible=False, range=[0, FIXED_WIDTH], fixedrange=True)
        fig.update_yaxes(visible=False, range=[0, FIXED_HEIGHT], fixedrange=True)
        fig.update_layout(width=FIXED_WIDTH, height=FIXED_HEIGHT, margin=dict(l=0,r=0,t=0,b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", hovermode="closest")

        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        selected_point = plotly_events(fig, click_event=True, override_width=FIXED_WIDTH, override_height=FIXED_HEIGHT)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if selected_point:
            point = selected_point[0]
            idx = point["pointIndex"]

            # Si on change de lieu → réinitialiser le Pokémon
            if st.session_state.clicked_location != locations[idx]:
                st.session_state.selected_pokemon = None

            st.session_state.clicked_location = locations[idx]


        if st.session_state.clicked_location:
            lieu = st.session_state.clicked_location
            pokemons = pokemon_by_location[lieu]
            unique_pokemons = {p["name"]: p for p in pokemons}.values()
            filtered_pokemons = [
                p for p in unique_pokemons
                if not search_term_global or search_term_global.lower() in p["name"].lower()
            ]

            st.markdown(f"<div class='pokemon-list'><h3>Pokémon à {lieu} ({len(unique_pokemons)} au total, {len(filtered_pokemons)} filtrés)</h3></div>", unsafe_allow_html=True)

            if filtered_pokemons:
                for poke in filtered_pokemons:
                    col1_, col2_ = st.columns([4, 1])
                    with col1_:
                        st.markdown(f"<div class='pokemon-item'><span><b>{poke['name']}</b> (ID: {poke['pokemon_id']})</span></div>", unsafe_allow_html=True)
                    with col2_:
                        if st.button(f"Plus d'infos", key=f"info_{poke['name']}_{lieu}"):
                            st.session_state.selected_pokemon = poke
                            st.session_state.selected_ability = None

            else:
                st.warning("Aucun Pokémon ne correspond à votre recherche.")

    # Détails Pokémon
    if st.session_state.selected_pokemon:
        poke = st.session_state.selected_pokemon
        pokemon_data = get_pokemon_data(poke['pokemon_id'])
        if pokemon_data:
            st.markdown(f"### {pokemon_data['name'].capitalize()} (ID: {pokemon_data['id']})")
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col1:
                st.image(pokemon_data['sprite'], width=300)
                # Évolution
                st.markdown("#### Évolution")
                if pokemon_data['evolution_chain']:
                    evo_html = '<div style="display:flex; gap:15px; flex-wrap:wrap; justify-content:center">'
                    for i, evo in enumerate(pokemon_data['evolution_chain']):
                        if i>0: evo_html += '<div style="font-size:18px;color:#666;">→</div>'
                        evo_html += f'<div style="text-align:center;"><img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{evo["id"]}.png" style="width:60px;height:60px;"><div style="font-size:12px">{evo["name"].capitalize()}</div></div>'
                    evo_html += '</div>'
                    st.markdown(evo_html, unsafe_allow_html=True)
                else:
                    st.markdown("Pas d'évolution connue")
            with col2:
                # Types
                st.markdown("#### Types")
                type_colors = {
                    'normal': '#A8A77A','fire': '#EE8130','water': '#6390F0','electric': '#F7D02C',
                    'grass': '#7AC74C','ice': '#96D9D6','fighting': '#C22E28','poison': '#A33EA1',
                    'ground': '#E2BF65','flying': '#A98FF3','psychic': '#F95587','bug': '#A6B91A',
                    'rock': '#B6A136','ghost': '#735797','dragon': '#6F35FC','dark': '#705746',
                    'steel': '#B7B7CE','fairy': '#D685AD'
                }
                type_badges = "".join([f'<span style="background-color:{type_colors.get(t,"#68A090")}; padding:5px 10px; border-radius:12px; color:white; margin-right:8px;">{t.capitalize()}</span>' for t in pokemon_data['types']])
                st.markdown(f"<div style='margin-bottom:15px'>{type_badges}</div>", unsafe_allow_html=True)
            
                # Stats
                st.markdown("#### Statistiques")

                stat_names = {
                    'hp': 'PV', 'attack': 'Attaque', 'defense': 'Défense',
                    'special-attack': 'Attaque Spé.', 'special-defense': 'Défense Spé.',
                    'speed': 'Vitesse'
                }

                stat_colors = {
                    'hp': '#FF5959', 'attack': '#F5AC78', 'defense': '#FAE078',
                    'special-attack': '#9DB7F5', 'special-defense': '#A7DB8D',
                    'speed': '#FA92B2'
                }

                for stat_name, stat_value in pokemon_data['stats'].items():
                    display_name = stat_names.get(stat_name, stat_name.capitalize())
                    color = stat_colors.get(stat_name, '#4CAF50')

                    # Capage entre 0 et 200 puis conversion en %
                    capped_value = max(0, min(stat_value, 200))
                    percent = (capped_value / 200) * 100

                    st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-weight:bold; margin-bottom:4px;">
                        <span>{display_name}</span>
                        <span>{stat_value}</span>
                    </div>
                    <div style="height:8px; background:#e0e0e0; border-radius:4px; overflow:hidden;">
                        <div style="height:8px; background:{color}; width:{percent}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                # Talents
                st.markdown("#### Talents")

                abilities = pokemon_data["abilities"]

                # Crée autant de colonnes que de talents
                cols = st.columns(len(abilities))

                for i, ability in enumerate(abilities):
                    # Bouton dans la colonne correspondante
                    if cols[i].button(ability, key=f"ability_{ability}_{poke['pokemon_id']}"):
                        st.session_state.selected_ability = ability

                # Affichage de la description du talent sélectionné
                if st.session_state.selected_ability:
                    ability_desc = get_ability_data(st.session_state.selected_ability)
                    st.markdown(f"**{st.session_state.selected_ability} :** {ability_desc}")
with tab2:
    st.header("📊 Statistiques Globales sur les Pokémon (via PokéAPI)")

    # -------------------------------------------------------------------
    # 1 - Chargement global via API (effectué UNE SEULE FOIS grâce au cache)
    # -------------------------------------------------------------------
    @st.cache_data(show_spinner=True)
    def load_all_pokemon_ids():
        url = "https://pokeapi.co/api/v2/pokemon?limit=800"
        r = requests.get(url)
        data = r.json()
        return [p["url"].split("/")[-2] for p in data["results"]]


    @st.cache_data(show_spinner=True)
    def get_region_from_species(species_id):
        url = f"https://pokeapi.co/api/v2/pokemon-species/{species_id}"
        r = requests.get(url)
        if r.status_code != 200:
            return None
        data = r.json()

        gen_url = data["generation"]["url"]
        gen_data = requests.get(gen_url).json()

        return gen_data["main_region"]["name"]


    @st.cache_data(show_spinner=True)
    def build_pokemon_df():
        ids = load_all_pokemon_ids()
        rows = []

        for pid in ids:
            data = get_pokemon_data(pid)
            if not data:
                continue

            rows.append({
                "pokemon_id": pid,
                "name": data["name"],
                "type_1": data["types"][0] if len(data["types"]) > 0 else None,
                "type_2": data["types"][1] if len(data["types"]) > 1 else None,
                "region": get_region_from_species(pid),
                "total_stats": sum(data["stats"].values())
            })

        return pd.DataFrame(rows)


    df = build_pokemon_df()
    st.success("✔️ Données chargées depuis PokéAPI")


    # -------------------------------------------------------------------
    # 2 - Sélecteurs (simples, légers)
    # -------------------------------------------------------------------
    region_choice = st.selectbox(
        "Filtrer par région d’origine :",
        ["Toutes"] + sorted(df["region"].dropna().unique().tolist())
    )

    type_choice = st.multiselect(
        "Filtrer par type",
        ["normal","fire","water","grass","electric","ice","fighting","poison","ground","flying",
        "psychic","bug","rock","ghost","dragon","dark","steel","fairy"]
    )


    # Petite fonction utilitaire pour filtrer rapidement
    def apply_filters(dataframe):
        df_loc = dataframe

        # Filtre région
        if region_choice != "Toutes":
            df_loc = df_loc[df_loc["region"] == region_choice]

        # Filtre types
        if type_choice:
            df_loc = df_loc[
                df_loc["type_1"].isin(type_choice) |
                df_loc["type_2"].isin(type_choice)
            ]

        return df_loc


    # -------------------------------------------------------------------
    # 🔵 GRAPHIQUE 1 : Nombre de Pokémon par région (ne dépend QUE de la région)
    # -------------------------------------------------------------------
    st.subheader("📌 Nombre de Pokémon par région")

    # Filtre uniquement par types (important !)
    df_plot1 = df.copy()
    if type_choice:
        df_plot1 = df_plot1[
            df_plot1["type_1"].isin(type_choice) |
            df_plot1["type_2"].isin(type_choice)
        ]

    region_counts = df_plot1["region"].value_counts().sort_values(ascending=True)

    fig1 = go.Figure()
    fig1.add_bar(
        x=region_counts.values,
        y=[r.capitalize() for r in region_counts.index],
        orientation="h"
    )
    fig1.update_layout(
        title="Nombre total de Pokémon par région",
        xaxis_title="Pokémon",
        yaxis_title="Régions"
    )
    st.plotly_chart(fig1, use_container_width=True)


    # -------------------------------------------------------------------
    # 🔵 GRAPHIQUE 2 : Répartition des Pokémon par type (filtre local complet)
    # -------------------------------------------------------------------
    st.subheader("📌 Répartition des Pokémon par type")

    df_plot2 = apply_filters(df)

    type_counts = pd.Series(dtype=int)
    for col in ["type_1", "type_2"]:
        type_counts = type_counts.add(df_plot2[col].value_counts(), fill_value=0)

    type_counts = type_counts.sort_values(ascending=False)

    fig2 = go.Figure()
    fig2.add_bar(
        x=type_counts.index.str.capitalize(),
        y=type_counts.values
    )
    fig2.update_layout(
        title=f"Répartition des Pokémon par type ({'Toutes régions' if region_choice=='Toutes' else region_choice})",
        xaxis_title="Types",
        yaxis_title="Nombre de Pokémon"
    )
    st.plotly_chart(fig2, use_container_width=True)


    # -------------------------------------------------------------------
    # 🔵 GRAPHIQUE 3 : Histogramme des stats (filtre local complet)
    # -------------------------------------------------------------------
    st.subheader("📌 Distribution du total des statistiques de base")

    df_plot3 = apply_filters(df)

    fig3 = go.Figure()
    fig3.add_histogram(x=df_plot3["total_stats"], nbinsx=30)
    fig3.update_layout(
        title="Distribution des totaux de statistiques",
        xaxis_title="Total des statistiques",
        yaxis_title="Nombre de Pokémon"
    )
    st.plotly_chart(fig3, use_container_width=True)


    st.success("📈 Graphiques mis à jour efficacement !")

# ============================================================
# 3️⃣ ONGLET : TEAM BUILDER
# ============================================================
with tab3:
    st.header("🧩 Générateur de Team Pokémon (6 Slots)")

    st.write("Choisis jusqu’à 6 Pokémon pour créer ta team.")

    cols = st.columns(3)

    team = [None]*6

    for i in range(6):
        with cols[i % 3]:
            team[i] = st.selectbox(
                f"Slot {i+1}",
                ["--- Choisir ---"] + ["Pikachu", "Dracaufeu", "Tyranocif", "Lucario", "Mewtwo"],  # remplace par ta BDD
                key=f"slot_{i}"
            )

            if team[i] != "--- Choisir ---":
                st.write(f"⭐ **{team[i]} sélectionné**")
                st.write("Talents : à compléter")
                st.write("Attaques : à compléter")
                st.write("Objet : à ajouter")

    st.success("Ta team est prête ! (Tu pourras ajouter import/export, synergies…)")


# ============================================================
# 4️⃣ ONGLET : Comparateur de Pokémon
# ============================================================
with tab4:
    st.header("⚔️ Comparateur de Pokémon")

    colA, colB = st.columns(2)

    with colA:
        pokeA = st.selectbox("Pokémon 1 :", ["Pikachu", "Salamèche", "Dracaufeu"], key="p1")
    with colB:
        pokeB = st.selectbox("Pokémon 2 :", ["Pikachu", "Salamèche", "Dracaufeu"], key="p2")

    st.subheader("📊 Comparaison Statistique")
    st.write("→ Statistiques (PV, Attaque, Défense...)")
    st.write("→ Types, résistance, immunités, faiblesses")
    st.write("→ Taille, poids, talents…")

    st.info("👉 Tu rempliras avec tes fonctions de récupération de stats + diagramme radar.")


# Données brutes
with st.expander("Données brutes"):
    st.dataframe(df_region)
