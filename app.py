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

# Configuration de base
st.set_page_config(layout="wide")


# Style CSS complet
st.markdown("""
<style>
    .block-container { padding-top: 0rem !important; }
    h1 { text-align: center !important; margin-top: 0px !important; }
    .map-container { display: flex; justify-content: center; }
    .stPlotlyChart { max-width: 900px; width: 900px; margin: 0 auto; }
    .pokemon-list { margin-top: 20px; }
    .pokemon-item {
        padding: 10px;
        border-bottom: 1px solid #eee;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .pokemon-item:hover { background-color: #272727; }
    .pokemon-details {
        margin-top: 20px;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #272727;
        position: relative;
    }
    .close-btn {
        position: absolute;
        top: 10px;
        right: 15px;
        background: none;
        border: none;
        color: #666;
        cursor: pointer;
        font-size: 20px;
        z-index: 10;
    }
    .types-container {
        display: flex;
        gap: 8px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    .type-badge {
        padding: 4px 10px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 12px;
        text-align: center;
        min-width: 60px;
    }
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 10px;
        margin: 15px 0;
    }
    .stat-item {
        background-color: #272727;
        padding: 10px;
        border-radius: 5px;
    }
    .stat-bar-container {
        height: 10px;
        background-color: #272727;
        border-radius: 5px;
        margin-top: 5px;
        overflow: hidden;
    }
    .stat-bar {
        height: 10px;
        border-radius: 5px;
    }
    .evolution-chain {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-top: 15px;
        flex-wrap: wrap;
        justify-content: center;
    }
    .evolution-step {
        text-align: center;
        min-width: 120px;
    }
    .evolution-arrow {
        font-size: 18px;
        color: #666;
    }
    .pokemon-sprite {
        width: 120px;
        height: 120px;
        image-rendering: pixelated;
    }
    .evolution-sprite {
        width: 60px;
        height: 60px;
        image-rendering: pixelated;
    }
</style>
""", unsafe_allow_html=True)


# Fonction pour récupérer les données depuis PokéAPI
@st.cache_data
def get_pokemon_data(pokemon_id):
    try:
        # Récupération des données de base
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}")
        pokemon_data = response.json()

        # Récupération des données d'évolution
        species_response = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}")
        species_data = species_response.json()
        evolution_chain_url = species_data['evolution_chain']['url']

        evolution_response = requests.get(evolution_chain_url)
        evolution_data = evolution_response.json()

        # Traitement des types
        types = [t['type']['name'] for t in pokemon_data['types']]

        # Traitement des statistiques
        stats = {s['stat']['name']: s['base_stat'] for s in pokemon_data['stats']}

        # Traitement de la chaîne d'évolution
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
            'evolution_chain': evolution_chain
        }
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        return None

# Chargement des données
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

st.title("Cartes des régions Pokémon")
# Sélection de la région
regions = list(regions_coords.keys())
region = st.selectbox("Choisir une région", regions)

# Chargement de la carte et des données
col1, col2 = st.columns([1.5, 2]) 

with col1:
  # TA CARTE
    try:
        img, encoded_img = load_image(f"cartes/carte_{region.lower()}.png")
        img_w, img_h = img.size
    except FileNotFoundError:
        st.error(f"Carte introuvable pour {region}")
        st.stop()
    df = load_data()
    df_region = df[df["region"] == region]

    # Préparation des données
    pokemon_by_location = defaultdict(list)
    for _, row in df_region.iterrows():
        pokemon_by_location[row["location"]].append(row.to_dict())

    # Coordonnées des points
    x_points = []
    y_points = []
    locations = []
    hover_texts = []
    for lieu, pokes in pokemon_by_location.items():
        if lieu in regions_coords[region]:
            x, y = regions_coords[region][lieu]
            x_points.append(x)
            y_points.append(y)
            locations.append(lieu)
            hover_texts.append(f"<b>{lieu}</b><br>{len(pokes)} Pokémon<br>Cliquez pour voir la liste")

    # Dimensions fixes
    FIXED_WIDTH = 900
    FIXED_HEIGHT = int(FIXED_WIDTH * img_h / img_w)

    # Initialisation des variables de session
    if "clicked_location" not in st.session_state:
        st.session_state.clicked_location = None
    if "search_term" not in st.session_state:
        st.session_state.search_term = ""
    if "selected_pokemon" not in st.session_state:
        st.session_state.selected_pokemon = None

    # Création de la figure Plotly
    fig = go.Figure()

    # Ajout de l'image de fond
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

    # Ajout des points avec hover informatif
    fig.add_trace(
        go.Scatter(
            x=x_points,
            y=[FIXED_HEIGHT - y for y in y_points],
            mode="markers",
            marker=dict(
                size=14,
                color="rgba(255, 215, 0, 0.8)",
                line=dict(width=1, color="rgba(200, 180, 0, 0.7)")
            ),
            customdata=locations,
            hoverinfo="text",
            hovertext=hover_texts,
            hovertemplate="%{hovertext}<extra></extra>"
        )
    )

    # Configuration des axes
    fig.update_xaxes(visible=False, range=[0, FIXED_WIDTH], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, FIXED_HEIGHT], fixedrange=True)

    fig.update_layout(
        width=FIXED_WIDTH,
        height=FIXED_HEIGHT,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest"
    )

    # Affichage de la carte
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    selected_point = plotly_events(fig, click_event=True, override_width=FIXED_WIDTH, override_height=FIXED_HEIGHT)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    # Gestion des clics
    if selected_point:
        point = selected_point[0]
        idx = point["pointIndex"]
        st.session_state.clicked_location = locations[idx]
        st.session_state.selected_pokemon = None

    # Champ de recherche (persistant)
    st.session_state.search_term = st.text_input(
        "Rechercher un Pokémon",
        value=st.session_state.search_term,
        key="search_input"
    )

    # Affichage de la liste des Pokémon si un lieu est cliqué
    if st.session_state.clicked_location:
        lieu = st.session_state.clicked_location
        pokemons = pokemon_by_location[lieu]

        # Filtrer les Pokémon selon le terme de recherche
        filtered_pokemons = [
            poke for poke in pokemons
            if st.session_state.search_term.lower() in poke["name"].lower()
        ]

        st.markdown(f"""
        <div class="pokemon-list">
            <h3>Pokémon à {lieu} ({len(pokemons)} au total, {len(filtered_pokemons)} filtrés)</h3>
        </div>
        """, unsafe_allow_html=True)

        if filtered_pokemons:
            for poke in filtered_pokemons:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="pokemon-item">
                        <span><b>{poke['name']}</b> (ID: {poke['pokemon_id']})</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button(f"Plus d'infos", key=f"info_{poke['pokemon_id']}_{lieu}"):
                        st.session_state.selected_pokemon = poke
        else:
            st.warning("Aucun Pokémon ne correspond à votre recherche.")

# Affichage des détails du Pokémon sélectionné
if st.session_state.selected_pokemon:
    poke = st.session_state.selected_pokemon
    pokemon_data = get_pokemon_data(poke['pokemon_id'])

    if pokemon_data:
        # Utilisation de colonnes Streamlit pour une meilleure structure
        with st.container():
            # Bouton de fermeture
            if st.button("✕ Fermer", key="close_details"):
                st.session_state.selected_pokemon = None
                st.rerun()

            # Titre centré
            st.markdown(f" {pokemon_data['name'].capitalize()} (ID: {pokemon_data['id']})""")

            # Layout en colonnes
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                # Affichage du sprite
                st.image(pokemon_data['sprite'], width=300)

            with col2:
                # Section Types
                st.markdown("#### Types")
                type_colors = {
                    'normal': '#A8A77A', 'fire': '#EE8130', 'water': '#6390F0', 'electric': '#F7D02C',
                    'grass': '#7AC74C', 'ice': '#96D9D6', 'fighting': '#C22E28', 'poison': '#A33EA1',
                    'ground': '#E2BF65', 'flying': '#A98FF3', 'psychic': '#F95587', 'bug': '#A6B91A',
                    'rock': '#B6A136', 'ghost': '#735797', 'dragon': '#6F35FC', 'dark': '#705746',
                    'steel': '#B7B7CE', 'fairy': '#D685AD'
                }

                # Affichage des types avec des badges colorés
                type_badges = ""
                for type_name in pokemon_data['types']:
                    color = type_colors.get(type_name, '#68A090')
                    type_badges += f'<span style="background-color:{color}; padding:5px 10px; border-radius:12px; color:white; margin-right:8px;">{type_name.capitalize()}</span>'

                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    {type_badges}
                </div>
                """, unsafe_allow_html=True)

                # Section Statistiques
                st.markdown("#### Statistiques")
                stat_names = {
                    'hp': 'PV', 'attack': 'Attaque', 'defense': 'Défense',
                    'special-attack': 'Attaque Spé.', 'special-defense': 'Défense Spé.',
                    'speed': 'Vitesse'
                }

                stat_colors = {
                    'hp': '#FF5959', 'attack': '#F5AC78', 'defense': '#FAE078',
                    'special-attack': '#9DB7F5', 'special-defense': '#A7DB8D', 'speed': '#FA92B2'
                }

                for stat_name, stat_value in pokemon_data['stats'].items():
                    display_name = stat_names.get(stat_name, stat_name.capitalize())
                    color = stat_colors.get(stat_name, '#4CAF50')
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span><b>{display_name}:</b></span>
                            <span>{stat_value}</span>
                        </div>
                        <div style="height: 8px; background-color: #e0e0e0; border-radius: 4px;">
                            <div style="height: 8px; background-color: {color}; border-radius: 4px; width: {min(100, stat_value)}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    # Section Évolution (pleine largeur)
                    st.markdown("#### Évolution")
                    if pokemon_data['evolution_chain']:
                        evolution_html = '<div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap; margin: 15px 0;">'
                        for i, evolution in enumerate(pokemon_data['evolution_chain']):
                            if i > 0:
                                evolution_html += '<div style="font-size: 18px; color: #666;">→</div>'
                            evolution_html += f"""
                            <div style="text-align: center;">
                                <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{evolution['id']}.png" style="width: 60px; height: 60px;">
                                <div style="margin-top: 5px; font-size: 12px;">{evolution['name'].capitalize()}</div>
                            </div>
                            """
                        evolution_html += '</div>'
                        st.markdown(evolution_html, unsafe_allow_html=True)
                    else:
                        st.markdown("Pas d'évolution connue")

        st.markdown("</div>", unsafe_allow_html=True)

        # Fermeture des conteneurs
        st.markdown("""
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Données brutes
with st.expander("Données brutes"):
    st.dataframe(df_region)
