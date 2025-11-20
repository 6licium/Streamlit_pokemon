import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from collections import defaultdict
from coords import regions_coords
from io import BytesIO
import base64

st.title("Cartes des régions Pokémon")

# Sélection de la région
regions = list(regions_coords.keys())
region = st.selectbox("Choisir une région", regions)

# Charger la carte
image_path = f"cartes/carte_{region.lower()}.png"
try:
    img = Image.open(image_path)
except FileNotFoundError:
    st.error(f"L'image pour {region} est introuvable : {image_path}")
    st.stop()

# Encode image
buffer = BytesIO()
img.save(buffer, format="PNG")
encoded_img = base64.b64encode(buffer.getvalue()).decode()

# Charger les données
try:
    df = pd.read_csv("pokemon_location_encounters_full.csv")
    df = df[df["region"] == region]
except FileNotFoundError:
    st.error("Fichier 'pokemon_location_encounters_full.csv' introuvable.")
    st.stop()

# Préparation
coords = regions_coords[region]
pokemon_by_location = defaultdict(list)
for _, row in df.iterrows():
    pokemon_by_location[row["location"]].append(row.to_dict())

# Création figure Plotly
fig = go.Figure()

# Image de fond
fig.add_layout_image(
    dict(
        source="data:image/png;base64," + encoded_img,
        xref="x",
        yref="y",
        x=0,
        y=img.size[1],
        sizex=img.size[0],
        sizey=img.size[1],
        sizing="stretch",
        layer="below"
    )
)

# Axes invisibles
fig.update_xaxes(visible=False, range=[0, img.size[0]], fixedrange=True)
fig.update_yaxes(visible=False, range=[img.size[1], 0], fixedrange=True)

fig.update_layout(
    dragmode=False,
    autosize=False,
    width=1000,
    height=800,
    margin=dict(l=0, r=0, t=0, b=0),
    clickmode='event+select'
)

# Ajout des bulles
x_points, y_points, hover_texts, locations = [], [], [], []
for lieu, pokes in pokemon_by_location.items():
    if lieu in coords:
        x, y = coords[lieu]
        x_points.append(x)
        y_points.append(y)
        locations.append(lieu)
        hover_texts.append(f"{len(pokes)} Pokémon<br><b>Plus →</b>")

# Trace cliquable
fig.add_trace(
    go.Scatter(
        x=x_points,
        y=y_points,
        mode="markers",
        marker=dict(size=40, color="white", opacity=0.25,
                    line=dict(width=2, color="black")),
        hoverinfo="text",
        text=hover_texts,
        customdata=locations,
        hovertemplate="<b>%{customdata}</b><br>%{text}<extra></extra>"
    )
)

# Survol (halo)
fig.add_trace(
    go.Scatter(
        x=x_points,
        y=y_points,
        mode="markers",
        marker=dict(size=55, color="yellow", opacity=0.08),
        hoverinfo="skip",
        showlegend=False
    )
)

# 👉 UNIQUE affichage (correct)
plot = st.plotly_chart(fig, use_container_width=True)

# Gestion clic
if 'clicked_location' not in st.session_state:
    st.session_state.clicked_location = None
    st.session_state.clicked_pokemon_list = []
    st.session_state.clicked_pokemon = None

selected = fig.data[0].selectedpoints
if selected:
    idx = selected[0]
    st.session_state.clicked_location = locations[idx]
    st.session_state.clicked_pokemon_list = pokemon_by_location[locations[idx]]

# Liste Pokémon
if st.session_state.clicked_location:
    lieu = st.session_state.clicked_location
    with st.expander(f"Pokémon trouvés à {lieu}"):
        for poke in st.session_state.clicked_pokemon_list:
            if st.button(poke["name"], key=f"btn-{poke['name']}"):
                st.session_state.clicked_pokemon = poke

# Détail Pokémon
if st.session_state.clicked_pokemon:
    p = st.session_state.clicked_pokemon
    st.subheader(f"{p['name']}")
    st.write(f"ID : {p['pokemon_id']}")
    st.write(f"Location : {p['location']}")

# Données brutes
st.subheader("Données brutes")
st.dataframe(df)
