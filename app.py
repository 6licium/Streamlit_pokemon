import streamlit as st
import pandas as pd
from PIL import Image
from collections import defaultdict
from io import BytesIO
import base64
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from coords import regions_coords

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Cartes des régions Pokémon")

# ---------------------------------------------------------
# CACHES
# ---------------------------------------------------------
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

@st.cache_data
def filter_region(df, region):
    return df[df["region"] == region]

# ---------------------------------------------------------
# SELECT REGION
# ---------------------------------------------------------
regions = list(regions_coords.keys())
region = st.selectbox("Choisir une région", regions)

# Chargement de la carte
image_path = f"cartes/carte_{region.lower()}.png"
try:
    img, encoded_img = load_image(image_path)
except FileNotFoundError:
    st.error(f"Image introuvable : {image_path}")
    st.stop()

img_w, img_h = img.size

# Chargement des données
df = load_data()
df_region = filter_region(df, region)

# ---------------------------------------------------------
# PREP DONNÉES
# ---------------------------------------------------------
coords = regions_coords[region]
pokemon_by_location = defaultdict(list)
for _, row in df_region.iterrows():
    pokemon_by_location[row["location"]].append(row.to_dict())

locations = []
x_points = []
y_points = []
hover_texts = []

for lieu, pokes in pokemon_by_location.items():
    if lieu in coords:
        x, y = coords[lieu]
        locations.append(lieu)
        x_points.append(x)
        y_points.append(y)
        hover_texts.append(f"{len(pokes)} Pokémon<br><b>Cliquez pour voir la liste</b>")

# ---------------------------------------------------------
# FIXED SIZE & SCALING
# ---------------------------------------------------------
FIXED_W = 900
FIXED_H = int(FIXED_W * (img_h / img_w))

scale_x = FIXED_W / img_w
scale_y = FIXED_H / img_h

scaled_x = [x * scale_x for x in x_points]
scaled_y = [y * scale_y for y in y_points]  # PAS D'INVERSION — Plotly s’en occupe

# ---------------------------------------------------------
# FIGURE PLOTLY
# ---------------------------------------------------------
fig = go.Figure()

fig.add_layout_image(
    dict(
        source="data:image/png;base64," + encoded_img,
        xref="x",
        yref="y",
        x=0,
        y=0,
        sizex=FIXED_W,
        sizey=FIXED_H,
        sizing="stretch",
        layer="below",
    )
)

fig.update_xaxes(
    visible=False,
    range=[0, FIXED_W],
    fixedrange=True
)

fig.update_yaxes(
    visible=False,
    range=[0, FIXED_H],
    fixedrange=True,
    autorange="reversed"   # CORRIGE LE BUG DE CLIC/HOVER
)

# Points cliquables
fig.add_trace(
    go.Scatter(
        x=scaled_x,
        y=scaled_y,
        mode="markers",
        marker=dict(
            size=18,
            color="white",
            opacity=0.7,
            line=dict(width=2, color="black")
        ),
        customdata=locations,
        text=hover_texts,
        hovertemplate="<b>%{customdata}</b><br>%{text}<extra></extra>",
    )
)

# Halo
fig.add_trace(
    go.Scatter(
        x=scaled_x,
        y=scaled_y,
        mode="markers",
        marker=dict(size=30, color="yellow", opacity=0.2),
        hoverinfo="skip"
    )
)

fig.update_layout(
    width=FIXED_W,
    height=FIXED_H,
    margin=dict(l=0, r=0, t=0, b=0),
    dragmode=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

# ---------------------------------------------------------
# CENTRAGE DE LA CARTE
# ---------------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex; justify-content:center; width:100%; margin-top:10px;">
        <div style="width:{FIXED_W}px; height:{FIXED_H}px;">
    """,
    unsafe_allow_html=True
)

click_info = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_width=FIXED_W,
    override_height=FIXED_H
)

st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "clicked_location" not in st.session_state:
    st.session_state.clicked_location = None
    st.session_state.clicked_pokemon_list = []
    st.session_state.clicked_pokemon = None

if click_info:
    idx = click_info[0]["pointIndex"]
    lieu = locations[idx]
    st.session_state.clicked_location = lieu
    st.session_state.clicked_pokemon_list = pokemon_by_location[lieu]
    st.session_state.clicked_pokemon = None

# ---------------------------------------------------------
# LISTE DES POKÉMON
# ---------------------------------------------------------
if st.session_state.clicked_location:
    lieu = st.session_state.clicked_location
    with st.expander(f"Pokémon trouvés à {lieu}", expanded=True):
        search_term = st.text_input("Rechercher un Pokémon")
        for i, poke in enumerate(st.session_state.clicked_pokemon_list):
            if search_term.lower() in poke["name"].lower():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- **{poke['name']}** (ID: {poke['pokemon_id']})")
                unique_key = f"details-{poke['pokemon_id']}-{poke['location']}-{i}"
                with col2:
                    if st.button("Détails", key=unique_key):
                        st.session_state.clicked_pokemon = poke

# ---------------------------------------------------------
# DÉTAILS POKÉMON
# ---------------------------------------------------------
if st.session_state.clicked_pokemon:
    p = st.session_state.clicked_pokemon
    st.subheader(f"{p['name']}")
    st.write(f"ID : {p['pokemon_id']}")
    st.write(f"Location : {p['location']}")

# ---------------------------------------------------------
# DONNÉES BRUTES
# ---------------------------------------------------------
with st.expander("Données brutes"):
    st.dataframe(df_region)
