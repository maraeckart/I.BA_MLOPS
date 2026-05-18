from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st


PREDICTIONS_DIR = Path("data/predictions/live")


REGION_COORDINATES = {
    "Europe": {"lat": 50.1109, "lon": 8.6821},
    "United Kingdom": {"lat": 51.5074, "lon": -0.1278},
    "France": {"lat": 48.8566, "lon": 2.3522},
    "Germany": {"lat": 52.52, "lon": 13.405},
    "Switzerland": {"lat": 46.8182, "lon": 8.2275},
    "United States": {"lat": 38.9072, "lon": -77.0369},
    "World": {"lat": 20.0, "lon": 0.0},
    "Middle East": {"lat": 29.2985, "lon": 42.551},
    "Asia": {"lat": 34.0479, "lon": 100.6197},
    "Africa": {"lat": -8.7832, "lon": 34.5085},
}


st.set_page_config(
    page_title="News Topic Monitoring",
    layout="wide",
)

# --- REFINED CSS STYLING ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Arial+Black&display=swap');

    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
        background-color: #fcfcfc;
        color: #111111;
    }

    .stApp {
        background-color: #fcfcfc;
    }

    /* Branded Header Styling */
    .main-title {
        font-family: 'Arial Black', Arial, sans-serif;
        font-weight: 900;
        letter-spacing: -0.01em;
        text-transform: uppercase;
        font-size: 4.2rem;
        color: #111111;
        margin-bottom: 0rem;
        line-height: 1.1;
    }

    .subtitle {
        display: inline-block;
        background-color: #111111;
        color: #ffffff;
        padding: 0.25rem 0.6rem;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 2.5rem;
        letter-spacing: 0.02em;
    }

    .date-banner {
        display: inline-block;
        background-color: #eff55a;
        color: #111111;
        padding: 0.4rem 1rem;
        border-radius: 2rem; /* Matches pill design in image */
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* Refined Topic Card Styles - Text forced to pure black */
    .topic-card {
        border-radius: 2rem; /* Pill/bubble aesthetic */
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: none;
        border: none;
    }

    .topic-title {
        color: #111111 !important; /* Pure black text */
        font-family: 'Arial Black', Arial, sans-serif;
        font-weight: 900;
        font-size: 1.25rem;
        line-height: 1.3;
        margin-bottom: 0.6rem;
    }

    .topic-count {
        color: #111111 !important;
        font-size: 0.85rem;
        font-weight: 700;
        opacity: 0.75;
    }

    section[data-testid="stSidebar"] {
        background-color: #f1f1f1;
    }

    /* FIXED: Force high-contrast text color on tables, expanders, and list items */
    table, th, td, tr, .dataframe {
        color: #111111 !important;
        background-color: #ffffff !important;
    }
    
    thead th {
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Target Streamlit expander headers and inner content text visibility */
    .streamlit-expanderHeader, p, span, li {
        color: #111111 !important;
    }

    /* Keep hyperlinks clear and legible */
    table a {
        color: #1a0dab !important;
        text-decoration: underline !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TOPIC_COLORS = [
    "#d5eef2",  # Light blue-gray
    "#aebbd0",  # Muted slate
    "#dedaff",  # Light lavender
    "#ef64ad",  # Vibrant pink
    "#eff55a",  # Trending Yellow
    "#d9f5c5",  # Soft green
    "#ffd6a5",  # Pastel orange
]


def get_prediction_files(predictions_dir: Path = PREDICTIONS_DIR) -> list[Path]:
    if not predictions_dir.exists():
        return []

    return sorted(
        predictions_dir.glob("topic_predictions_*.csv"),
        reverse=True,
    )


def extract_date_from_filename(file_path: Path) -> str:
    return file_path.stem.replace("topic_predictions_", "")


def format_date_display(date_string: str) -> str:
    parsed_date = pd.to_datetime(date_string, errors="coerce")

    if pd.isna(parsed_date):
        return date_string

    month_names = {
        1: "JAN",
        2: "FEB",
        3: "MAR",
        4: "APR",
        5: "MAI",
        6: "JUN",
        7: "JUL",
        8: "AUG",
        9: "SEP",
        10: "OKT",
        11: "NOV",
        12: "DEZ",
    }

    return f"{parsed_date.day:02d}.{month_names[parsed_date.month]}.{parsed_date.year}"


@st.cache_data
def load_predictions(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(
            df["published_at"],
            errors="coerce",
        )

    if "region" not in df.columns:
        if "country" in df.columns:
            df["region"] = df["country"]
        elif "source_name" in df.columns:
            df["region"] = df["source_name"]
        else:
            df["region"] = "World"

    return df


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df.copy()

    with st.sidebar:
        st.header("Filters")

        if "source_name" in df.columns:
            sources = sorted(df["source_name"].dropna().unique())
            selected_sources = st.multiselect(
                "Source",
                options=sources,
                default=sources,
            )
            filtered_df = filtered_df[filtered_df["source_name"].isin(selected_sources)]

        if "region" in df.columns:
            regions = sorted(df["region"].dropna().unique())
            selected_regions = st.multiselect(
                "Region",
                options=regions,
                default=regions,
            )
            filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]

        if "topic_id" in df.columns:
            topics = sorted(df["topic_id"].dropna().unique())
            selected_topics = st.multiselect(
                "Topic",
                options=topics,
                default=topics,
            )
            filtered_df = filtered_df[filtered_df["topic_id"].isin(selected_topics)]

        if "headline" in df.columns:
            search_query = st.text_input("Search headline")

            if search_query:
                filtered_df = filtered_df[
                    filtered_df["headline"]
                    .fillna("")
                    .str.contains(search_query, case=False, regex=False)
                ]

    return filtered_df

def show_header(selected_date: str) -> None:
    # Uses the streamlined layout matching the attached image
    st.markdown('<div class="main-title">NEWS TOPIC MONITORING</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Live news articles assigned to unsupervised topics using the trained NMF topic model.</div>',
        unsafe_allow_html=True,
    )
    
    formatted_date = format_date_display(selected_date)
    st.markdown(
        f'<div class="date-banner">TRENDING TODAY : {formatted_date}</div>',
        unsafe_allow_html=True,
    )

def show_topic_cards(df: pd.DataFrame) -> None:
    if "topic_id" not in df.columns or "topic_keywords" not in df.columns:
        st.warning("Topic columns not found.")
        return

    topic_summary = get_topic_summary(df)

    # Render stacked topic cards cleanly inside its layout container
    for index, row in topic_summary.iterrows():
        topic_id = int(row["topic_id"])
        count = int(row["count"])
        importance = row["importance"] * 100
        keywords = str(row["topic_keywords"])

        color = TOPIC_COLORS[topic_id % len(TOPIC_COLORS)]

        st.markdown(
            f"""
            <div class="topic-card" style="background-color: {color};">
                <div class="topic-title">{keywords}</div>
                <div class="topic-count">{count} articles · {importance:.1f}% importance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def show_kpis(df: pd.DataFrame) -> None:
    total_articles = len(df)

    num_sources = df["source_name"].nunique() if "source_name" in df.columns else 0
    num_topics = df["topic_id"].nunique() if "topic_id" in df.columns else 0
    num_regions = df["region"].nunique() if "region" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_articles}</div>
                <div class="metric-label">Articles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{num_topics}</div>
                <div class="metric-label">Topics</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{num_sources}</div>
                <div class="metric-label">Sources</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{num_regions}</div>
                <div class="metric-label">Regions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    topic_summary = (
        df.groupby("topic_id")
        .agg(
            count=("topic_id", "size"),
            topic_keywords=("topic_keywords", "first"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    total = topic_summary["count"].sum()

    if total > 0:
        topic_summary["importance"] = topic_summary["count"] / total
    else:
        topic_summary["importance"] = 0

    return topic_summary



def add_region_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    map_df = df.copy()

    def get_lat(region: str) -> float:
        return REGION_COORDINATES.get(region, REGION_COORDINATES["World"])["lat"]

    def get_lon(region: str) -> float:
        return REGION_COORDINATES.get(region, REGION_COORDINATES["World"])["lon"]

    map_df["region"] = map_df["region"].fillna("World").astype(str)
    map_df["lat"] = map_df["region"].apply(get_lat)
    map_df["lon"] = map_df["region"].apply(get_lon)

    if "topic_id" in map_df.columns:
        map_df["color"] = map_df["topic_id"].apply(
            lambda topic_id: hex_to_rgb(TOPIC_COLORS[int(topic_id) % len(TOPIC_COLORS)])
        )
    else:
        map_df["color"] = [[239, 245, 90] for _ in range(len(map_df))]

    return map_df


def hex_to_rgb(hex_color: str) -> list[int]:
    hex_color = hex_color.lstrip("#")
    return [
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    ]


def show_region_map(df: pd.DataFrame) -> None:
    #st.markdown("## Regional Topic Map")

    required_columns = {"region", "lat", "lon"}
    if not required_columns.issubset(df.columns):
        st.warning("Map requires 'region', 'lat', and 'lon' columns.")
        return

    if df.empty:
        st.info("No articles available for the current filters.")
        return

    map_df = df.copy()

    region_counts = (
        map_df.groupby(["region", "lat", "lon"], as_index=False)
        .agg(
            article_count=("headline", "count"),
            topic_id=("topic_id", "first"),
        )
    )

    max_count = region_counts["article_count"].max()
    if max_count == 0:
        max_count = 1

    region_counts["intensity"] = (
        200 + 55 * (region_counts["article_count"] / max_count)
    ).astype(int)

    region_counts["color"] = region_counts["intensity"].apply(
        lambda intensity: [239, 245, 90, intensity]
    )

    region_counts["radius"] = 90000

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=region_counts,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=1.0,        
        stroked=False,       
    )

    view_state = pdk.ViewState(
        latitude=35,
        longitude=15,
        zoom=1.15,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": """
                <b>{region}</b><br/>
                Articles: {article_count}<br/>
                Main topic: {topic_id}
            """,
            "style": {
                "backgroundColor": "white",
                "color": "black",
                "fontFamily": "Arial",
            },
        },
        map_style="light",
    )

    st.pydeck_chart(deck, use_container_width=True)


def show_article_table(df: pd.DataFrame) -> None:
    st.markdown("## Articles")

    if "headline" not in df.columns:
        st.warning("Column 'headline' not found.")
        return

    table_df = df.copy()

    if "url" in table_df.columns:
        table_df["article"] = table_df.apply(
            lambda row: f'<a href="{row["url"]}" target="_blank">{row["headline"]}</a>',
            axis=1,
        )
    else:
        table_df["article"] = table_df["headline"]

    columns = ["article"]

    for column in [ "topic_keywords", "published_at"]:
        if column in table_df.columns:
            columns.append(column)

    st.write(
        table_df[columns].to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

def show_example_headlines(df: pd.DataFrame) -> None:
    st.markdown("## Example Headlines by Topic")

    if "topic_id" not in df.columns or "headline" not in df.columns:
        st.warning("Columns 'topic_id' and/or 'headline' not found.")
        return

    for topic_id in sorted(df["topic_id"].dropna().unique()):
        topic_df = df[df["topic_id"] == topic_id]
        examples = (
                    df[df["topic_id"] == topic_id]
                    .sort_values("topic_score", ascending=False)
                    .head(8)
                )

        if topic_df.empty:
            continue

        keywords = ""

        if "topic_keywords" in examples.columns:
            keywords = examples["topic_keywords"].iloc[0]

        with st.expander(f"{keywords}"):
            for _, row in examples.head(8).iterrows():
                headline = row.get("headline", "")

                if "url" in row and pd.notna(row["url"]):
                    st.markdown(f"- [{headline}]({row['url']})")
                else:
                    st.write(f"- {headline}")




def main() -> None:
    prediction_files = get_prediction_files()

    if not prediction_files:
        st.error("No prediction files found.")
        return

    file_options = {extract_date_from_filename(f): f for f in prediction_files}
    selected_date = st.sidebar.selectbox("Prediction date", options=list(file_options.keys()))
    df = load_predictions(str(file_options[selected_date]))
    filtered_df = filter_dataframe(df)

    # Main Visual Layout Configuration
    show_header(selected_date)
    
    # Side-by-side visualization matching your design breakdown
    left_column, right_column = st.columns([1, 1.2])

    with left_column:
        show_topic_cards(filtered_df)

    with right_column:
        # Adds coordinates dynamically to map data 
        map_ready_df = add_region_coordinates(filtered_df)
        show_region_map(map_ready_df)

    st.divider()
    show_example_headlines(filtered_df)
    st.divider()
    show_article_table(filtered_df)

if __name__ == "__main__":
    main()