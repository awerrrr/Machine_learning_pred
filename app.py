from pathlib import Path
import html as html_lib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "data" / "laptop_data.csv"
EUR_TO_IDR = 17500

FEATURE_COLS = ["company", "type_name", "opsys", "inches", "ram_gb", "weight_kg"]
CAT_COLS = ["company", "type_name", "opsys"]
NUM_COLS = ["inches", "ram_gb", "weight_kg"]

# ========== TOKEN DESAIN ==========
# ========== TOKEN DESAIN ==========
# Konsep: "Laptop Price Intelligence" — dashboard teknologi modern yang
# terasa seperti product lab / price analytics, bukan tema film.
BG = "#0B1220"
BG_PANEL = "#111A2B"
SURFACE_BORDER = "#25344D"
ACCENT = "#5B8CFF"
ACCENT_BRIGHT = "#8FB1FF"
PRICE = "#FF8A3D"
PRICE_BRIGHT = "#FFB066"
TEAL = "#2DD4BF"
TEXT = "#F5F7FB"
TEXT_MUTED = "#98A6BB"
PAPER = "#EAF0F8"
INK = "#162033"
ACCENT_TINT = "#192746"
WIN = TEAL
SURFACE = BG_PANEL



def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

        :root {{
            --app-bg: {BG};
            --panel-bg: {BG_PANEL};
            --surface-border: {SURFACE_BORDER};
            --accent: {ACCENT};
            --accent-bright: {ACCENT_BRIGHT};
            --price: {PRICE};
            --price-bright: {PRICE_BRIGHT};
            --text: {TEXT};
            --text-muted: {TEXT_MUTED};
            --ink: {INK};
            --win: {WIN};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, rgba(91,140,255,0.08) 0, rgba(91,140,255,0) 80px),
                radial-gradient(circle at 88% 8%, rgba(212,167,44,0.05) 0, rgba(91,140,255,0) 70px),
                radial-gradient(circle at 60% 90%, rgba(45,212,191,0.10) 0, rgba(92,26,34,0) 80px),
                var(--app-bg) !important;
            color: var(--text);
        }}

        h1, h2, h3, h4 {{
            font-family: 'Bebas Neue', sans-serif !important;
            letter-spacing: 0.04em;
        }}

        .lab-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3.15rem;
            letter-spacing: 0.12em;
            color: var(--accent-bright);
            text-shadow: 0 0 12px rgba(244,217,118,0.38), 0 0 2px rgba(244,217,118,0.6);
            line-height: 1.05;
            margin-bottom: 0;
        }}

        .lab-sub {{
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 4px;
        }}

        .lab-rule {{
            height: 2px;
            background: repeating-linear-gradient(90deg, var(--accent) 0 10px, transparent 10px 18px);
            margin: 10px 0 22px 0;
            opacity: 0.7;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--price) 0%, #0F1A2C 44%, var(--app-bg) 100%);
            border-right: 2px dotted rgba(91,140,255,0.32);
        }}
        section[data-testid="stSidebar"] * {{ color: var(--text); }}
        section[data-testid="stSidebar"] .stCaption {{ color: var(--text-muted) !important; }}

        .lab-label {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            letter-spacing: 0.15em;
            color: var(--accent-bright) !important;
            border-bottom: 1px dashed rgba(143,177,255,0.42);
            padding-bottom: 7px;
            margin-bottom: 11px;
        }}

        [data-testid="stMetric"] {{
            background: var(--panel-bg);
            border: 1px solid rgba(91,140,255,0.22);
            border-top: 2px solid var(--accent);
            border-radius: 10px;
            padding: 12px 16px;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 22px rgba(0,0,0,0.35);
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'Bebas Neue', sans-serif !important;
            color: var(--accent-bright) !important;
            letter-spacing: 0.05em;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'Space Mono', monospace !important;
            color: var(--text-muted) !important;
            text-transform: uppercase;
            font-size: 0.68rem !important;
            letter-spacing: 0.08em;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel-bg);
            border-color: rgba(91,140,255,0.20) !important;
            border-radius: 12px;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            box-shadow: 0 12px 26px rgba(0,0,0,0.28);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid rgba(91,140,255,0.22);
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            background: var(--panel-bg);
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }}
        .stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
        .stTabs [aria-selected="true"] {{
            color: var(--accent-bright) !important;
            background: #16243D !important;
            box-shadow: inset 0 -3px 0 var(--accent);
        }}

        .section-caption {{
            font-family: 'Space Mono', monospace;
            font-size: 0.74rem;
            color: var(--text-muted);
            letter-spacing: 0.03em;
        }}

        .ledger-caption {{
            color: var(--text-muted);
            font-size: 0.82rem;
        }}

        /* ---------- signature: laptop price ticket ---------- */
        .price-ticket-wrapper {{
            padding: 8px 0 14px 0;
        }}
        .price-ticket {{
            position: relative;
            display: flex;
            min-height: 132px;
            margin: 10px 2px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--text) 0%, #E7EDF5 100%);
            box-shadow: 0 12px 24px rgba(0,0,0,0.46);
            border: 1px solid rgba(22,32,51,0.15);
            overflow: hidden;
        }}
        .price-stub {{
            width: 72px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 7px;
            background: repeating-linear-gradient(135deg, #0F1726, #0F1726 6px, #182338 6px, #182338 12px);
            color: var(--accent);
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            padding: 10px 0;
        }}
        .price-stub .stub-label {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            letter-spacing: 0.18em;
        }}
        .price-stub .stub-no {{
            font-family: 'Space Mono', monospace;
            font-size: 0.62rem;
            opacity: 0.8;
        }}
        .price-perforation {{
            border-left: 2px dashed rgba(35,26,21,0.3);
        }}
        .price-body {{
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 4px;
            padding: 16px 122px 16px 18px;
            color: var(--ink);
        }}
        .price-kicker {{
            font-family: 'Space Mono', monospace;
            font-size: 0.64rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--price);
            font-weight: 700;
        }}
        .price-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.75rem;
            letter-spacing: 0.03em;
            line-height: 1.1;
        }}
        .price-sub {{
            font-size: 0.74rem;
            color: #5F5346;
        }}
        .price-badge {{
            position: absolute;
            top: 18px;
            right: 18px;
            min-width: 92px;
            height: 72px;
            padding: 0 12px;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at 30% 30%, var(--accent-bright), var(--accent) 75%);
            border: 2px solid var(--ink);
            box-shadow: 0 0 14px rgba(255,138,61,0.40), inset 0 0 0 2px rgba(255,255,255,0.3);
        }}
        .price-badge .value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.28rem;
            line-height: 1;
            color: var(--ink);
        }}
        .price-badge .label {{
            font-family: 'Space Mono', monospace;
            font-size: 0.46rem;
            letter-spacing: 0.05em;
            color: var(--ink);
            text-transform: uppercase;
        }}
        .notch {{
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            top: 50%;
            transform: translateY(-50%);
            background: var(--app-bg);
            box-shadow: inset 0 0 5px rgba(0,0,0,0.35);
        }}
        .notch-left {{ left: -10px; }}
        .notch-right {{ right: -10px; }}

        /* ---------- specification cards ---------- */
        .spec-card {{
            background: var(--panel-bg);
            border: 1px solid rgba(91,140,255,0.20);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }}
        .spec-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 26px rgba(0,0,0,0.35);
            border-color: var(--accent);
        }}
        .spec-card h4 {{
            color: var(--accent-bright);
            margin-top: 0;
            margin-bottom: 0.8rem;
        }}
        .spec-chip-row {{ margin-bottom: 0.35rem; }}
        .spec-chip-label {{
            color: var(--text-muted);
            font-family: 'Space Mono', monospace;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            display: block;
            margin-bottom: 2px;
        }}
        .spec-chip {{
            display: inline-block;
            background: #182640;
            color: var(--accent-bright);
            border: 1px solid rgba(91,140,255,0.55);
            border-radius: 999px;
            padding: 0.18rem 0.72rem;
            font-size: 0.83rem;
        }}
        .spec-chip-win {{
            background: rgba(79,209,161,0.12);
            color: var(--win);
            border-color: var(--win);
            box-shadow: 0 0 0 1px var(--win);
            font-weight: 700;
        }}
        .spec-chip-win::after {{ content: "  ✓"; }}

        [data-testid="stDataFrame"] {{
            border: 1px solid rgba(91,140,255,0.22);
            border-radius: 10px;
            overflow: hidden;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 8px;
            border: 1px solid rgba(91,140,255,0.32);
            background: #142037;
            color: var(--text);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--accent);
            color: var(--accent-bright);
        }}

        .compare-price {{
            margin-top: 1rem;
            padding-top: 0.85rem;
            border-top: 1px solid {SURFACE_BORDER};
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.75rem;
        }}
        .compare-price-value {{
            color: {PRICE};
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.15rem;
            font-weight: 700;
        }}
        .compare-price-sub {{
            color: {WIN};
            font-size: 0.78rem;
            font-weight: 700;
        }}

        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{ animation: none !important; transition: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def register_plotly_theme() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(family="DM Sans, sans-serif", color=PAPER, size=13),
        title_font=dict(family="Space Grotesk, sans-serif", color=ACCENT_BRIGHT, size=18),
        colorway=[ACCENT, PRICE, TEAL, "#A78BFA", "#60A5FA", "#F59E0B"],
        xaxis=dict(gridcolor="rgba(91,140,255,0.14)", zerolinecolor="rgba(91,140,255,0.18)", linecolor="rgba(91,140,255,0.20)"),
        yaxis=dict(gridcolor="rgba(91,140,255,0.14)", zerolinecolor="rgba(91,140,255,0.18)", linecolor="rgba(91,140,255,0.20)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PAPER)),
        margin=dict(t=48, l=10, r=10, b=10),
    )
    pio.templates["laptop_price_lab"] = template
    px.defaults.template = "laptop_price_lab"

def one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data(show_spinner="Memuat data laptop...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, encoding="ISO-8859-1")
    df = df.rename(
        columns={
            "laptop_ID": "laptop_id",
            "Company": "company",
            "Product": "product",
            "TypeName": "type_name",
            "Inches": "inches",
            "ScreenResolution": "screen_resolution",
            "Cpu": "cpu",
            "Ram": "ram",
            "Memory": "memory",
            "Gpu": "gpu",
            "OpSys": "opsys",
            "Weight": "weight",
            "Price_in_euros": "price_in_euros",
        }
    )
    df["price_in_idr"] = df["price_in_euros"] * EUR_TO_IDR
    df["ram_gb"] = df["ram"].str.replace("GB", "", regex=False).astype(int)
    df["weight_kg"] = df["weight"].str.replace("kg", "", regex=False).astype(float)
    return df.drop_duplicates().copy()


@st.cache_resource(show_spinner="Melatih model prediksi harga...")
def train_regressor(df: pd.DataFrame):
    model_df = df[FEATURE_COLS + ["price_in_euros"]].copy()
    X = model_df.drop(columns=["price_in_euros"])
    y = model_df["price_in_euros"]  # dilatih dalam EUR, dikonversi saat ditampilkan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", one_hot(), CAT_COLS),
                        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM_COLS),
                    ]
                ),
            ),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae_eur": mean_absolute_error(y_test, preds),
        "rmse_eur": mean_squared_error(y_test, preds) ** 0.5,
    }
    return pipeline, X_test, y_test, preds, metrics


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    ohe_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT_COLS)
    feature_names = list(ohe_names) + NUM_COLS
    importances = pipeline.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )


def fmt_price(value_eur: float, currency: str) -> str:
    if currency == "IDR":
        return f"Rp{value_eur * EUR_TO_IDR:,.0f}".replace(",", ".")
    return f"€{value_eur:,.2f}"


def price_tag_html(value_eur: float, currency: str, sub: str = "", title: str = "Laptop Price") -> str:
    return f"""
    <div class="price-ticket-wrapper">
      <div class="price-ticket">
        <div class="notch notch-left"></div>
        <div class="price-stub">
          <span class="stub-label">PRICE LAB</span>
          <span class="stub-no">EST. 2026</span>
        </div>
        <div class="price-perforation"></div>
        <div class="price-body">
          <span class="price-kicker">Random Forest · Laptop Price Model</span>
          <span class="price-title">{title}</span>
          {f'<span class="price-sub">{sub}</span>' if sub else '<span class="price-sub">Estimated market value based on selected specifications.</span>'}
        </div>
        <div class="price-badge">
          <span class="value">{fmt_price(value_eur, currency)}</span>
          <span class="label">ESTIMATED PRICE</span>
        </div>
        <div class="notch notch-right"></div>
      </div>
    </div>
    """


def spec_chip(label: str, value: str, is_win: bool = False) -> str:
    """Render one comparison specification chip as safe HTML."""
    chip_class = "spec-chip spec-chip-win" if is_win else "spec-chip"
    safe_label = html_lib.escape(str(label))
    safe_value = html_lib.escape(str(value))
    return (
        f'<div class="spec-chip-row">'
        f'<span class="spec-chip-label">{safe_label}</span>'
        f'<span class="{chip_class}">{safe_value}</span>'
        f'</div>'
    )


def render_compare_card(label: str, row: pd.Series, other: pd.Series, spec_rows, currency: str) -> str:
    """Build the complete Compare card as one HTML fragment."""
    higher_is_better = {"ram_gb"}
    lower_is_better = {"weight_kg"}

    chips = []
    for name, key, numeric_key in spec_rows:
        is_win = False
        if numeric_key is not None:
            this_val, other_val = row[numeric_key], other[numeric_key]
            if numeric_key in higher_is_better:
                is_win = this_val > other_val
            elif numeric_key in lower_is_better:
                is_win = this_val < other_val
        chips.append(spec_chip(name, row[key], is_win=is_win))

    safe_label = html_lib.escape(str(label))
    safe_price = html_lib.escape(fmt_price(float(row["price_in_euros"]), currency))
    cheaper = float(row["price_in_euros"]) < float(other["price_in_euros"])
    price_sub = "💰 Termurah" if cheaper else ""

    return (
        f'<div class="spec-card">'
        f'<h4>{safe_label}</h4>'
        f'{"".join(chips)}'
        f'<div class="compare-price">'
        f'<span class="compare-price-value">{safe_price}</span>'
        f'<span class="compare-price-sub">{price_sub}</span>'
        f'</div>'
        f'</div>'
    )

def price_position_gauge(pred_eur: float, filtered: pd.DataFrame, currency: str) -> tuple[go.Figure, float]:
    """Gauge 'posisi harga': nunjukin prediksi ini duduk di rentang harga mana
    dibanding laptop lain yang lolos filter saat ini, biar angka prediksi
    nggak berdiri sendiri tanpa konteks pasar."""
    values = filtered["price_in_euros"]
    v_min, v_max = float(values.min()), float(values.max())
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    percentile = float((values <= pred_eur).mean() * 100)

    factor = EUR_TO_IDR if currency == "IDR" else 1
    prefix = "Rp" if currency == "IDR" else "€"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred_eur * factor,
            number={
                "prefix": prefix,
                "valueformat": ",.0f",
                "font": {"family": "JetBrains Mono", "color": TEXT, "size": 30},
            },
            gauge={
                "axis": {"range": [v_min * factor, v_max * factor], "tickfont": {"color": TEXT_MUTED, "size": 10}},
                "bar": {"color": PRICE, "thickness": 0.35},
                "bgcolor": SURFACE,
                "borderwidth": 0,
                "steps": [
                    {"range": [v_min * factor, q1 * factor], "color": ACCENT_TINT},
                    {"range": [q1 * factor, q3 * factor], "color": SURFACE_BORDER},
                    {"range": [q3 * factor, v_max * factor], "color": ACCENT_TINT},
                ],
            },
        )
    )
    fig.update_layout(height=200, margin=dict(t=10, b=10, l=30, r=30), paper_bgcolor=SURFACE, font_color=TEXT)
    return fig, percentile


# ========== PAGE ==========
st.set_page_config(page_title="Laptop Price Predict", page_icon="💻", layout="wide")
inject_theme()
register_plotly_theme()

st.markdown(
    """
    <div class="lab-title">💻 LAPTOP PRICE INTELLIGENCE</div>
    <div class="lab-sub">Specification analytics · Random Forest regression · Market price estimation</div>
    <div class="lab-rule"></div>
    """,
    unsafe_allow_html=True,
)

if not DATA_FILE.exists():
    st.warning("Dataset belum ada. Taruh `laptop_data.csv` di dalam folder `data/`.")
    st.stop()

df = load_data()
overall_median_price = df["price_in_euros"].median()
overall_avg_ram = df["ram_gb"].mean()

currency = st.sidebar.radio("Mata uang tampilan", ["EUR", "IDR"], horizontal=True)

st.sidebar.markdown('<div class="lab-label">⚙️ LAPTOP FILTER</div>', unsafe_allow_html=True)
company_filter = st.sidebar.multiselect(
    "Company", sorted(df["company"].unique()), default=sorted(df["company"].unique())[:6],
    key="company_filter",
)
opsys_filter = st.sidebar.multiselect(
    "OS", sorted(df["opsys"].unique()), default=sorted(df["opsys"].unique()),
    key="opsys_filter",
)
min_ram, max_ram = int(df["ram_gb"].min()), int(df["ram_gb"].max())
ram_range = st.sidebar.slider("RAM (GB)", min_ram, max_ram, (min_ram, max_ram), key="ram_range")

if st.sidebar.button("↺ Reset filter", use_container_width=True):
    for k in ("company_filter", "opsys_filter", "ram_range"):
        st.session_state.pop(k, None)
    st.rerun()

filtered = df[
    df["company"].isin(company_filter)
    & df["opsys"].isin(opsys_filter)
    & df["ram_gb"].between(ram_range[0], ram_range[1])
].copy()
st.sidebar.caption(f"📌 {len(filtered):,} dari {len(df):,} laptop cocok dengan filter ini.".replace(",", "."))

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Rows", f"{len(filtered):,}".replace(",", "."),
    delta=f"{len(filtered) / len(df) * 100:.0f}% dari total data" if not filtered.empty else None,
    delta_color="off",
)
if not filtered.empty:
    price_delta_pct = (filtered["price_in_euros"].median() - overall_median_price) / overall_median_price * 100
    ram_delta = filtered["ram_gb"].mean() - overall_avg_ram
    col2.metric(
        "Median Price", fmt_price(filtered["price_in_euros"].median(), currency),
        delta=f"{price_delta_pct:+.0f}% vs median semua data",
    )
    col3.metric(
        "Avg RAM", f"{filtered['ram_gb'].mean():.1f} GB",
        delta=f"{ram_delta:+.1f} GB vs rata-rata semua data",
    )
else:
    col2.metric("Median Price", "-")
    col3.metric("Avg RAM", "-")
col4.metric("Top Brand", filtered["company"].mode().iat[0] if not filtered.empty else "-")

tab_overview, tab_model, tab_whatif, tab_compare, tab_data = st.tabs(
    ["📊 Overview", "🌲 Model", "🔮 What-if", "⚖️ Compare", "📄 Data"]
)

price_col = "price_in_idr" if currency == "IDR" else "price_in_euros"
EMPTY_MSG = "Belum ada laptop yang cocok dengan filter ini. Coba longgarkan pilihan Company/OS/RAM di sidebar."

with tab_overview:
    if filtered.empty:
        st.info(EMPTY_MSG)
    else:
        with st.container(border=True):
            st.markdown('<span class="section-caption">SHOWROOM SNAPSHOT · MARKET DISTRIBUTION</span>', unsafe_allow_html=True)
            st.write("")
            left, right = st.columns(2)
            with left:
                fig = px.histogram(filtered, x=price_col, nbins=40, title=f"Sebaran Harga ({currency})")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                top_brands = filtered["company"].value_counts().head(10).reset_index()
                top_brands.columns = ["company", "count"]
                fig = px.bar(top_brands, x="count", y="company", orientation="h", title="Top 10 Brand")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            st.markdown('<span class="section-caption">PRICE RANGE · BRAND POSITIONING</span>', unsafe_allow_html=True)
            st.markdown("**Rentang harga per brand**")
            top8 = filtered["company"].value_counts().head(8).index
            box_df = filtered[filtered["company"].isin(top8)]
            fig = px.box(box_df, x="company", y=price_col, color="company", title=f"Sebaran harga ({currency}) di 8 brand terbanyak")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<p class="ledger-caption">Kotak yang panjang = harga di brand itu bervariasi jauh (banyak lini produk, dari murah sampai premium). Titik di luar kotak = outlier harga.</p>',
                unsafe_allow_html=True,
            )
            st.info("💡 Harga laptop di dataset ini cenderung *right-skewed* (banyak yang murah, sedikit yang sangat mahal) — ini alasan model tree-based (Random Forest) dipilih sebagai baseline, karena nggak terlalu terganggu distribusi yang miring kayak gini dibanding regresi linear biasa.")

with tab_model:
    if len(filtered) < 20:
        st.info("Perlu minimal 20 baris data setelah filter buat melatih model.")
    else:
        pipeline, X_test, y_test, preds, metrics = train_regressor(filtered)

        with st.container(border=True):
            st.markdown('<span class="section-caption">MODEL DIAGNOSTICS · VALIDATION ROOM</span>', unsafe_allow_html=True)
            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{metrics['r2']:.3f}", help="1.0 = prediksi sempurna; makin kecil, makin banyak variasi harga yang belum terjelaskan model.")
            c2.metric("MAE", fmt_price(metrics["mae_eur"], currency), help="Rata-rata selisih absolut prediksi vs harga asli.")
            c3.metric("RMSE", fmt_price(metrics["rmse_eur"], currency), help="Mirip MAE, tapi lebih menghukum kesalahan besar.")

            display_preds = preds * EUR_TO_IDR if currency == "IDR" else preds
            display_actual = y_test * EUR_TO_IDR if currency == "IDR" else y_test

            left, right = st.columns(2)
            with left:
                fig = px.scatter(x=display_actual, y=display_preds, labels={"x": "Harga Asli", "y": "Harga Prediksi"}, title="Actual vs Predicted")
                line_val = [min(display_actual.min(), display_preds.min()), max(display_actual.max(), display_preds.max())]
                fig.add_trace(go.Scatter(x=line_val, y=line_val, mode="lines", name="Prediksi sempurna", line=dict(color=PRICE, dash="dash")))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                residual = display_preds - display_actual
                fig = px.histogram(residual, nbins=40, title="Sebaran selisih (prediksi − aktual)", labels={"value": f"Selisih ({currency})"})
                fig.add_vline(x=0, line_dash="dash", line_color=PRICE)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<p class="ledger-caption">Garis putus-putus di scatter = target ideal (prediksi = aktual). Histogram kanan idealnya menumpuk di sekitar 0 — kalau condong ke satu sisi, model cenderung under/overestimate.</p>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown('<span class="section-caption">FEATURE IMPORTANCE · SPECIFICATION SIGNALS</span>', unsafe_allow_html=True)
            st.markdown("**Feature importance**")
            importance_df = get_feature_importance(pipeline)
            fig = px.bar(importance_df, x="importance", y="feature", orientation="h", title="Top 15 fitur paling berpengaruh")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

with tab_whatif:
    if len(filtered) < 20:
        st.info("Perlu minimal 20 baris data untuk melatih model prediksi.")
    else:
        pipeline, *_ = train_regressor(filtered)
        st.markdown('<span class="section-caption">WHAT-IF COUNTER · MASUKKAN SPESIFIKASI UNTUK ESTIMASI HARGA</span>', unsafe_allow_html=True)
        with st.form("laptop_whatif_form"):
            wc1, wc2 = st.columns(2)
            with wc1:
                company = st.selectbox("Company", sorted(filtered["company"].unique()))
                type_name = st.selectbox("Type", sorted(filtered["type_name"].unique()))
                opsys = st.selectbox("OS", sorted(filtered["opsys"].unique()))
            with wc2:
                inches = st.number_input("Screen size (inches)", value=float(filtered["inches"].median()))
                ram_gb = st.number_input("RAM (GB)", value=int(filtered["ram_gb"].median()), step=1)
                weight_kg = st.number_input("Weight (kg)", value=float(filtered["weight_kg"].median()))
            predict_clicked = st.form_submit_button("Prediksi Harga", use_container_width=True)

        if predict_clicked:
            input_row = pd.DataFrame([{
                "company": company,
                "type_name": type_name,
                "opsys": opsys,
                "inches": inches,
                "ram_gb": ram_gb,
                "weight_kg": weight_kg,
            }])
            pred_eur = pipeline.predict(input_row)[0]
            median_eur = filtered["price_in_euros"].median()
            diff_pct = (pred_eur - median_eur) / median_eur * 100
            arah = "lebih mahal" if diff_pct >= 0 else "lebih murah"
            sub = f"{abs(diff_pct):.0f}% {arah} dari median laptop terfilter"

            with st.container(border=True):
                st.markdown('<span class="section-caption">ESTIMATION TICKET</span>', unsafe_allow_html=True)
                st.markdown(price_tag_html(pred_eur, currency, sub, title="Predicted Laptop Value"), unsafe_allow_html=True)

                gauge_fig, percentile = price_position_gauge(pred_eur, filtered, currency)
                st.plotly_chart(gauge_fig, use_container_width=True)
                posisi = "lebih murah" if percentile <= 50 else "lebih mahal"
                st.markdown(
                    f'<p class="ledger-caption">📍 Prediksi ini <b>{posisi}</b> dari '
                    f'<b style="color:{ACCENT};">{percentile:.0f}%</b> laptop yang lolos filter saat ini. '
                    f'Zona abu-abu di tengah gauge = rentang IQR (25%–75%) harga pasar terfilter.</p>',
                    unsafe_allow_html=True,
                )

with tab_compare:
    st.markdown('<span class="section-caption">COMPARE DESK · SIDE-BY-SIDE SPECIFICATION CHECK</span>', unsafe_allow_html=True)
    st.write("")
    if filtered.empty:
        st.info(EMPTY_MSG)
    else:
        options = (filtered["company"] + " " + filtered["product"] + " (" + filtered["type_name"] + ")").tolist()
        cc1, cc2 = st.columns(2)
        with cc1:
            pick_a = st.selectbox("Laptop A", options, index=0)
        with cc2:
            pick_b = st.selectbox("Laptop B", options, index=min(1, len(options) - 1))

        row_a = filtered.iloc[options.index(pick_a)]
        row_b = filtered.iloc[options.index(pick_b)]

        spec_rows = [
            ("Company", "company", None),
            ("Type", "type_name", None),
            ("Inches", "inches", None),
            ("CPU", "cpu", None),
            ("RAM", "ram", "ram_gb"),
            ("Memory", "memory", None),
            ("GPU", "gpu", None),
            ("OS", "opsys", None),
            ("Weight", "weight", "weight_kg"),
        ]
        card_a, card_b = st.columns(2)
        with card_a:
            st.markdown(
                render_compare_card(pick_a, row_a, row_b, spec_rows, currency),
                unsafe_allow_html=True,
            )
        with card_b:
            st.markdown(
                render_compare_card(pick_b, row_b, row_a, spec_rows, currency),
                unsafe_allow_html=True,
            )

with tab_data:
    st.markdown('<span class="section-caption">DATA LEDGER · FILTERED LAPTOP INVENTORY</span>', unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        display_df = filtered.copy()
        display_df["price"] = display_df[price_col]
        st.dataframe(display_df.head(200), use_container_width=True)
        st.download_button(
            "⬇️ Download data terfilter (CSV)",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="laptop_filtered.csv",
            mime="text/csv",
        )
    with st.expander("Ringkasan statistik (describe)"):
        st.dataframe(filtered[NUM_COLS + ["price_in_euros"]].describe().round(2), use_container_width=True)
