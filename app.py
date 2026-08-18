from pathlib import Path

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
# Tema "spec sheet" (katalog produk): terang, ringan, aksen indigo buat data
# teknis dan aksen koral khusus buat HARGA -- ditampilkan sebagai price tag
# beneran (ada lubang gantungannya), bukan cuma angka polos.
BG = "#1C1A22"
SURFACE = "#272430"
SURFACE_BORDER = "#3D3947"
TEXT = "#F1EEF7"
TEXT_MUTED = "#9C96AC"
ACCENT = "#8B8FFF"
ACCENT_TINT = "#33304A"
PRICE = "#FF7A5C"
CATEGORY_COLORS = ["#8B8FFF", "#FF7A5C", "#22B8A3", "#F2B705", "#E0568C", "#4C9BE0", "#B39DFF", "#5CC98A"]


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background-color: {BG}; color: {TEXT}; }}

        h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; color: {TEXT}; letter-spacing: -0.01em; }}

        [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace !important;
            color: {ACCENT} !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}
        div[data-testid="stMetric"] {{
            background-color: {SURFACE};
            border: 1px solid {SURFACE_BORDER};
            border-top: 2px solid {ACCENT};
            border-radius: 6px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {SURFACE};
            border-color: {SURFACE_BORDER} !important;
            border-radius: 10px;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {SURFACE_BORDER}; }}
        .stTabs [data-baseweb="tab"] {{ color: {TEXT_MUTED}; font-family: 'Space Grotesk', sans-serif; }}
        .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

        .ledger-caption {{ color: {TEXT_MUTED}; font-size: 0.85rem; margin-top: -0.4rem; }}
        section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {SURFACE_BORDER}; }}

        /* --- signature: price tag --- */
        .price-tag {{
            position: relative;
            display: inline-flex;
            flex-direction: column;
            background: {PRICE};
            color: white;
            padding: 0.7rem 1.6rem 0.7rem 2.3rem;
            border-radius: 0 10px 10px 0;
            margin: 0.4rem 0 0.8rem 14px;
        }}
        .price-tag::before {{
            content: "";
            position: absolute;
            left: -12px; top: 50%;
            transform: translateY(-50%);
            width: 18px; height: 18px;
            background: {BG};
            border-radius: 50%;
            border: 2px solid {PRICE};
        }}
        .price-tag-value {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.4rem; }}
        .price-tag-sub {{ font-family: 'Inter', sans-serif; font-size: 0.78rem; opacity: 0.9; }}

        /* --- signature: spec chip (dipakai di tab Compare) --- */
        .spec-chip-row {{ margin-bottom: 0.35rem; }}
        .spec-chip-label {{ color: {TEXT_MUTED}; font-size: 0.78rem; display: block; margin-bottom: 2px; }}
        .spec-chip {{
            display: inline-block;
            background: {ACCENT_TINT};
            color: {ACCENT};
            border: 1px solid {ACCENT};
            border-radius: 999px;
            padding: 0.15rem 0.75rem;
            font-size: 0.85rem;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
        }}
        .spec-card {{
            background: {SURFACE};
            border: 1px solid {SURFACE_BORDER};
            border-radius: 10px;
            padding: 1rem 1.1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def register_plotly_theme() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        title_font=dict(family="Space Grotesk, sans-serif", color=TEXT, size=16),
        colorway=CATEGORY_COLORS,
        xaxis=dict(gridcolor=SURFACE_BORDER, zerolinecolor=SURFACE_BORDER, linecolor=SURFACE_BORDER),
        yaxis=dict(gridcolor=SURFACE_BORDER, zerolinecolor=SURFACE_BORDER, linecolor=SURFACE_BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=48, l=10, r=10, b=10),
    )
    pio.templates["spec_sheet"] = template
    px.defaults.template = "spec_sheet"


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


def price_tag_html(value_eur: float, currency: str, sub: str = "") -> str:
    return f"""
    <div class="price-tag">
        <span class="price-tag-value">{fmt_price(value_eur, currency)}</span>
        {f'<span class="price-tag-sub">{sub}</span>' if sub else ''}
    </div>
    """


def spec_chip(label: str, value: str) -> str:
    return f"""
    <div class="spec-chip-row">
        <span class="spec-chip-label">{label}</span>
        <span class="spec-chip">{value}</span>
    </div>
    """


# ========== PAGE ==========
st.set_page_config(page_title="Laptop Price Predict", page_icon="💻", layout="wide")
inject_theme()
register_plotly_theme()

st.title("💻 Laptop Price Predict")
st.caption("Regresi Random Forest buat estimasi harga laptop dari spesifikasi teknisnya.")

if not DATA_FILE.exists():
    st.warning("Dataset belum ada. Taruh `laptop_data.csv` di dalam folder `data/`.")
    st.stop()

df = load_data()

currency = st.sidebar.radio("Mata uang tampilan", ["EUR", "IDR"], horizontal=True)

st.sidebar.header("Filter")
company_filter = st.sidebar.multiselect(
    "Company", sorted(df["company"].unique()), default=sorted(df["company"].unique())[:6],
)
opsys_filter = st.sidebar.multiselect(
    "OS", sorted(df["opsys"].unique()), default=sorted(df["opsys"].unique()),
)
min_ram, max_ram = int(df["ram_gb"].min()), int(df["ram_gb"].max())
ram_range = st.sidebar.slider("RAM (GB)", min_ram, max_ram, (min_ram, max_ram))

filtered = df[
    df["company"].isin(company_filter)
    & df["opsys"].isin(opsys_filter)
    & df["ram_gb"].between(ram_range[0], ram_range[1])
].copy()
st.sidebar.caption(f"📌 {len(filtered):,} dari {len(df):,} laptop cocok dengan filter ini.".replace(",", "."))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(filtered):,}".replace(",", "."))
col2.metric("Median Price", fmt_price(filtered["price_in_euros"].median(), currency) if not filtered.empty else "-")
col3.metric("Avg RAM", f"{filtered['ram_gb'].mean():.1f} GB" if not filtered.empty else "-")
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
        st.write("Masukkan spesifikasi laptop buat memprediksi harganya secara langsung.")
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
                st.markdown(price_tag_html(pred_eur, currency, sub), unsafe_allow_html=True)

with tab_compare:
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
            ("Company", "company"), ("Type", "type_name"), ("Inches", "inches"), ("CPU", "cpu"),
            ("RAM", "ram"), ("Memory", "memory"), ("GPU", "gpu"), ("OS", "opsys"), ("Weight", "weight"),
        ]

        card_a, card_b = st.columns(2)
        for col, row, label in ((card_a, row_a, pick_a), (card_b, row_b, pick_b)):
            with col:
                chips_html = "".join(spec_chip(name, str(row[key])) for name, key in spec_rows)
                st.markdown(f'<div class="spec-card"><h4>{label}</h4>{chips_html}</div>', unsafe_allow_html=True)
                cheaper = row_a["price_in_euros"] <= row_b["price_in_euros"]
                is_a = row.equals(row_a)
                sub = "Lebih murah" if (cheaper == is_a) else ""
                st.markdown(price_tag_html(row["price_in_euros"], currency, sub), unsafe_allow_html=True)

with tab_data:
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
