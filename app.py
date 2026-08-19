from pathlib import Path
import html as html_lib
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(exc)
    XGBRegressor = None


# ============================================================
# CONFIG
# ============================================================
APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "data" / "laptop_data.csv"
EUR_TO_IDR = 17500
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 600,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "min_child_weight": 2,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "objective": "reg:squarederror",
}

BASE_CAT_COLS = [
    "company",
    "type_name",
    "screen_resolution",
    "opsys",
    "cpu_brand",
    "cpu_family",
    "gpu_brand",
    "gpu_series",
]

BASE_NUM_COLS = [
    "ram",
    "inches",
    "weight",
    "ssd",
    "hdd",
    "flash",
    "hybrid",
    "total_storage",
    "screen_width",
    "screen_height",
    "screen_pixels",
    "cpu_generation",
    "cpu_clock",
    "cpu_tier",
    "gpu_model_num",
    "gpu_tier",
    "ppi_proxy",
    "ram_per_weight",
]

ALL_RAW_COLS = [
    "company",
    "product",
    "type_name",
    "inches",
    "screen_resolution",
    "cpu",
    "ram",
    "memory",
    "gpu",
    "opsys",
    "weight",
    "price_in_euros",
]

# ============================================================
# DESIGN TOKENS
# ============================================================
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
            --app-bg: {BG}; --panel-bg: {BG_PANEL}; --surface-border: {SURFACE_BORDER};
            --accent: {ACCENT}; --accent-bright: {ACCENT_BRIGHT}; --price: {PRICE};
            --price-bright: {PRICE_BRIGHT}; --text: {TEXT}; --text-muted: {TEXT_MUTED};
            --ink: {INK}; --win: {WIN};
        }}
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, rgba(91,140,255,0.08) 0, rgba(91,140,255,0) 80px),
                radial-gradient(circle at 88% 8%, rgba(212,167,44,0.05) 0, rgba(91,140,255,0) 70px),
                radial-gradient(circle at 60% 90%, rgba(45,212,191,0.10) 0, rgba(92,26,34,0) 80px),
                var(--app-bg) !important;
            color: var(--text);
        }}
        h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: 0.04em; }}
        .lab-title {{ font-family:'Space Grotesk',sans-serif; font-size:3rem; letter-spacing:.12em; color:var(--accent-bright); line-height:1.05; margin-bottom:0; }}
        .lab-sub {{ font-family:'JetBrains Mono',monospace; font-size:.75rem; letter-spacing:.08em; color:var(--text-muted); text-transform:uppercase; margin-top:4px; }}
        .lab-rule {{ height:2px; background:repeating-linear-gradient(90deg,var(--accent) 0 10px,transparent 10px 18px); margin:10px 0 22px; opacity:.7; }}
        section[data-testid="stSidebar"] {{ background:linear-gradient(180deg,var(--price) 0%,#0F1A2C 44%,var(--app-bg) 100%); border-right:2px dotted rgba(91,140,255,.32); }}
        section[data-testid="stSidebar"] * {{ color:var(--text); }}
        .lab-label {{ font-family:'Space Grotesk',sans-serif; font-size:1.15rem; letter-spacing:.12em; color:var(--accent-bright)!important; border-bottom:1px dashed rgba(143,177,255,.42); padding-bottom:7px; margin-bottom:11px; }}
        [data-testid="stMetric"] {{ background:var(--panel-bg); border:1px solid rgba(91,140,255,.22); border-top:2px solid var(--accent); border-radius:10px; padding:12px 16px; }}
        [data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace !important; color:var(--accent-bright)!important; }}
        [data-testid="stMetricLabel"] {{ font-family:'JetBrains Mono',monospace!important; color:var(--text-muted)!important; text-transform:uppercase; font-size:.68rem!important; letter-spacing:.08em; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--panel-bg); border-color:rgba(91,140,255,.20)!important; border-radius:12px; }}
        .stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid rgba(91,140,255,.22); }}
        .stTabs [data-baseweb="tab"] {{ font-family:'Space Grotesk',sans-serif; font-size:1rem; letter-spacing:.06em; color:var(--text-muted); background:var(--panel-bg); border-radius:8px 8px 0 0; padding:8px 14px; }}
        .stTabs [aria-selected="true"] {{ color:var(--accent-bright)!important; background:#16243D!important; box-shadow:inset 0 -3px 0 var(--accent); }}
        .section-caption {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--text-muted); letter-spacing:.03em; }}
        .ledger-caption {{ color:var(--text-muted); font-size:.82rem; }}
        .price-ticket {{ position:relative; display:flex; min-height:132px; margin:10px 2px; border-radius:12px; background:linear-gradient(135deg,var(--text) 0%,#E7EDF5 100%); box-shadow:0 12px 24px rgba(0,0,0,.46); overflow:hidden; }}
        .price-stub {{ width:72px; flex-shrink:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:7px; background:repeating-linear-gradient(135deg,#0F1726,#0F1726 6px,#182338 6px,#182338 12px); color:var(--accent); writing-mode:vertical-rl; transform:rotate(180deg); padding:10px 0; }}
        .price-stub .stub-label {{ font-family:'Space Grotesk',sans-serif; font-size:.9rem; letter-spacing:.18em; }}
        .price-stub .stub-no {{ font-family:'JetBrains Mono',monospace; font-size:.62rem; opacity:.8; }}
        .price-body {{ flex:1; display:flex; flex-direction:column; justify-content:center; gap:4px; padding:16px 122px 16px 18px; color:var(--ink); }}
        .price-kicker {{ font-family:'JetBrains Mono',monospace; font-size:.62rem; letter-spacing:.1em; text-transform:uppercase; color:var(--price); font-weight:700; }}
        .price-title {{ font-family:'Space Grotesk',sans-serif; font-size:1.65rem; line-height:1.1; }}
        .price-sub {{ font-size:.74rem; color:#5F5346; }}
        .price-badge {{ position:absolute; top:18px; right:18px; min-width:92px; height:72px; padding:0 12px; border-radius:12px; display:flex; flex-direction:column; align-items:center; justify-content:center; background:radial-gradient(circle at 30% 30%,var(--accent-bright),var(--accent) 75%); border:2px solid var(--ink); box-shadow:0 0 14px rgba(255,138,61,.40),inset 0 0 0 2px rgba(255,255,255,.3); }}
        .price-badge .value {{ font-family:'Space Grotesk',sans-serif; font-size:1.15rem; line-height:1; color:var(--ink); }}
        .price-badge .label {{ font-family:'JetBrains Mono',monospace; font-size:.43rem; letter-spacing:.05em; color:var(--ink); text-transform:uppercase; }}
        .spec-card {{ background:var(--panel-bg); border:1px solid rgba(91,140,255,.20); border-radius:12px; padding:1rem 1.1rem; }}
        .spec-card h4 {{ color:var(--accent-bright); margin-top:0; }}
        .spec-chip-label {{ color:var(--text-muted); font-family:'JetBrains Mono',monospace; font-size:.6rem; text-transform:uppercase; display:block; margin-bottom:2px; }}
        .spec-chip {{ display:inline-block; background:#182640; color:var(--accent-bright); border:1px solid rgba(91,140,255,.55); border-radius:999px; padding:.18rem .72rem; font-size:.82rem; margin-bottom:.3rem; }}
        .compare-price {{ margin-top:1rem; padding-top:.85rem; border-top:1px solid {SURFACE_BORDER}; display:flex; align-items:baseline; justify-content:space-between; }}
        .compare-price-value {{ color:{PRICE}; font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:700; }}
        .compare-price-sub {{ color:{WIN}; font-size:.78rem; font-weight:700; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def register_plotly_theme() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        font=dict(family="Inter, sans-serif", color=PAPER, size=13),
        title_font=dict(family="Space Grotesk, sans-serif", color=ACCENT_BRIGHT, size=18),
        colorway=[ACCENT, PRICE, TEAL, "#A78BFA", "#60A5FA", "#F59E0B"],
        xaxis=dict(gridcolor="rgba(91,140,255,0.14)", zerolinecolor="rgba(91,140,255,0.18)"),
        yaxis=dict(gridcolor="rgba(91,140,255,0.14)", zerolinecolor="rgba(91,140,255,0.18)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PAPER)),
        margin=dict(t=48, l=10, r=10, b=10),
    )
    pio.templates["laptop_price_lab"] = template
    px.defaults.template = "laptop_price_lab"


# ============================================================
# DATA + FEATURE ENGINEERING — aligned with notebook
# ============================================================
@st.cache_data(show_spinner="Memuat dan menyiapkan dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, encoding="ISO-8859-1")
    df = df.rename(columns={
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
    })
    df = df.drop(columns=["laptop_id"], errors="ignore")
    df["price_in_idr"] = df["price_in_euros"] * EUR_TO_IDR
    df["ram"] = df["ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
    df["weight"] = df["weight"].astype(str).str.replace("kg", "", regex=False).astype(float)
    df = df.drop_duplicates().copy()
    return df


def extract_cpu_features(cpu):
    text = str(cpu)
    low = text.lower()
    if low.startswith("intel"):
        brand = "intel"
    elif low.startswith("amd"):
        brand = "amd"
    elif low.startswith("samsung"):
        brand = "samsung"
    else:
        brand = "other"

    family_match = re.search(r"\b(core i[3579]|celeron|pentium|atom|xeon|ryzen [3579]|a\d(?:-series)?)\b", low)
    family = family_match.group(1).replace("-series", "") if family_match else "other"

    model_match = re.search(r"\b(\d{4,5})[a-z]{0,3}\b", low)
    generation = np.nan
    if model_match:
        model_number = model_match.group(1)
        generation = float(model_number[0] if len(model_number) == 4 else model_number[:2])

    clock_match = re.search(r"(\d+(?:\.\d+)?)\s*ghz", low)
    clock = float(clock_match.group(1)) if clock_match else np.nan

    cpu_tier = 0
    if "i9" in family or "ryzen 9" in family:
        cpu_tier = 4
    elif "i7" in family or "ryzen 7" in family or "xeon" in family:
        cpu_tier = 3
    elif "i5" in family or "ryzen 5" in family:
        cpu_tier = 2
    elif "i3" in family or "ryzen 3" in family or "pentium" in family:
        cpu_tier = 1
    return pd.Series([brand, family, generation, clock, cpu_tier])


def extract_gpu_features(gpu):
    low = str(gpu).lower()
    if low.startswith("nvidia"):
        brand = "nvidia"
    elif low.startswith("amd"):
        brand = "amd"
    elif low.startswith("intel"):
        brand = "intel"
    elif low.startswith("arm"):
        brand = "arm"
    else:
        brand = "other"

    if "geforce rtx" in low:
        series = "rtx"
    elif "geforce gtx" in low:
        series = "gtx"
    elif "geforce mx" in low:
        series = "mx"
    elif "quadro" in low:
        series = "quadro"
    elif "radeon" in low:
        series = "radeon"
    elif "iris" in low:
        series = "iris"
    elif "uhd" in low:
        series = "uhd"
    elif "hd graphics" in low:
        series = "hd"
    else:
        series = "other"

    numeric_tokens = re.findall(r"\b\d{3,4}\b", low)
    model_num = float(numeric_tokens[-1]) if numeric_tokens else np.nan
    return pd.Series([brand, series, model_num])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # Storage parsing — same logic as notebook
    for col in ["ssd", "hdd", "flash", "hybrid"]:
        work[col] = 0.0
    work["memory"] = work["memory"].astype(str).str.lower()

    for idx, memory_value in work["memory"].items():
        for item in str(memory_value).split("+"):
            item = item.strip()
            size_match = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb)", item)
            if not size_match:
                continue
            size = float(size_match.group(1))
            if size_match.group(2) == "tb":
                size *= 1000
            if "ssd" in item:
                work.at[idx, "ssd"] += size
            elif "hdd" in item:
                work.at[idx, "hdd"] += size
            elif "flash" in item:
                work.at[idx, "flash"] += size
            elif "hybrid" in item:
                work.at[idx, "hybrid"] += size

    work["total_storage"] = work[["ssd", "hdd", "flash", "hybrid"]].sum(axis=1)
    work["has_ssd"] = (work["ssd"] > 0).astype(int)
    work["has_hdd"] = (work["hdd"] > 0).astype(int)
    work["has_dual_storage"] = ((work["ssd"] > 0) & (work["hdd"] > 0)).astype(int)

    cpu_feats = work["cpu"].apply(extract_cpu_features)
    cpu_feats.columns = ["cpu_brand", "cpu_family", "cpu_generation", "cpu_clock", "cpu_tier"]
    work[cpu_feats.columns] = cpu_feats

    gpu_feats = work["gpu"].apply(extract_gpu_features)
    gpu_feats.columns = ["gpu_brand", "gpu_series", "gpu_model_num"]
    work[gpu_feats.columns] = gpu_feats
    work["gpu_tier"] = 0
    work.loc[work["gpu_series"].isin(["rtx", "quadro"]), "gpu_tier"] = 4
    work.loc[work["gpu_series"].eq("gtx"), "gpu_tier"] = 3
    work.loc[work["gpu_series"].isin(["mx", "radeon"]), "gpu_tier"] = 2
    work.loc[work["gpu_series"].isin(["iris", "uhd"]), "gpu_tier"] = 1

    resolution = work["screen_resolution"].astype(str).str.extract(r"(\d+)\s*x\s*(\d+)")
    work["screen_width"] = pd.to_numeric(resolution[0], errors="coerce")
    work["screen_height"] = pd.to_numeric(resolution[1], errors="coerce")
    work["screen_pixels"] = work["screen_width"] * work["screen_height"]
    work["is_ips"] = work["screen_resolution"].str.contains("ips", case=False, regex=False).astype(int)
    work["is_touchscreen"] = work["screen_resolution"].str.contains("touchscreen", case=False, regex=False).astype(int)
    work["is_retina"] = work["screen_resolution"].str.contains("retina", case=False, regex=False).astype(int)
    work["ppi_proxy"] = np.sqrt(work["screen_pixels"]) / work["inches"]
    work["ram_per_weight"] = work["ram"] / work["weight"]
    work = work.replace([np.inf, -np.inf], np.nan)
    return work


def build_feature_matrix(feature_df: pd.DataFrame):
    X = feature_df.drop(columns=["price_in_euros", "price_in_idr", "product", "memory", "cpu", "gpu"], errors="ignore").copy()
    X = pd.get_dummies(X, columns=BASE_CAT_COLS, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in BASE_NUM_COLS if c in X.columns]
    medians = X[numeric_cols].median(numeric_only=True)
    X[numeric_cols] = X[numeric_cols].fillna(medians)
    X = X.fillna(0)
    return X, numeric_cols, medians


# ============================================================
# MODEL — exact tuned configuration from latest notebook
# ============================================================
@st.cache_resource(show_spinner="XGBoost regression model trained with a feature-engineered pipeline for laptop price estimation.")
def train_model(df: pd.DataFrame):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(XGBOOST_ERROR)

    work = engineer_features(df)
    X, numeric_cols, medians = build_feature_matrix(work)
    y = work["price_in_idr"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Same ordering as notebook: standardization after split; only train is fitted.
    # StandardScaler returns float values. Build a new DataFrame for the scaled
    # block instead of assigning floats into integer-typed pandas columns.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    if numeric_cols:
        train_numeric = X_train[numeric_cols].astype("float64")
        test_numeric = X_test[numeric_cols].astype("float64")

        train_scaled = pd.DataFrame(
            scaler.fit_transform(train_numeric),
            index=X_train.index,
            columns=numeric_cols,
        )
        test_scaled = pd.DataFrame(
            scaler.transform(test_numeric),
            index=X_test.index,
            columns=numeric_cols,
        )

        X_train = X_train.drop(columns=numeric_cols).join(train_scaled)
        X_test = X_test.drop(columns=numeric_cols).join(test_scaled)

        # Restore the exact feature order used before scaling.
        X_train = X_train.reindex(columns=X.columns)
        X_test = X_test.reindex(columns=X.columns)

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, preds),
        "mae_idr": mean_absolute_error(y_test, preds),
        "rmse_idr": mean_squared_error(y_test, preds) ** 0.5,
    }
    context = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "medians": medians,
        "scaler": scaler,
    }
    return context, X_test, y_test, preds, metrics


def transform_new_laptop(input_row: pd.DataFrame, context):
    work = engineer_features(input_row)
    X_new = work.drop(columns=["price_in_euros", "price_in_idr", "product", "memory", "cpu", "gpu"], errors="ignore")
    X_new = pd.get_dummies(X_new, columns=BASE_CAT_COLS, drop_first=True)
    X_new = X_new.reindex(columns=context["feature_columns"], fill_value=0)
    for col in context["numeric_cols"]:
        if col in X_new.columns:
            X_new[col] = pd.to_numeric(X_new[col], errors="coerce").fillna(context["medians"].get(col, 0))
    X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)
    if context["numeric_cols"]:
        numeric_cols = context["numeric_cols"]
        numeric_frame = X_new[numeric_cols].astype("float64")

        scaled = pd.DataFrame(
            context["scaler"].transform(numeric_frame),
            index=X_new.index,
            columns=numeric_cols,
        )

        X_new = X_new.drop(columns=numeric_cols).join(scaled)
        X_new = X_new.reindex(columns=context["feature_columns"])

    return X_new


def get_feature_importance(context) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": context["feature_columns"], "importance": context["model"].feature_importances_})
    return imp.sort_values("importance", ascending=False).head(15)


def fmt_price(value_idr: float, currency: str) -> str:
    if currency == "IDR":
        return f"Rp{value_idr:,.0f}".replace(",", ".")
    return f"€{value_idr / EUR_TO_IDR:,.2f}"


def price_tag_html(value_idr: float, currency: str, sub: str = "", title: str = "Predicted Laptop Value") -> str:
    return f"""
    <div class="price-ticket">
      <div class="price-stub"><span class="stub-label">PRICE LAB</span><span class="stub-no">MODEL v2.0</span></div>
      <div class="price-body"><span class="price-kicker">XGBoost · Tuned Laptop Price Model</span><span class="price-title">{html_lib.escape(title)}</span><span class="price-sub">{html_lib.escape(sub or 'Estimated market value from CPU, GPU, storage, screen and specification signals.')}</span></div>
      <div class="price-badge"><span class="value">{fmt_price(value_idr, currency)}</span><span class="label">ESTIMATED PRICE</span></div>
    </div>
    """


def spec_chip(label: str, value: str) -> str:
    return f'<div><span class="spec-chip-label">{html_lib.escape(str(label))}</span><span class="spec-chip">{html_lib.escape(str(value))}</span></div>'


def render_compare_card(label: str, row: pd.Series, other: pd.Series, currency: str) -> str:
    chips = [
        spec_chip("Company", row["company"]),
        spec_chip("Type", row["type_name"]),
        spec_chip("CPU", row["cpu"]),
        spec_chip("RAM", row["ram"]),
        spec_chip("Memory", row["memory"]),
        spec_chip("GPU", row["gpu"]),
        spec_chip("Screen", row["screen_resolution"]),
        spec_chip("OS", row["opsys"]),
        spec_chip("Weight", row["weight"]),
    ]
    price = float(row["price_in_idr"])
    other_price = float(other["price_in_idr"])
    cheaper = price < other_price
    return (
        f'<div class="spec-card"><h4>{html_lib.escape(str(label))}</h4>'
        f'{"".join(chips)}'
        f'<div class="compare-price"><span class="compare-price-value">{fmt_price(price, currency)}</span>'
        f'<span class="compare-price-sub">{"💰 Termurah" if cheaper else ""}</span></div></div>'
    )


def price_position_gauge(pred_idr: float, filtered: pd.DataFrame, currency: str):
    vals = filtered["price_in_idr"]
    v_min, v_max = float(vals.min()), float(vals.max())
    q1, q3 = float(vals.quantile(.25)), float(vals.quantile(.75))
    percentile = float((vals <= pred_idr).mean() * 100)
    factor = 1 / EUR_TO_IDR if currency == "EUR" else 1
    prefix = "€" if currency == "EUR" else "Rp"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pred_idr * factor,
        number={"prefix": prefix, "valueformat": ",.0f", "font": {"family": "JetBrains Mono", "color": TEXT, "size": 28}},
        gauge={"axis": {"range": [v_min * factor, v_max * factor]}, "bar": {"color": PRICE, "thickness": .35},
               "bgcolor": SURFACE, "borderwidth": 0,
               "steps": [{"range": [v_min * factor, q1 * factor], "color": ACCENT_TINT},
                         {"range": [q1 * factor, q3 * factor], "color": SURFACE_BORDER},
                         {"range": [q3 * factor, v_max * factor], "color": ACCENT_TINT}]}
    ))
    fig.update_layout(height=200, margin=dict(t=10,b=10,l=30,r=30), paper_bgcolor=SURFACE)
    return fig, percentile


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Laptop Price Intelligence", page_icon="💻", layout="wide")
inject_theme()
register_plotly_theme()

st.markdown('<div class="lab-title">💻 LAPTOP PRICE INTELLIGENCE</div><div class="lab-sub">Feature-engineered XGBoost · CPU · GPU · Storage · Screen · Market price estimation</div><div class="lab-rule"></div>', unsafe_allow_html=True)

if not DATA_FILE.exists():
    st.error("Dataset belum ada. Taruh laptop_data.csv di folder data/.")
    st.stop()

if not XGBOOST_AVAILABLE:
    st.error("XGBoost tidak dapat dimuat di environment ini. Dashboard terbaru memang menggunakan XGBoost, sama seperti notebook.")
    st.code(XGBOOST_ERROR)
    st.info("Untuk macOS, dependency OpenMP biasanya perlu dipasang dengan: brew install libomp")
    st.stop()

df = load_data()
overall_median = df["price_in_idr"].median()
overall_avg_ram = df["ram"].mean()

currency = st.sidebar.radio("Mata uang tampilan", ["EUR", "IDR"], horizontal=True)
st.sidebar.markdown('<div class="lab-label">⚙️ MARKET FILTER</div>', unsafe_allow_html=True)
company_filter = st.sidebar.multiselect("Company", sorted(df["company"].unique()), default=sorted(df["company"].unique())[:6])
opsys_filter = st.sidebar.multiselect("OS", sorted(df["opsys"].unique()), default=sorted(df["opsys"].unique()))
ram_range = st.sidebar.slider("RAM (GB)", int(df["ram"].min()), int(df["ram"].max()), (int(df["ram"].min()), int(df["ram"].max())))

filtered = df[df["company"].isin(company_filter) & df["opsys"].isin(opsys_filter) & df["ram"].between(*ram_range)].copy()
st.sidebar.caption(f"📌 {len(filtered):,} dari {len(df):,} laptop cocok dengan filter ini.".replace(",", "."))

# Model is ALWAYS trained on the complete 1,275-row cleaned dataset,
# matching the notebook. Sidebar filters are market-context filters only.
model_context, X_test, y_test, preds, metrics = train_model(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(filtered):,}".replace(",", "."), delta=f"{len(filtered)/len(df)*100:.0f}% of dataset", delta_color="off")
if not filtered.empty:
    price_delta = (filtered["price_in_idr"].median() - overall_median) / overall_median * 100
    col2.metric("Median Price", fmt_price(filtered["price_in_idr"].median(), currency), delta=f"{price_delta:+.0f}% vs all data")
    col3.metric("Avg RAM", f"{filtered['ram'].mean():.1f} GB", delta=f"{filtered['ram'].mean()-overall_avg_ram:+.1f} GB vs all")
    col4.metric("Top Brand", filtered["company"].mode().iat[0])
else:
    col2.metric("Median Price", "-"); col3.metric("Avg RAM", "-"); col4.metric("Top Brand", "-")

tab_overview, tab_model, tab_whatif, tab_compare, tab_data = st.tabs(["📊 Overview", "🌲 Model", "🔮 What-if", "⚖️ Compare", "📄 Data"])

with tab_overview:
    if filtered.empty:
        st.info("Belum ada laptop yang cocok dengan filter.")
    else:
        with st.container(border=True):
            st.markdown('<span class="section-caption">SHOWROOM SNAPSHOT · MARKET DISTRIBUTION</span>', unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                fig = px.histogram(filtered, x="price_in_idr" if currency == "IDR" else "price_in_euros", nbins=40, title=f"Sebaran Harga ({currency})")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                counts = filtered["company"].value_counts().head(10).reset_index()
                counts.columns = ["company", "count"]
                fig = px.bar(counts, x="count", y="company", orientation="h", title="Top 10 Brand")
                fig.update_layout(yaxis={"categoryorder":"total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            st.markdown('<span class="section-caption">PRICE RANGE · BRAND POSITIONING</span>', unsafe_allow_html=True)
            top8 = filtered["company"].value_counts().head(8).index
            box_df = filtered[filtered["company"].isin(top8)]
            fig = px.box(box_df, x="company", y="price_in_idr" if currency == "IDR" else "price_in_euros", color="company", title=f"Sebaran harga ({currency}) di 8 brand terbanyak")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<p class="ledger-caption">Filter di sidebar hanya mengubah konteks pasar. Model dan metrik tetap berasal dari dataset lengkap seperti pada notebook.</p>', unsafe_allow_html=True)

with tab_model:
    with st.container(border=True):
        st.markdown('<span class="section-caption">MODEL DIAGNOSTICS · VALIDATION ROOM</span>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("R²", f"{metrics['r2']:.4f}", help="Notebook terbaru menghasilkan sekitar 0.9067 pada random_state=42.")
        c2.metric("MAE", fmt_price(metrics["mae_idr"], currency), help="Mean Absolute Error pada test set 20%.")
        c3.metric("RMSE", fmt_price(metrics["rmse_idr"], currency), help="Root Mean Squared Error pada test set 20%.")
        display_pred = preds if currency == "IDR" else preds / EUR_TO_IDR
        display_actual = y_test if currency == "IDR" else y_test / EUR_TO_IDR
        left,right = st.columns(2)
        with left:
            fig = px.scatter(x=display_actual, y=display_pred, labels={"x":"Harga Aktual","y":"Harga Prediksi"}, title="Actual vs Predicted")
            line = [min(display_actual.min(),display_pred.min()), max(display_actual.max(),display_pred.max())]
            fig.add_trace(go.Scatter(x=line,y=line,mode="lines",name="Ideal",line=dict(color=PRICE,dash="dash")))
            st.plotly_chart(fig,use_container_width=True)
        with right:
            residual = display_pred-display_actual
            fig = px.histogram(residual, nbins=40, title="Sebaran residual (prediksi − aktual)")
            fig.add_vline(x=0,line_dash="dash",line_color=PRICE)
            st.plotly_chart(fig,use_container_width=True)

    with st.container(border=True):
        st.markdown('<span class="section-caption">FEATURE IMPORTANCE · NOTEBOOK SIGNALS</span>', unsafe_allow_html=True)
        imp = get_feature_importance(model_context)
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Top 15 fitur paling berpengaruh")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f"Model menggunakan {len(model_context['feature_columns'])} fitur setelah feature engineering dan one-hot encoding.")

with tab_whatif:
    st.markdown('<span class="section-caption">WHAT-IF COUNTER · INPUT SPESIFIKASI YANG SAMA DENGAN NOTEBOOK</span>', unsafe_allow_html=True)
    with st.form("whatif_form"):
        c1,c2 = st.columns(2)
        with c1:
            company = st.selectbox("Company", sorted(df["company"].unique()))
            type_name = st.selectbox("Type", sorted(df["type_name"].unique()))
            opsys = st.selectbox("OS", sorted(df["opsys"].unique()))
            cpu = st.selectbox("CPU", sorted(df["cpu"].unique()))
            gpu = st.selectbox("GPU", sorted(df["gpu"].unique()))
        with c2:
            screen_resolution = st.selectbox("Screen resolution", sorted(df["screen_resolution"].unique()))
            memory = st.selectbox("Memory", sorted(df["memory"].unique()))
            inches = st.number_input("Screen size (inches)", min_value=float(df["inches"].min()), max_value=float(df["inches"].max()), value=float(df["inches"].median()))
            ram = st.number_input("RAM (GB)", min_value=int(df["ram"].min()), max_value=int(df["ram"].max()), value=int(df["ram"].median()), step=1)
            weight = st.number_input("Weight (kg)", min_value=float(df["weight"].min()), max_value=float(df["weight"].max()), value=float(df["weight"].median()), step=0.1)
        clicked = st.form_submit_button("Prediksi Harga", use_container_width=True)

    if clicked:
        input_row = pd.DataFrame([{
            "company":company,"product":"custom","type_name":type_name,"inches":inches,
            "screen_resolution":screen_resolution,"cpu":cpu,"ram":ram,"memory":memory,
            "gpu":gpu,"opsys":opsys,"weight":weight,"price_in_euros":0.0,"price_in_idr":0.0
        }])
        X_new = transform_new_laptop(input_row, model_context)
        pred_idr = float(model_context["model"].predict(X_new)[0])
        median_idr = filtered["price_in_idr"].median() if not filtered.empty else df["price_in_idr"].median()
        diff_pct = (pred_idr - median_idr) / median_idr * 100
        arah = "lebih mahal" if diff_pct >= 0 else "lebih murah"
        with st.container(border=True):
            st.markdown('<span class="section-caption">ESTIMATION TICKET</span>', unsafe_allow_html=True)
            st.markdown(price_tag_html(pred_idr, currency, f"{abs(diff_pct):.0f}% {arah} dari median pasar terfilter", "Predicted Laptop Value"), unsafe_allow_html=True)
            gauge_fig, pct = price_position_gauge(pred_idr, filtered if not filtered.empty else df, currency)
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.caption(f"Prediksi berada di sekitar persentil {pct:.0f} pada market context yang sedang dipilih.")

with tab_compare:
    if filtered.empty:
        st.info("Belum ada laptop yang cocok dengan filter.")
    else:
        options = (filtered["company"] + " " + filtered["product"] + " (" + filtered["type_name"] + ")").tolist()
        c1,c2 = st.columns(2)
        with c1: pick_a = st.selectbox("Laptop A", options, index=0)
        with c2: pick_b = st.selectbox("Laptop B", options, index=min(1,len(options)-1))
        row_a = filtered.iloc[options.index(pick_a)]
        row_b = filtered.iloc[options.index(pick_b)]
        ca,cb = st.columns(2)
        with ca: st.markdown(render_compare_card(pick_a,row_a,row_b,currency),unsafe_allow_html=True)
        with cb: st.markdown(render_compare_card(pick_b,row_b,row_a,currency),unsafe_allow_html=True)

with tab_data:
    with st.container(border=True):
        display_df = filtered.copy()
        display_df["price"] = display_df["price_in_idr"] if currency == "IDR" else display_df["price_in_euros"]
        st.dataframe(display_df.head(200), use_container_width=True)
        st.download_button("⬇️ Download data terfilter (CSV)", filtered.to_csv(index=False).encode("utf-8"), file_name="laptop_filtered.csv", mime="text/csv")
    with st.expander("Ringkasan statistik"):
        st.dataframe(filtered[["ram","inches","weight","price_in_euros","price_in_idr"]].describe().round(2), use_container_width=True)

st.caption("Model dashboard: XGBoost tuned, feature-engineered pipeline aligned with LaporanML_fixed_improved.ipynb · random_state=42 · train/test=80/20")
