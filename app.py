from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
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

st.set_page_config(page_title="Laptop Price Predict", page_icon="💻", layout="wide")
st.title("Laptop Price Predict")
st.caption("Interactive Streamlit version of the laptop regression project.")


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
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


def get_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_resource
def train_laptop_model(data: pd.DataFrame):
    model_df = data[["company", "type_name", "opsys", "inches", "ram_gb", "weight_kg", "price_in_idr"]].copy()
    X = model_df.drop(columns=["price_in_idr"])
    y = model_df["price_in_idr"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = ["company", "type_name", "opsys"]
    num_cols = ["inches", "ram_gb", "weight_kg"]

    preprocessor = ColumnTransformer(
        [
            ("cat", get_one_hot_encoder(), cat_cols),
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                num_cols,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # Feature importances
    ohe_cat_names = pipeline.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
    all_feat_names = list(ohe_cat_names) + num_cols
    importances = pipeline.named_steps["model"].feature_importances_

    feat_imp_df = (
        pd.DataFrame({"feature": all_feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    return pipeline, r2, mae, rmse, feat_imp_df, y_test, preds


df = load_data()
if df is None:
    st.warning("Dataset not found. Place `laptop_data.csv` inside `data/`.")
    st.stop()

st.sidebar.header("Filters")
company_filter = st.sidebar.multiselect(
    "Company",
    sorted(df["company"].unique()),
    default=sorted(df["company"].unique())[:6],
)
opsys_filter = st.sidebar.multiselect(
    "OS",
    sorted(df["opsys"].unique()),
    default=sorted(df["opsys"].unique()),
)
min_ram, max_ram = int(df["ram_gb"].min()), int(df["ram_gb"].max())
ram_range = st.sidebar.slider("RAM (GB)", min_ram, max_ram, (min_ram, max_ram))

filtered = df[
    df["company"].isin(company_filter)
    & df["opsys"].isin(opsys_filter)
    & df["ram_gb"].between(ram_range[0], ram_range[1])
].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(filtered):,}".replace(",", "."))
col2.metric("Median Price", f"Rp{filtered['price_in_idr'].median():,.0f}".replace(",", "."))
col3.metric("Avg RAM", f"{filtered['ram_gb'].mean():.1f} GB")
col4.metric("Top Brand", filtered["company"].mode().iat[0] if not filtered.empty else "-")

tab_overview, tab_predict, tab_model, tab_data = st.tabs(["Overview", "Live Predict", "Model & Evaluation", "Data"])

pipeline, r2, mae, rmse, feat_imp_df, y_test, preds = train_laptop_model(df)

with tab_overview:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            filtered,
            x="price_in_idr",
            nbins=40,
            title="Price Distribution (IDR)",
            color_discrete_sequence=["#e8a33d"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        top_brands = filtered["company"].value_counts().head(10).reset_index()
        top_brands.columns = ["company", "count"]
        fig = px.bar(top_brands, x="count", y="company", orientation="h", title="Top Brands")
        st.plotly_chart(fig, use_container_width=True)

with tab_predict:
    st.subheader("💡 Live Laptop Price Predictor")
    st.write("Masukkan spesifikasi laptop untuk mendapatkan estimasi harga secara real-time.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        in_company = st.selectbox("Company / Brand", sorted(df["company"].unique()))
        in_type = st.selectbox("Type Name", sorted(df["type_name"].unique()))
    with col_b:
        in_opsys = st.selectbox("Operating System", sorted(df["opsys"].unique()))
        in_ram = st.select_slider("RAM (GB)", options=sorted(df["ram_gb"].unique()), value=8)
    with col_c:
        in_inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
        in_weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.8, step=0.1)

    input_df = pd.DataFrame(
        [
            {
                "company": in_company,
                "type_name": in_type,
                "opsys": in_opsys,
                "inches": in_inches,
                "ram_gb": in_ram,
                "weight_kg": in_weight,
            }
        ]
    )

    pred_price_idr = pipeline.predict(input_df)[0]
    pred_price_eur = pred_price_idr / EUR_TO_IDR

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Predicted Price (IDR)", f"Rp {pred_price_idr:,.0f}".replace(",", "."))
    res_col2.metric("Predicted Price (EUR)", f"€ {pred_price_eur:,.2f}")

with tab_model:
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"Rp{mae:,.0f}".replace(",", "."))
    c3.metric("RMSE", f"Rp{rmse:,.0f}".replace(",", "."))

    left_m, right_m = st.columns(2)
    with left_m:
        fig_scatter = px.scatter(
            x=y_test,
            y=preds,
            labels={"x": "Actual (IDR)", "y": "Predicted (IDR)"},
            title="Actual vs Predicted",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with right_m:
        fig_feat = px.bar(
            feat_imp_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Feature Importances",
        )
        fig_feat.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_feat, use_container_width=True)

with tab_data:
    st.dataframe(filtered.head(200), use_container_width=True)
