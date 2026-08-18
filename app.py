from pathlib import Path

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


def one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


st.set_page_config(page_title="Laptop Price Predict", page_icon="💻", layout="wide")
st.title("Laptop Price Predict")
st.caption("Interactive Streamlit version of the laptop regression project.")

if not DATA_FILE.exists():
    st.warning("Dataset not found. Place `laptop_data.csv` inside `data/`.")
    st.stop()

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
df["price_in_idr"] = df["price_in_euros"] * 17500
df["ram_gb"] = df["ram"].str.replace("GB", "", regex=False).astype(int)
df["weight_kg"] = df["weight"].str.replace("kg", "", regex=False).astype(float)
df = df.drop_duplicates().copy()

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

tab_overview, tab_model, tab_data = st.tabs(["Overview", "Model", "Data"])

with tab_overview:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            filtered,
            x="price_in_idr",
            nbins=40,
            title="Price Distribution",
            color_discrete_sequence=["#e8a33d"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        top_brands = filtered["company"].value_counts().head(10).reset_index()
        top_brands.columns = ["company", "count"]
        fig = px.bar(
            top_brands,
            x="count",
            y="company",
            orientation="h",
            title="Top Brands",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick insight")
    st.write(
        "Harga laptop di project ini cenderung right-skewed, sehingga model tree-based cocok sebagai baseline."
    )

with tab_model:
    model_df = filtered[["company", "type_name", "opsys", "inches", "ram_gb", "weight_kg", "price_in_idr"]].copy()
    X = model_df.drop(columns=["price_in_idr"])
    y = model_df["price_in_idr"]
    if len(model_df) < 20:
        st.info("Need at least 20 rows after filtering to train a quick benchmark.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("cat", one_hot(), ["company", "type_name", "opsys"]),
                            (
                                "num",
                                Pipeline(
                                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                                ),
                                ["inches", "ram_gb", "weight_kg"],
                            ),
                        ]
                    ),
                ),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{r2_score(y_test, preds):.3f}")
        c2.metric("MAE", f"Rp{mean_absolute_error(y_test, preds):,.0f}".replace(",", "."))
        c3.metric("RMSE", f"Rp{rmse:,.0f}".replace(",", "."))

        fig = px.scatter(
            x=y_test,
            y=preds,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.dataframe(filtered.head(200), use_container_width=True)
