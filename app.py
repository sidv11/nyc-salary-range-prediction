import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NYC Salary Range Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.2rem 2.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #151820 0%, #202633 100%);
            border: 1px solid rgba(255,255,255,.08);
            margin-bottom: 1.8rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.55rem;
            line-height: 1.15;
            letter-spacing: -.03em;
        }

        .hero p {
            margin: .8rem 0 0 0;
            color: #aeb6c4;
            font-size: 1.05rem;
            max-width: 760px;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin: 1.4rem 0 .8rem 0;
        }

        .result-card {
            padding: 1.8rem 2rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.09);
            background: linear-gradient(135deg, #151820 0%, #1b202a 100%);
            margin-top: 1rem;
        }

        .result-label {
            color: #aeb6c4;
            font-size: .92rem;
            margin-bottom: .35rem;
        }

        .result-value {
            font-size: 2.35rem;
            font-weight: 750;
            letter-spacing: -.035em;
        }

        .range-value {
            font-size: 1.45rem;
            font-weight: 650;
            margin-top: .35rem;
        }

        .stat-card {
            padding: 1rem 1.2rem;
            border-radius: 13px;
            border: 1px solid rgba(255,255,255,.08);
            background: rgba(255,255,255,.025);
            min-height: 88px;
        }

        .stat-label {
            color: #8f98a8;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .06em;
        }

        .stat-value {
            font-size: 1.05rem;
            font-weight: 650;
            margin-top: .35rem;
        }

        .footer {
            text-align: center;
            color: #747d8c;
            font-size: .82rem;
            margin-top: 2.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(255,255,255,.07);
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 16px;
            padding: 1.2rem 1.4rem .5rem 1.4rem;
            background: rgba(255,255,255,.015);
        }

        .stButton > button {
            border-radius: 10px;
            min-height: 3rem;
            font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CAT_COLS = [
    "Agency",
    "Posting Type",
    "Job Category",
    "Full-Time/Part-Time indicator",
    "Career Level",
    "Level",
    "Title Classification",
    "Civil Service Title",
]


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "salary_from_xgb.pkl", "rb") as f:
        model_from = pickle.load(f)

    with open(MODEL_DIR / "salary_to_xgb.pkl", "rb") as f:
        model_to = pickle.load(f)

    with open(BASE_DIR / "target_encoding.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model_from, model_to, encoder


@st.cache_data
def load_options():
    data_path = BASE_DIR / "data" / "processed" / "jobs_cleaned.csv"
    df = pd.read_csv(data_path)

    return {
        col: (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        for col in CAT_COLS
    }


try:
    model_from, model_to, encoder = load_artifacts()
    options = load_options()
except Exception as exc:
    st.error("The application could not load the required model files.")
    st.code(str(exc))
    st.stop()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def word_count(text):
    return len(str(text).split()) if str(text).strip() else 0


def target_encode(value, column):
    mapping = encoder["encodings"].get(column, {})
    return float(mapping.get(str(value), encoder["global_mean"]))


def build_features(values):
    features = {
        "# Of Positions": values["positions"],
        "desc_word_count": word_count(values["description"]),
        "qual_word_count": word_count(values["qualifications"]),
        "has_preferred_skills": int(bool(values["preferred_skills"].strip())),
        "posting_month": values["posting_month"],
    }

    for col in CAT_COLS:
        features[col + "_te"] = target_encode(values[col], col)

    feature_order = [
        "# Of Positions",
        "desc_word_count",
        "qual_word_count",
        "has_preferred_skills",
        "posting_month",
        "Agency_te",
        "Posting Type_te",
        "Job Category_te",
        "Full-Time/Part-Time indicator_te",
        "Career Level_te",
        "Level_te",
        "Title Classification_te",
        "Civil Service Title_te",
    ]

    return pd.DataFrame([[features[c] for c in feature_order]], columns=feature_order)


def money(value):
    return f"${value:,.0f}"


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💼 NYC Salary Range Predictor</h1>
        <p>
            Estimate the annual salary range of a New York City job posting
            using an XGBoost machine learning model trained on NYC job data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.markdown('<div class="section-title">Job Information</div>', unsafe_allow_html=True)

with st.form("salary_prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        agency = st.selectbox("Agency", options["Agency"])
        posting_type = st.selectbox("Posting Type", options["Posting Type"])
        job_category = st.selectbox("Job Category", options["Job Category"])
        ft_pt = st.selectbox(
            "Full-Time / Part-Time",
            options["Full-Time/Part-Time indicator"],
        )

    with c2:
        career_level = st.selectbox("Career Level", options["Career Level"])
        level = st.selectbox("Level", options["Level"])
        title_classification = st.selectbox(
            "Title Classification",
            options["Title Classification"],
        )
        civil_service_title = st.selectbox(
            "Civil Service Title",
            options["Civil Service Title"],
        )

    c3, c4 = st.columns(2)

    with c3:
        positions = st.number_input(
            "Number of Positions",
            min_value=1,
            value=1,
            step=1,
        )

    with c4:
        posting_month = st.selectbox(
            "Posting Month",
            list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2026, m, 1).strftime("%B"),
        )

    st.markdown('<div class="section-title">Job Content</div>', unsafe_allow_html=True)

    description = st.text_area(
        "Job Description",
        height=150,
        placeholder="Paste the job description here...",
    )

    qualifications = st.text_area(
        "Minimum Qualifications",
        height=120,
        placeholder="Paste the minimum qualifications here...",
    )

    preferred_skills = st.text_area(
        "Preferred Skills",
        height=100,
        placeholder="Paste preferred skills, tools, technologies, or experience...",
    )

    submitted = st.form_submit_button(
        "Predict Salary Range",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    values = {
        "Agency": agency,
        "Posting Type": posting_type,
        "Job Category": job_category,
        "Full-Time/Part-Time indicator": ft_pt,
        "Career Level": career_level,
        "Level": level,
        "Title Classification": title_classification,
        "Civil Service Title": civil_service_title,
        "positions": positions,
        "posting_month": posting_month,
        "description": description,
        "qualifications": qualifications,
        "preferred_skills": preferred_skills,
    }

    X_input = build_features(values)

    predicted_from = float(model_from.predict(X_input)[0])
    predicted_to = float(model_to.predict(X_input)[0])

    low = min(predicted_from, predicted_to)
    high = max(predicted_from, predicted_to)
    midpoint = (low + high) / 2

    st.markdown("---")
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated annual salary range</div>
            <div class="result-value">{money(low)} – {money(high)}</div>
            <div class="range-value">Estimated midpoint: {money(midpoint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Minimum</div>
                <div class="stat-value">{money(low)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Maximum</div>
                <div class="stat-value">{money(high)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Model</div>
                <div class="stat-value">XGBoost</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Technical details"):
        st.write(
            "The application applies the same feature engineering and "
            "smoothed target-encoding approach used by the project model."
        )

        st.markdown("**Engineered input features**")
        st.dataframe(
            X_input.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

        st.caption(
            "The prediction is a model estimate and should not be treated as an official salary offer."
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        NYC Salary Range Prediction · Machine Learning Project · XGBoost
    </div>
    """,
    unsafe_allow_html=True,
)
