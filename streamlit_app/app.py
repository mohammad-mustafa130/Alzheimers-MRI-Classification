
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Alzheimer MRI AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "best_custom_cnn_v4.keras"
)

CLASS_NAMES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented"
]

LABELS = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

st.markdown("""
<style>
.stApp {
    background: #F4F8FC;
    color: #10294B;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
}
html, body, p, span, label, li {
    color: #193B5A;
}
h1, h2, h3, h4 {
    color: #10294B !important;
}
[data-testid="stSidebar"] {
    background: #E9F3FC;
    border-right: 1px solid #C9DDEC;
}
[data-testid="stSidebar"] * {
    color: #123B65;
}
[data-testid="stSidebar"] .stRadio label {
    font-weight: 650;
}
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #8FB9D9;
    border-radius: 14px;
    padding: 12px;
}
[data-testid="stFileUploader"] * {
    color: #193B5A !important;
}
[data-testid="stFileUploader"] button {
    color: #10294B !important;
    background: #EAF3FB !important;
}
[data-testid="stDataFrame"] {
    background: white;
}
div.stButton > button[kind="primary"] {
    background: #1262BD;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 750;
    min-height: 3rem;
}
div.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #0B4C99;
}
.hero {
    background: linear-gradient(110deg, #FFFFFF, #E8F5FF);
    border: 1px solid #D1E3F2;
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 22px;
}
.hero h1 {
    font-size: clamp(1.5rem, 2.6vw, 2.2rem);
    margin: 0 0 8px;
}
.hero p {
    color: #47627F !important;
    margin: 0;
}
.panel {
    background: #FFFFFF;
    border: 1px solid #D6E4F0;
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}
.result {
    background: #E5FAF5;
    border: 1px solid #A7E2D8;
    border-left: 5px solid #009D99;
    border-radius: 15px;
    padding: 22px;
    margin: 12px 0;
}
.result-title {
    color: #087C78 !important;
    font-weight: 800;
    font-size: 1.7rem;
}
.result-label {
    color: #315C59 !important;
    font-weight: 650;
}
.metric-box {
    background: #FFFFFF;
    border: 1px solid #D6E4F0;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 10px;
}
.metric-box .label {
    color: #47627F !important;
    font-weight: 650;
}
.metric-box .value {
    color: #1262BD !important;
    font-size: 1.8rem;
    font-weight: 800;
}
.disclaimer {
    background: #FFF4EC;
    border: 1px solid #EECBB4;
    border-left: 5px solid #C96C35;
    border-radius: 14px;
    padding: 18px;
    color: #713D21 !important;
    line-height: 1.6;
    margin-top: 25px;
}
.disclaimer * {
    color: #713D21 !important;
}
[data-testid="stAlert"] p {
    color: inherit;
}
</style>
""", unsafe_allow_html=True)


st.markdown('''
<style>
/* Overall readability */
.stApp {
    background: #F4F8FC !important;
    color: #10294B !important;
}

[data-testid="stAppViewContainer"] {
    background: #F4F8FC !important;
}

[data-testid="stMainBlockContainer"] {
    color: #10294B !important;
}

.stMarkdown p,
.stMarkdown li,
.stMarkdown span,
.stCaption,
.stText,
.stWidgetLabel,
label {
    color: #193B5A !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #10294B !important;
}

/* Top navigation */
div[role="radiogroup"] {
    gap: 8px;
}

.top-nav-heading {
    background: #FFFFFF;
    border: 1px solid #D6E4F0;
    border-radius: 14px;
    padding: 12px 18px;
    color: #123B65 !important;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Upload widget */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #86B5D9 !important;
    border-radius: 14px;
}

[data-testid="stFileUploader"] section {
    background: #F8FBFF !important;
    color: #10294B !important;
}

[data-testid="stFileUploader"] section * {
    color: #10294B !important;
}

[data-testid="stFileUploader"] button {
    background: #E6F1FC !important;
    color: #123B65 !important;
    border: 1px solid #A8C9E8 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #E9F3FC !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #123B65 !important;
}

/* Buttons */
div.stButton > button[kind="primary"] {
    background: #1262BD !important;
    color: #FFFFFF !important;
}

div.stButton > button[kind="primary"] * {
    color: #FFFFFF !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D6E4F0;
    padding: 15px;
    border-radius: 12px;
}
</style>
''', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

def preprocess(image):
    image = image.convert("L")
    image = image.resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, 128, 128, 1)

def probability_chart(probabilities):
    values = [float(v) * 100 for v in probabilities]
    colors = ["#2F83D6", "#13A8A2", "#8A68D8", "#E19B62"]
    colors[int(np.argmax(probabilities))] = "#008F88"

    fig = go.Figure(go.Bar(
        x=values,
        y=LABELS,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        cliponaxis=False
    ))
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=60, t=15, b=35),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#10294B", size=14),
        xaxis_tickfont=dict(color="#10294B", size=12),
        yaxis_tickfont=dict(color="#10294B", size=13),
        xaxis=dict(
            range=[0, 110],
            ticksuffix="%",
            gridcolor="#E2ECF5"
        ),
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    return fig

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("## 🧠 MRI AI Dashboard")
    st.caption("Alzheimer's MRI Research Prototype")
    st.divider()

    sidebar_page = st.radio(
        "Navigation",
        [
            "🏠 MRI Analysis",
            "📊 Model Performance",
            "🧬 About Project",
            "📖 How to Use"
        ],
        label_visibility="collapsed",
        key="sidebar_navigation"
    )

    st.divider()
    st.markdown("**Model:** Custom CNN V4")
    st.markdown("**Input:** 128 × 128 grayscale")
    st.markdown("**Classes:** 4")
    st.caption("Academic research use only")

st.markdown("""
<div class="hero">
    <h1>🧠 Alzheimer's MRI Classification</h1>
    <p>AI-assisted brain MRI research dashboard |
       Custom CNN Version 4</p>
</div>
""", unsafe_allow_html=True)


page = sidebar_page

if page == "🏠 MRI Analysis":
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.subheader("📤 Upload Brain MRI")
        st.write(
            "Select a PNG or JPEG MRI image for research-only analysis."
        )

        uploaded = st.file_uploader(
            "Choose MRI image",
            type=["png", "jpg", "jpeg"]
        )

        image = None
        if uploaded is not None:
            try:
                image = Image.open(uploaded)
                image.load()
                st.image(
                    image,
                    caption="Uploaded MRI preview",
                    use_container_width=True
                )
            except Exception:
                st.error("Please upload a valid image.")

        analyze = st.button(
            "🧠 Analyze MRI",
            type="primary",
            use_container_width=True,
            disabled=image is None
        )

    with right:
        st.subheader("📋 Prediction Results")

        if analyze and image is not None:
            try:
                with st.spinner("Analyzing MRI..."):
                    model = load_model()
                    probabilities = np.asarray(
                        model.predict(
                            preprocess(image), verbose=0
                        )[0],
                        dtype=float
                    )

                if (
                    probabilities.shape != (4,)
                    or not np.all(np.isfinite(probabilities))
                ):
                    raise ValueError("Invalid model output")

                predicted = int(np.argmax(probabilities))

                st.markdown(
                    '<div class="result">'
                    '<div class="result-label">'
                    'Predicted Dataset Category</div>'
                    f'<div class="result-title">'
                    f'{LABELS[predicted]}</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.subheader("Class Probabilities")
                st.plotly_chart(
                    probability_chart(probabilities),
                    use_container_width=True
                )

                table = pd.DataFrame({
                    "Category": LABELS,
                    "Model Probability": probabilities
                })
                st.dataframe(
                    table.style.format({
                        "Model Probability": "{:.2%}"
                    }).set_properties(**{
                        "background-color": "#FFFFFF",
                        "color": "#10294B"
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                st.caption(
                    "Model probabilities are not clinically "
                    "validated confidence estimates."
                )

            except Exception as error:
                st.error("Unable to complete prediction.")
                with st.expander("Technical details"):
                    st.code(str(error))
        else:
            st.info(
                "Upload an MRI image and click Analyze MRI "
                "to view the prediction."
            )

    st.divider()
    st.subheader("📈 CNN V4 Test Performance")

    cols = st.columns(4)
    metrics = [
        ("Test Accuracy", "62.40%"),
        ("Macro F1", "0.43"),
        ("Macro Recall", "0.44"),
        ("Macro Precision", "0.43")
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                '<div class="metric-box">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                '</div>',
                unsafe_allow_html=True
            )

elif page == "📊 Model Performance":
    st.subheader("CNN Experiment Comparison")
    comparison = pd.DataFrame({
        "Model": ["CNN V2", "CNN V3", "CNN V4"],
        "Validation Accuracy (%)": [53.44, 60.62, 64.69],
        "Validation Macro F1": [0.27, 0.37, 0.46]
    })
    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )
    st.bar_chart(
        comparison.set_index("Model")[
            "Validation Accuracy (%)"
        ]
    )
    st.warning(
        "CNN V4 Moderate Demented recall was 0% "
        "on the held-out test set."
    )

elif page == "🧬 About Project":
    st.subheader("About This Project")
    st.write(
        "Final-year B.Tech CSE academic project using "
        "a custom CNN for four-category MRI classification."
    )
    st.markdown("""
    - **Dataset:** 6,400 preprocessed MRI images
    - **Model:** Custom CNN Version 4
    - **Input:** 128 × 128 grayscale
    - **Split:** 70% train, 15% validation, 15% test
    - **Test accuracy:** 62.40%
    - **Test macro F1-score:** 0.43
    """)
    st.info(
        "Patient-independent splitting could not be verified "
        "because patient identifiers were unavailable."
    )

elif page == "📖 How to Use":
    st.subheader("Using the MRI Dashboard")
    st.markdown("""
    1. Select **MRI Analysis** in the sidebar.
    2. Upload a PNG or JPEG MRI image.
    3. Review the preview.
    4. Click **Analyze MRI**.
    5. Read the predicted category and probability chart.
    """)
    st.warning(
        "Do not upload identifiable patient information."
    )

st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Medical Disclaimer</strong><br>
    This application is an academic research prototype,
    not a clinically validated medical device.
    It must not be used for diagnosis, screening,
    or treatment decisions. Consult a qualified
    healthcare professional for medical advice.
</div>
""", unsafe_allow_html=True)

st.caption(
    "Alzheimer MRI Classification | B.Tech CSE Project"
)
