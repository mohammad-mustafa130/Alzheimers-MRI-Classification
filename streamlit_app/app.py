
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

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

st.set_page_config(
    page_title="Alzheimer MRI Classification",
    page_icon="🧠"
)

st.title("🧠 Alzheimer's MRI Classification")

st.write("Brain MRI classification using a custom CNN.")

st.warning(
    "Academic research prototype only. "
    "Not intended for medical diagnosis."
)

st.info(
    "Held-out test accuracy: 62.40%. "
    "Macro F1-score: 0.43. "
    "Moderate_Demented recall: 0%."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded MRI image",
            width=300
        )

        image = image.convert("L")
        image = image.resize((128, 128))

        image_array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        image_array = image_array.reshape(
            1, 128, 128, 1
        )

        if st.button("Predict MRI Category"):
            with st.spinner("Predicting..."):
                model = load_model()
                probabilities = model.predict(
                    image_array, verbose=0
                )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            st.subheader("Prediction Result")

            st.write(
                "Predicted dataset category:",
                CLASS_NAMES[predicted_index]
            )

            results = pd.DataFrame({
                "Category": CLASS_NAMES,
                "Model probability": probabilities
            })

            st.bar_chart(
                results.set_index("Category")
            )

            st.dataframe(
                results.style.format({
                    "Model probability": "{:.2%}"
                }),
                hide_index=True
            )

            st.caption(
                "Model probabilities are not clinically "
                "validated confidence estimates."
            )

    except Exception as error:
        st.error(
            "Unable to process image or load model."
        )
        st.code(str(error))

st.caption(
    "Final-year B.Tech CSE academic project."
)
