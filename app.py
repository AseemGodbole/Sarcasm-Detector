import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# ---- Page config ----
st.set_page_config(page_title="Sarcasm Detector", page_icon="🎭", layout="centered")

# ---- Load model + tokenizer (cached so it only loads once) ----
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('sarcasm_model.h5')
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('max_seq_length.pickle', 'rb') as f:
        max_seq_length = pickle.load(f)
    return model, tokenizer, max_seq_length

model, tokenizer, max_seq_length = load_artifacts()

# ---- Prediction function (raw text -> class probabilities) ----
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_seq_length, padding='post'
    )
    preds = model.predict(padded, verbose=0)
    return np.hstack([1 - preds, preds])

explainer = LimeTextExplainer(class_names=['No Sarcasm', 'Sarcasm'])

# ---- UI ----
st.title("🎭 Sarcasm Detector")
st.write("Type a news headline below and the model will predict whether it's sarcastic, with word-level explanations.")

example_headlines = [
    "Type your own...",
    "corporation surprised to see its tax money circle back around to it so soon",
    "study finds eating vegetables linked to improved health",
    "man discovers that drinking water cures his thirst",
    "local government announces new park opening next month",
]

choice = st.selectbox("Try an example headline, or type your own:", example_headlines)

if choice == "Type your own...":
    headline = st.text_input("Enter a headline:", "")
else:
    headline = choice
    st.text_input("Enter a headline:", value=headline, disabled=True)

if st.button("Predict", type="primary") and headline.strip():
    with st.spinner("Analyzing..."):
        probs = predict_proba([headline])[0]
        sarcasm_prob = probs[1]

        # ---- Prediction display ----
        col1, col2 = st.columns(2)
        col1.metric("No Sarcasm", f"{probs[0]*100:.1f}%")
        col2.metric("Sarcasm", f"{probs[1]*100:.1f}%")

        if sarcasm_prob >= 0.5:
            st.success(f"Prediction: **Sarcastic** ({sarcasm_prob*100:.1f}% confidence)")
        else:
            st.info(f"Prediction: **Not Sarcastic** ({(1-sarcasm_prob)*100:.1f}% confidence)")

        # ---- LIME explanation ----
        st.subheader("Why did the model predict this?")
        st.write("Word highlights show what pushed the prediction toward Sarcasm (orange) or No Sarcasm (blue).")

        exp = explainer.explain_instance(
            headline, predict_proba, num_features=10
        )

        # Render LIME's HTML explanation directly in the app
        components.html(exp.as_html(), height=400, scrolling=True)

elif headline.strip() == "" and st.session_state.get('clicked', False):
    st.warning("Please enter a headline.")

st.markdown("---")
st.caption("Built with a Keras Embedding + Dense neural network trained from scratch on 26k news headlines. "
            "Explanations powered by LIME (Local Interpretable Model-agnostic Explanations).")
