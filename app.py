import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import os

# Load model and vectorizer
with open("meme_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("😂 Meme Sentiment Classifier")
st.write("Predict whether a meme is **Funny**, **Cringe**, or **Neutral**")

# Text input
caption = st.text_input("Enter meme caption:")

# Image upload
uploaded_image = st.file_uploader("Upload a meme image", type=["jpg", "png", "jpeg"])

if st.button("Predict Sentiment"):
    if caption.strip() == "":
        st.warning("Please enter a caption")
    else:
        # Vectorize text
        caption_vec = vectorizer.transform([caption])
        prediction = model.predict(caption_vec)[0]

        st.success(f"Prediction: **{prediction.upper()}**")

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Meme", use_container_width=True)
