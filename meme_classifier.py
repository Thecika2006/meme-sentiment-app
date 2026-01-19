# meme_classifier.py

import pandas as pd
import string
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset


#data = pd.read_csv("meme_dataset/meme_data.csv")
data = pd.read_csv(r"V:\Meme Sentiment\meme_dataset\meme_data.csv")



print("Dataset Loaded Successfully!\n")
print(data.head())

# -----------------------------
# 2. Text Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

data["clean_caption"] = data["caption"].apply(clean_text)

# -----------------------------
# 3. Convert Text to Numbers (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["clean_caption"])
y = data["label"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 7. Predict on New Memes
# -----------------------------
new_memes = [
    ("meme2.jpg", "Ugh not again"),
    ("meme4.jpg", "This is too good"),
    ("meme3.jpg", "Just another day")
]

predictions = []

for image_name, caption in new_memes:
    cleaned = clean_text(caption)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    predictions.append([image_name, caption, prediction])

    print("\n----------------------------")
    print("Meme:", image_name)
    print("Caption:", caption)
    print("Prediction:", prediction)

    # BONUS: Show Image
    try:
        img = Image.open(f"meme_dataset/{image_name}")
        plt.imshow(img)
        plt.title(f"Prediction: {prediction}")
        plt.axis("off")
        plt.show()
    except:
        print("Image not found!")

# -----------------------------
# 8. Save Predictions to CSV
# -----------------------------
pred_df = pd.DataFrame(
    predictions,
    columns=["image", "caption", "predicted_label"]
)

pred_df.to_csv("meme_predictions.csv", index=False)
print("\nPredictions saved to meme_predictions.csv")


import pickle

# Save trained model
with open("meme_sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")

