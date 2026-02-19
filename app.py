import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download once
nltk.download('stopwords')
nltk.download('wordnet')

# ==========================
# LOAD MODEL & VECTORIZER
# ==========================
model = pickle.load(open("Sentiment Analysis.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# ==========================
# CLEAN TEXT FUNCTION
# ==========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)

# ==========================
# STREAMLIT UI
# ==========================
st.title("🍽️ Restaurant Review Sentiment Analysis")

review_input = st.text_input("Enter your review:")

if st.button("Predict Sentiment"):

    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        clean = clean_text(review_input)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😡")
