import streamlit as st
import joblib
import re
import string


# Load the trained model and TF-IDF vectorizer
model = joblib.load("spam_svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Same preprocessing used during training
def clean_text(text):
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# Page settings
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧"
)


# Title
st.title("📧 Spam Email Detector")

st.write(
    "Enter an email below and the ML model "
    "will classify it as Spam or Not Spam."
)


# Email input
email = st.text_area(
    "Enter your email:",
    height=250,
    placeholder="Paste your email here..."
)


# Check button
if st.button("🔍 Check Email"):

    if email.strip() == "":
        st.warning("Please enter an email.")

    else:

        # Clean email
        cleaned_email = clean_text(email)

        # Convert text to TF-IDF
        email_tfidf = vectorizer.transform(
            [cleaned_email]
        )

        # Predict using SVM
        prediction = model.predict(email_tfidf)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 SPAM EMAIL")
            st.write(
                "The model classified this email as spam."
            )

        else:
            st.success("✅ NOT SPAM")
            st.write(
                "The model classified this email as legitimate."
            )


# Model information
st.divider()

st.subheader("Model Information")

st.write("**Algorithm:** Linear SVM")
st.write("**Feature Extraction:** TF-IDF")
st.write("**Test Accuracy:** 98.79%")