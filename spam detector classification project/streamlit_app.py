import streamlit as st
import joblib

# Load model and vectorizer
clf = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
labels = {0: "Ham", 1: "Spam"}

# App title
st.title("📩 Spam Detector")
st.write("Type a message and see if it's spam or not.")

# Message input with a sample
msg = st.text_area("Enter your message:")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # Transform and predict
        new_X = vectorizer.transform([msg])
        pred = clf.predict(new_X)[0]
        label = labels[pred]

        # Show result
        if label == "Spam":
            st.error(f"🚨 This message looks like *{label}*!")
        else:
            st.success(f"✅ This message looks safe ({label}).")
            st.balloons()

# Sidebar - Extra info
st.sidebar.title("ℹ️ About")
st.sidebar.write("This app uses a **Multinomial Naive Bayes** model to detect spam messages.")
st.sidebar.write("Model trained with scikit-learn")
