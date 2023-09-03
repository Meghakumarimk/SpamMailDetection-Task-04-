import pickle
import streamlit as st

model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))


def main():
    st.title("SPAM EMAIL DETECTOR")
    st.subheader("Created by Megha Kumari with Stremlit & Python")
    msg = st.text_input("Enter a Text Message: ")
    if st.button("Check"):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result ==1:
            st.error("Warning! Spam mail detected")
        else:
            st.success("This is a Ham Mail")

main()