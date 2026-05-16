import streamlit as st
import pickle

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

model = pickle.load(
    open('sentiment_pipeline.pkl', 'rb')
)

with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )
    
st.markdown(
    "<h1 class='title'>🧠 Twitter Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class='subtitle'>Analyze emotions from tweets using Machine Learning</p>",
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Enter Tweet",
    height=180,
    placeholder="Type something here..."
)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])
        if prediction[0] == 1:
            st.markdown(
                """
                <div class='positive-box'>
                    😊 Positive Sentiment
                </div>
                """,
                unsafe_allow_html=True
            )
        else:

            st.markdown(
                """
                <div class='negative-box'>
                    😠 Negative Sentiment
                </div>
                """,
                unsafe_allow_html=True
            )
