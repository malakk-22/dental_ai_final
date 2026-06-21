import streamlit as st

st.title("Dental AI")

text = st.text_input("Enter text")

if text:
    st.success("Dental AI is running ✅")
