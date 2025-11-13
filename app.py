# app.py
import streamlit as st
from src.clean import load_and_clean
from src.recommender import CosmeticsRecommender

st.set_page_config(page_title="Makeup Recommender", page_icon="💄")

st.title("💄 Makeup Recommender")

df = load_and_clean()
rec = CosmeticsRecommender(df)

product = st.text_input("Enter product name (e.g., lipstick, foundation)")

if product:
    st.write(rec.recommend(product))
