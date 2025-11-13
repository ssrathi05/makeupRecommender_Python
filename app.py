# app.py
import streamlit as st
import pandas as pd
from src.recommender import CosmeticsRecommender

# Page configuration
st.set_page_config(
    page_title="Cosmetics Recommender",
    page_icon="💄",
    layout="wide"
)

# Title
st.title("💄 Cosmetics Recommendation System")
st.markdown("Find similar cosmetics products based on ingredients!")

# Cache the recommender to speed up loading
@st.cache_resource
def load_recommender():
    """Load and cache the recommender system."""
    import os
    # Check both possible locations
    if os.path.exists("cosmetics.csv"):
        csv_path = "cosmetics.csv"
    elif os.path.exists("data/cosmetics.csv"):
        csv_path = "data/cosmetics.csv"
    else:
        raise FileNotFoundError("cosmetics.csv not found. Please ensure the file is in the root directory or data/ folder.")
    return CosmeticsRecommender(csv_path)

# Load recommender
try:
    recommender = load_recommender()
    df = recommender.df
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Product selection dropdown
st.subheader("Select a Product")
product_names = sorted(df['Name'].unique().tolist())
selected_product = st.selectbox(
    "Choose a product:",
    options=product_names,
    index=0,
    help="Select a product to find similar recommendations"
)

# Number of recommendations
top_n = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=20,
    value=5,
    help="Select how many similar products to show"
)

# Get recommendations
if st.button("Get Recommendations", type="primary"):
    if selected_product:
        with st.spinner("Finding similar products..."):
            recommendations = recommender.recommend(selected_product, top_n=top_n)
        
        if recommendations.empty:
            st.warning(f"❌ No recommendations found for '{selected_product}'.")
        else:
            st.subheader(f"✨ Top {len(recommendations)} Similar Products")
            
            # Display as a nice table
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True
            )
            
            # Show selected product info
            selected_info = df[df['Name'].str.lower() == selected_product.lower()].iloc[0]
            with st.expander("📋 Selected Product Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Brand", selected_info['Brand'])
                with col2:
                    st.metric("Price", f"${selected_info['Price']:.2f}" if pd.notna(selected_info['Price']) else "N/A")
                with col3:
                    st.metric("Rank", f"{selected_info['Rank']:.1f}" if pd.notna(selected_info['Rank']) else "N/A")
                
                if pd.notna(selected_info['Ingredients']) and selected_info['Ingredients']:
                    st.text_area("Ingredients", selected_info['Ingredients'], height=100)
    else:
        st.warning("Please select a product first.")

# Footer
st.markdown("---")
st.caption("💡 Tip: Select a product and click 'Get Recommendations' to find similar items based on ingredients!")
