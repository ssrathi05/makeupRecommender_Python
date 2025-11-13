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

# Product search interface
st.subheader("Search for a Product")
st.markdown("💡 **Tip:** Type keywords like 'wrinkle cream', 'estee lauder', 'moisturizer', etc.")

# Search input
search_query = st.text_input(
    "Search by product name or brand:",
    placeholder="e.g., 'wrinkle cream', 'estee lauder', 'moisturizer'",
    help="Type any part of the product name or brand to find matches"
)

# Create a searchable product list
def search_products(df, query):
    """Search products by brand or name."""
    if not query or query.strip() == '':
        return pd.DataFrame()
    
    query_lower = query.lower().strip()
    # Search in both Brand and Name columns
    mask = (
        df['Brand'].str.lower().str.contains(query_lower, na=False) |
        df['Name'].str.lower().str.contains(query_lower, na=False)
    )
    results = df[mask].copy()
    
    # Create a display format: "Brand - Name"
    results['display'] = results['Brand'] + ' - ' + results['Name']
    
    return results[['Brand', 'Name', 'display', 'Price']].head(20)  # Limit to 20 results

# Show search results
selected_product = None
search_results = pd.DataFrame()

if search_query:
    search_results = search_products(df, search_query)
    
    if not search_results.empty:
        st.markdown(f"**Found {len(search_results)} matching product(s):**")
        
        # Create a selectbox with filtered results
        display_options = search_results['display'].tolist()
        selected_display = st.selectbox(
            "Select a product:",
            options=display_options,
            index=0,
            help="Choose from the filtered results"
        )
        
        # Extract the actual product name from selection
        if selected_display:
            selected_row = search_results[search_results['display'] == selected_display].iloc[0]
            selected_product = selected_row['Name']
            
            # Show selected product preview
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"**Selected:** {selected_row['Brand']} - {selected_row['Name']}")
            with col2:
                if pd.notna(selected_row['Price']):
                    st.metric("Price", f"${selected_row['Price']:.2f}")
    else:
        st.warning(f"❌ No products found matching '{search_query}'. Try different keywords!")
        st.caption("💡 Try searching for: brand names, product types (moisturizer, serum, cream), or specific features")
else:
    st.info("👆 Start typing to search for products...")

# Number of recommendations
top_n = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=20,
    value=5,
    help="Select how many similar products to show"
)

# Get recommendations
if selected_product:
    st.markdown("---")
    if st.button("Get Recommendations", type="primary", use_container_width=True):
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
elif search_query and not search_results.empty:
    st.markdown("---")
    st.info("👆 Select a product above and click 'Get Recommendations' to see similar items!")

# Footer
st.markdown("---")
st.caption("💡 Tip: Select a product and click 'Get Recommendations' to find similar items based on ingredients!")
