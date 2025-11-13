# analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Optional wordcloud import
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Page configuration
st.set_page_config(
    page_title="Cosmetics Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Title
st.title("📊 Cosmetics Data Analytics Dashboard")
st.markdown("Comprehensive analysis of cosmetics products dataset")

# Load data
@st.cache_data
def load_data():
    """Load and prepare the cosmetics dataset."""
    import os
    # Check both possible locations
    if os.path.exists("cosmetics.csv"):
        csv_path = "cosmetics.csv"
    elif os.path.exists("data/cosmetics.csv"):
        csv_path = "data/cosmetics.csv"
    else:
        raise FileNotFoundError("cosmetics.csv not found. Please ensure the file is in the root directory or data/ folder.")
    
    df = pd.read_csv(csv_path)
    
    # Clean data
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df['Ingredients'] = df['Ingredients'].fillna('')
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filters")
selected_label = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df['Label'].unique()),
    default=sorted(df['Label'].unique())
)

selected_brands = st.sidebar.multiselect(
    "Brands",
    options=sorted(df['Brand'].unique()),
    default=[]
)

price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=float(df['Price'].min()) if not df['Price'].isna().all() else 0.0,
    max_value=float(df['Price'].max()) if not df['Price'].isna().all() else 1000.0,
    value=(float(df['Price'].min()) if not df['Price'].isna().all() else 0.0, 
           float(df['Price'].max()) if not df['Price'].isna().all() else 1000.0)
)

# Apply filters
filtered_df = df[df['Label'].isin(selected_label)]
if selected_brands:
    filtered_df = filtered_df[filtered_df['Brand'].isin(selected_brands)]
filtered_df = filtered_df[
    (filtered_df['Price'] >= price_range[0]) & 
    (filtered_df['Price'] <= price_range[1])
]

# Overview Section
st.header("📈 Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Products", f"{len(filtered_df):,}")
with col2:
    st.metric("Unique Brands", f"{filtered_df['Brand'].nunique():,}")
with col3:
    avg_price = filtered_df['Price'].mean()
    st.metric("Average Price", f"${avg_price:.2f}" if not pd.isna(avg_price) else "N/A")
with col4:
    avg_rank = filtered_df['Rank'].mean()
    st.metric("Average Rating", f"{avg_rank:.2f}" if not pd.isna(avg_rank) else "N/A")

st.markdown("---")

# Price Analysis
st.header("💰 Price Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    price_data = filtered_df['Price'].dropna()
    if len(price_data) > 0:
        ax.hist(price_data, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Product Prices')
        ax.axvline(price_data.mean(), color='red', linestyle='--', label=f'Mean: ${price_data.mean():.2f}')
        ax.axvline(price_data.median(), color='green', linestyle='--', label=f'Median: ${price_data.median():.2f}')
        ax.legend()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Price by Category")
    fig, ax = plt.subplots(figsize=(10, 6))
    price_by_label = filtered_df.groupby('Label')['Price'].mean().sort_values(ascending=False)
    if len(price_by_label) > 0:
        ax.barh(range(len(price_by_label)), price_by_label.values, color='coral')
        ax.set_yticks(range(len(price_by_label)))
        ax.set_yticklabels(price_by_label.index)
        ax.set_xlabel('Average Price ($)')
        ax.set_title('Average Price by Product Category')
        ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

# Top Brands by Price
st.subheader("Top 10 Brands by Average Price")
top_brands_price = filtered_df.groupby('Brand')['Price'].agg(['mean', 'count']).reset_index()
top_brands_price = top_brands_price[top_brands_price['count'] >= 3].sort_values('mean', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
if len(top_brands_price) > 0:
    ax.barh(range(len(top_brands_price)), top_brands_price['mean'].values, color='mediumpurple')
    ax.set_yticks(range(len(top_brands_price)))
    ax.set_yticklabels(top_brands_price['Brand'].values)
    ax.set_xlabel('Average Price ($)')
    ax.set_title('Top 10 Brands by Average Price (min 3 products)')
    ax.invert_yaxis()
    for i, v in enumerate(top_brands_price['mean'].values):
        ax.text(v + 2, i, f'${v:.2f}', va='center')
st.pyplot(fig)
plt.close()

st.markdown("---")

# Brand Analysis
st.header("🏢 Brand Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Brands by Product Count")
    top_brands = filtered_df['Brand'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_brands)), top_brands.values, color='lightseagreen')
    ax.set_yticks(range(len(top_brands)))
    ax.set_yticklabels(top_brands.index)
    ax.set_xlabel('Number of Products')
    ax.set_title('Top 10 Brands by Product Count')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Brand Distribution")
    brand_counts = filtered_df['Brand'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    # Show distribution of brand sizes
    brand_size_dist = brand_counts.value_counts().sort_index()
    ax.bar(brand_size_dist.index, brand_size_dist.values, color='gold')
    ax.set_xlabel('Number of Products per Brand')
    ax.set_ylabel('Number of Brands')
    ax.set_title('Distribution of Brand Sizes')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Category Analysis
st.header("📦 Category Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Products by Category")
    label_counts = filtered_df['Label'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(label_counts)), label_counts.values, color='salmon')
    ax.set_yticks(range(len(label_counts)))
    ax.set_yticklabels(label_counts.index)
    ax.set_xlabel('Number of Products')
    ax.set_title('Product Count by Category')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Average Rating by Category")
    avg_rank_by_label = filtered_df.groupby('Label')['Rank'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(avg_rank_by_label)), avg_rank_by_label.values, color='plum')
    ax.set_yticks(range(len(avg_rank_by_label)))
    ax.set_yticklabels(avg_rank_by_label.index)
    ax.set_xlabel('Average Rating')
    ax.set_title('Average Rating by Category')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Rating Analysis
st.header("⭐ Rating Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rating Distribution")
    rank_data = filtered_df['Rank'].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(rank_data) > 0:
        ax.hist(rank_data, bins=20, edgecolor='black', alpha=0.7, color='lightblue')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Product Ratings')
        ax.axvline(rank_data.mean(), color='red', linestyle='--', label=f'Mean: {rank_data.mean():.2f}')
        ax.legend()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Price vs Rating")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_data = filtered_df[['Price', 'Rank']].dropna()
    if len(scatter_data) > 0:
        ax.scatter(scatter_data['Price'], scatter_data['Rank'], alpha=0.5, color='crimson')
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Rating')
        ax.set_title('Price vs Rating Relationship')
        # Add trend line
        z = np.polyfit(scatter_data['Price'], scatter_data['Rank'], 1)
        p = np.poly1d(z)
        ax.plot(scatter_data['Price'], p(scatter_data['Price']), "r--", alpha=0.8, label='Trend Line')
        ax.legend()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Skin Type Compatibility Analysis
st.header("🧴 Skin Type Compatibility Analysis")

skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
skin_compatibility = {}

for skin_type in skin_types:
    compatible = filtered_df[filtered_df[skin_type] == 1]
    skin_compatibility[skin_type] = {
        'count': len(compatible),
        'percentage': len(compatible) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0,
        'avg_price': compatible['Price'].mean() if len(compatible) > 0 else 0,
        'avg_rating': compatible['Rank'].mean() if len(compatible) > 0 else 0
    }

col1, col2 = st.columns(2)

with col1:
    st.subheader("Products Compatible by Skin Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    skin_counts = [skin_compatibility[st]['count'] for st in skin_types]
    ax.bar(skin_types, skin_counts, color='lightcoral')
    ax.set_ylabel('Number of Products')
    ax.set_title('Products Compatible with Each Skin Type')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(skin_counts):
        ax.text(i, v + 5, str(v), ha='center', va='bottom')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Average Price by Skin Type Compatibility")
    fig, ax = plt.subplots(figsize=(10, 6))
    skin_prices = [skin_compatibility[st]['avg_price'] for st in skin_types]
    ax.bar(skin_types, skin_prices, color='mediumaquamarine')
    ax.set_ylabel('Average Price ($)')
    ax.set_title('Average Price for Skin Type Compatible Products')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(skin_prices):
        if not pd.isna(v):
            ax.text(i, v + 2, f'${v:.2f}', ha='center', va='bottom')
    st.pyplot(fig)
    plt.close()

# Skin type compatibility table
st.subheader("Skin Type Compatibility Summary")
skin_df = pd.DataFrame({
    'Skin Type': skin_types,
    'Compatible Products': [skin_compatibility[st]['count'] for st in skin_types],
    'Percentage': [f"{skin_compatibility[st]['percentage']:.1f}%" for st in skin_types],
    'Avg Price': [f"${skin_compatibility[st]['avg_price']:.2f}" if not pd.isna(skin_compatibility[st]['avg_price']) else "N/A" for st in skin_types],
    'Avg Rating': [f"{skin_compatibility[st]['avg_rating']:.2f}" if not pd.isna(skin_compatibility[st]['avg_rating']) else "N/A" for st in skin_types]
})
st.dataframe(skin_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Ingredient Analysis
st.header("🧪 Ingredient Analysis")

# Extract ingredients
def extract_ingredients(ingredient_text):
    """Extract individual ingredients from text."""
    if pd.isna(ingredient_text) or ingredient_text == '':
        return []
    # Split by comma and clean
    ingredients = [ing.strip() for ing in str(ingredient_text).split(',')]
    # Remove empty strings and very short ingredients
    ingredients = [ing for ing in ingredients if len(ing) > 2]
    return ingredients

all_ingredients = []
for ingredients_text in filtered_df['Ingredients']:
    all_ingredients.extend(extract_ingredients(ingredients_text))

ingredient_counts = Counter(all_ingredients)
top_ingredients = dict(ingredient_counts.most_common(20))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 20 Most Common Ingredients")
    fig, ax = plt.subplots(figsize=(10, 8))
    if top_ingredients:
        ingredients_list = list(top_ingredients.keys())
        counts_list = list(top_ingredients.values())
        ax.barh(range(len(ingredients_list)), counts_list, color='teal')
        ax.set_yticks(range(len(ingredients_list)))
        ax.set_yticklabels(ingredients_list)
        ax.set_xlabel('Frequency')
        ax.set_title('Top 20 Most Common Ingredients')
        ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Ingredient Word Cloud")
    if all_ingredients:
        if HAS_WORDCLOUD:
            # Create word cloud
            ingredient_text = ' '.join([ing.lower() for ing in all_ingredients])
            try:
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(ingredient_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Most Common Ingredients Word Cloud')
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Could not generate word cloud: {str(e)}")
                st.info("💡 Try installing wordcloud: pip install wordcloud")
        else:
            st.info("📦 Word cloud visualization requires the 'wordcloud' package.")
            st.code("pip install wordcloud", language="bash")
            st.caption("💡 Install wordcloud to see the ingredient word cloud visualization")

# Top ingredients table
st.subheader("Top Ingredients Summary")
top_ing_df = pd.DataFrame({
    'Ingredient': list(top_ingredients.keys())[:15],
    'Frequency': list(top_ingredients.values())[:15],
    'Percentage': [f"{(count/len(filtered_df)*100):.1f}%" for count in list(top_ingredients.values())[:15]]
})
st.dataframe(top_ing_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Additional Insights
st.header("💡 Key Insights")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.subheader("Price Insights")
    price_data = filtered_df['Price'].dropna()
    if len(price_data) > 0:
        max_price_idx = price_data.idxmax()
        min_price_idx = price_data.idxmin()
        st.write(f"• **Most Expensive Product**: {filtered_df.loc[max_price_idx, 'Brand']} - {filtered_df.loc[max_price_idx, 'Name']} (${price_data.max():.2f})")
        st.write(f"• **Most Affordable Product**: {filtered_df.loc[min_price_idx, 'Brand']} - {filtered_df.loc[min_price_idx, 'Name']} (${price_data.min():.2f})")
        st.write(f"• **Price Range**: ${price_data.min():.2f} - ${price_data.max():.2f}")
        st.write(f"• **Median Price**: ${price_data.median():.2f}")
    else:
        st.write("No price data available")

with insights_col2:
    st.subheader("Rating Insights")
    rank_data = filtered_df['Rank'].dropna()
    if len(rank_data) > 0:
        max_rank_idx = rank_data.idxmax()
        min_rank_idx = rank_data.idxmin()
        st.write(f"• **Highest Rated Product**: {filtered_df.loc[max_rank_idx, 'Brand']} - {filtered_df.loc[max_rank_idx, 'Name']} ({rank_data.max():.2f}⭐)")
        st.write(f"• **Lowest Rated Product**: {filtered_df.loc[min_rank_idx, 'Brand']} - {filtered_df.loc[min_rank_idx, 'Name']} ({rank_data.min():.2f}⭐)")
        st.write(f"• **Rating Range**: {rank_data.min():.2f} - {rank_data.max():.2f}")
        st.write(f"• **Median Rating**: {rank_data.median():.2f}")
    else:
        st.write("No rating data available")

# Footer
st.markdown("---")
st.caption("📊 Analytics Dashboard | Use the sidebar filters to explore different segments of the data")

