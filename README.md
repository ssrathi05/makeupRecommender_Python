# 💄 Cosmetics Recommendation & Analytics System

A comprehensive Python project featuring both a **product recommendation system** and a **data analytics dashboard** for cosmetics products. Built using Streamlit with TF-IDF-based similarity matching and interactive visualizations.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Dataset](#dataset)

## ✨ Features

### 🔍 Recommendation System (`app.py`)
- **Smart Product Search**: Search products by keywords (e.g., "wrinkle cream", "estee lauder")
- **TF-IDF Similarity Matching**: Find similar products based on ingredient analysis
- **User-Friendly Interface**: Easy-to-use search and dropdown selection
- **Product Details**: View brand, price, and ingredient information
- **Customizable Recommendations**: Adjust the number of recommendations (1-20)

### 📊 Analytics Dashboard (`analytics.py`)
- **Comprehensive Data Analysis**: Explore the entire cosmetics dataset
- **Price Analysis**: Distribution, category comparison, top brands
- **Brand Analysis**: Product counts, brand distribution
- **Category Analysis**: Product distribution and ratings by category
- **Rating Analysis**: Rating distribution and price vs rating correlation
- **Skin Type Compatibility**: Analysis for Combination, Dry, Normal, Oily, and Sensitive skin
- **Ingredient Analysis**: Top ingredients, word cloud visualization
- **Interactive Filters**: Filter by category, brand, and price range
- **Key Insights**: Summary statistics and notable findings

## 📁 Project Structure

```
makeupRecommender_Python/
│
├── app.py                 # Recommendation system (Streamlit app)
├── analytics.py           # Data analytics dashboard (Streamlit app)
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── data/
│   └── cosmetics.csv     # Cosmetics dataset
│
└── src/
    ├── utils.py          # Text cleaning and TF-IDF utilities
    └── recommender.py    # CosmeticsRecommender class
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Recommendation System

Run the product recommendation app:

```bash
streamlit run app.py
```

**How to use:**
1. Type keywords in the search box (e.g., "moisturizer", "estee lauder", "wrinkle cream")
2. Select a product from the filtered results
3. Click "Get Recommendations" to see similar products
4. Adjust the slider to change the number of recommendations

### Analytics Dashboard

Run the data analytics dashboard:

```bash
streamlit run analytics.py
```

**How to use:**
1. Use the sidebar filters to explore different segments:
   - Filter by product category
   - Filter by specific brands
   - Adjust price range
2. Explore the various visualizations and insights
3. Scroll through different analysis sections

## 🛠 Technologies

- **Python 3.8+**
- **Streamlit** - Web app framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (TF-IDF, cosine similarity)
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **NLTK** - Natural language processing
- **WordCloud** - Word cloud generation (optional)

## 📊 Dataset

The project uses a cosmetics dataset (`cosmetics.csv`) with the following columns:

- **Label**: Product category (e.g., Moisturizer, Serum, Cleanser)
- **Brand**: Product brand name
- **Name**: Product name
- **Price**: Product price in USD
- **Rank**: Product rating (1-5 scale)
- **Ingredients**: Product ingredients list
- **Combination**: Compatibility with combination skin (1 = compatible, 0 = not)
- **Dry**: Compatibility with dry skin (1 = compatible, 0 = not)
- **Normal**: Compatibility with normal skin (1 = compatible, 0 = not)
- **Oily**: Compatibility with oily skin (1 = compatible, 0 = not)
- **Sensitive**: Compatibility with sensitive skin (1 = compatible, 0 = not)

## 📝 Notes

- The recommendation system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to analyze product ingredients and find similar products
- The analytics dashboard provides comprehensive insights into pricing, brands, categories, ratings, and skin type compatibility
- Both apps use `@st.cache_resource` and `@st.cache_data` for optimal performance
- The dataset should be placed in the `data/` folder or root directory

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available for educational purposes.

---

**Made with 💄 for beauty enthusiasts and data lovers!**
