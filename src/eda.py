# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.clean import load_and_clean

def plot_price_distribution(df):
    sns.histplot(df["price"], bins=30, kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.show()

def plot_top_brands(df, n=10):
    top_brands = df["brand"].value_counts().head(n)
    sns.barplot(x=top_brands.values, y=top_brands.index)
    plt.title(f"Top {n} Brands by Number of Products")
    plt.xlabel("Count")
    plt.ylabel("Brand")
    plt.show()

def plot_categories(df):
    top_cats = df["category"].value_counts().head(10)
    sns.barplot(x=top_cats.values, y=top_cats.index)
    plt.title("Most Common Product Categories")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.show()

if __name__ == "__main__":
    df = load_and_clean()

    plot_price_distribution(df)
    plot_top_brands(df)
    plot_categories(df)
