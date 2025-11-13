# src/clean.py
import pandas as pd

def load_and_clean(path="data/cosmetics.csv"):
    df = pd.read_csv(path)

    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df.drop_duplicates(subset=["name", "brand"], inplace=True)

    text_cols = ["name", "brand", "category", "description", "ingredients"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[df["price"].notna()]

    df["combined_text"] = (
        df.get("description", "") + " " +
        df.get("ingredients", "") + " " +
        df.get("category", "")
    )

    return df

if __name__ == "__main__":
    df = load_and_clean()
    print(df.head())
    print("Rows:", len(df))
