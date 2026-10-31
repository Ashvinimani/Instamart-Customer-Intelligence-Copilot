
import os
import sqlite3
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_customers():
    df = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    df = df.drop_duplicates()

    if "city" in df.columns:
        df["city"] = df["city"].astype("string").str.strip().str.title()

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"].astype("string")
            .str.strip().str.lower()
            .replace({"m": "male", "f": "female", "": pd.NA})
        )

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df.loc[~df["age"].between(18, 100), "age"] = pd.NA

        if df["age"].notna().any():
            df["age"] = df["age"].fillna(df["age"].median())

    return df


def clean_orders():
    df = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    df = df.drop_duplicates()

    for col in ["quantity", "discount_pct",
                "order_amount", "delivery_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "quantity" in df.columns:
        df.loc[df["quantity"] < 0, "quantity"] = pd.NA

    if "order_amount" in df.columns:
        df.loc[df["order_amount"] < 0, "order_amount"] = pd.NA
        if df["order_amount"].notna().any():
            df["order_amount"] = df["order_amount"].fillna(
                df["order_amount"].median()
            )

    if "delivery_minutes" in df.columns:
        df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA
        if df["delivery_minutes"].notna().any():
            df["delivery_minutes"] = df["delivery_minutes"].fillna(
                df["delivery_minutes"].median()
            )

    return df


def clean_products():
    df = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
    df = df.drop_duplicates()

    for col in ["product_name", "category"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    if "category" in df.columns:
        df["category"] = df["category"].str.title()

    if "price_inr" in df.columns:
        df["price_inr"] = pd.to_numeric(
            df["price_inr"], errors="coerce"
        )

    return df


def clean_tickets():
    df = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"))
    df = df.drop_duplicates()

    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype("string").str.strip()
        df = df[df["text"].str.len() > 0].copy()

    return df


def clean_agent_scenarios():
    df = pd.read_csv(
        os.path.join(RAW_DIR, "agent_scenarios.csv")
    )
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def clean_policy_eval():
    df = pd.read_csv(os.path.join(RAW_DIR, "policy_eval.csv"))
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def main():

    datasets = {
        "customers": clean_customers(),
        "orders": clean_orders(),
        "products": clean_products(),
        "tickets": clean_tickets(),
        "agent_scenarios": clean_agent_scenarios(),
        "policy_eval": clean_policy_eval()
    }

    # Save cleaned CSV files
    for name, df in datasets.items():
        output_path = os.path.join(
            PROCESSED_DIR, name + "_clean.csv"
        )

        df.to_csv(output_path, index=False)

        print(name, "cleaned:", df.shape)

    # Create SQLite database
    db_path = os.path.join("data", "instamart.db")

    with sqlite3.connect(db_path) as connection:
        for name, df in datasets.items():
            df.to_sql(
                name,
                connection,
                if_exists="replace",
                index=False
            )

    print("All 6 datasets cleaned successfully!")
    print("SQLite database created successfully!")


if __name__ == "__main__":
    main()
